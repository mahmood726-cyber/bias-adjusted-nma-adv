"""Fail-closed screening for open-access Kaplan-Meier reconstruction inputs."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
import string
import tomllib
from typing import Any
from urllib.parse import urlparse

import numpy as np

from bias_nma_adv.data import ValidationError
from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources


KM_RECONSTRUCTION_POLICY_SCHEMA_VERSION = "km_reconstruction_policy/v1"
KM_RECONSTRUCTION_SCREEN_SCHEMA_VERSION = "km_reconstruction_screen/v1"
KM_CURVE_FIDELITY_SCHEMA_VERSION = "km_curve_fidelity/v1"

_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass(frozen=True)
class KMReconstructionPolicy:
    """Static policy for accepting OA Kaplan-Meier reconstruction results."""

    source_policy: str
    certification_effect: str
    reuse_origin: str
    required_source_type: str
    synthetic_ipd_policy: str
    min_curve_points: int
    required_source_fields: tuple[str, ...]
    blocked_hr_methods: tuple[str, ...]
    blocked_orientation_methods: tuple[str, ...]
    blocked_warning_terms: tuple[str, ...]
    required_result_fields: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "KMReconstructionPolicy":
        required = {
            "schema_version",
            "source_policy",
            "certification_effect",
            "reuse_origin",
            "required_source_type",
            "synthetic_ipd_policy",
            "min_curve_points",
            "required_source_fields",
            "blocked_hr_methods",
            "blocked_orientation_methods",
            "blocked_warning_terms",
            "required_result_fields",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"KM reconstruction policy missing required keys: {missing}")
        if raw["schema_version"] != KM_RECONSTRUCTION_POLICY_SCHEMA_VERSION:
            raise ValidationError(
                f"KM reconstruction policy schema_version must be {KM_RECONSTRUCTION_POLICY_SCHEMA_VERSION}."
            )
        policy = cls(
            source_policy=str(raw["source_policy"]),
            certification_effect=str(raw["certification_effect"]),
            reuse_origin=str(raw["reuse_origin"]),
            required_source_type=str(raw["required_source_type"]),
            synthetic_ipd_policy=str(raw["synthetic_ipd_policy"]),
            min_curve_points=int(raw["min_curve_points"]),
            required_source_fields=tuple(str(item) for item in raw["required_source_fields"]),
            blocked_hr_methods=tuple(str(item) for item in raw["blocked_hr_methods"]),
            blocked_orientation_methods=tuple(str(item) for item in raw["blocked_orientation_methods"]),
            blocked_warning_terms=tuple(str(item) for item in raw["blocked_warning_terms"]),
            required_result_fields=tuple(str(item) for item in raw["required_result_fields"]),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("KM reconstruction policy is outside the project evidence boundary.")
        if self.certification_effect != "none":
            raise ValidationError("KM reconstruction policy cannot certify model performance.")
        if self.reuse_origin != "wasserstein_method_pattern_only":
            raise ValidationError("KM reconstruction policy must not import uncertified wasserstein outputs.")
        if self.required_source_type != "open_access_paper":
            raise ValidationError("KM reconstruction requires an open-access paper source.")
        if self.synthetic_ipd_policy != "blocked":
            raise ValidationError("Synthetic IPD must be blocked for KM validation evidence.")
        if self.min_curve_points < 2:
            raise ValidationError("KM reconstruction min_curve_points must be at least 2.")
        if not self.required_source_fields:
            raise ValidationError("KM reconstruction policy must declare required source fields.")
        if not self.blocked_hr_methods:
            raise ValidationError("KM reconstruction policy must declare blocked HR methods.")
        if not self.blocked_warning_terms:
            raise ValidationError("KM reconstruction policy must declare blocked warning terms.")
        if not self.required_result_fields:
            raise ValidationError("KM reconstruction policy must declare required result fields.")


@dataclass(frozen=True)
class KMPaperSource:
    """One open-access paper and figure locator for a KM reconstruction target."""

    study_id: str
    trial: str
    nct_id: str
    pmid: str
    source_type: str
    source_url: str
    access_statement: str
    source_snapshot_sha256: str
    article_identity_status: str
    open_access_status: str
    figure_label: str
    figure_page: str
    outcome_label: str
    active_treatment: str
    control_treatment: str
    risk_table_status: str
    reuse_origin: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "KMPaperSource":
        required = {
            "study_id",
            "trial",
            "nct_id",
            "pmid",
            "source_type",
            "source_url",
            "access_statement",
            "source_snapshot_sha256",
            "article_identity_status",
            "open_access_status",
            "figure_label",
            "figure_page",
            "outcome_label",
            "active_treatment",
            "control_treatment",
            "risk_table_status",
            "reuse_origin",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"KM paper source missing required keys: {missing}")
        source = cls(**{key: str(raw[key]) for key in required})
        source.validate()
        return source

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("KM paper source study_id must not be empty.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError(f"{self.study_id}: malformed PMID.")
        if self.source_type != "open_access_paper":
            raise ValidationError(f"{self.study_id}: KM source must be an open_access_paper.")
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValidationError(f"{self.study_id}: source_url must be an HTTP(S) URL.")
        validate_sources(
            [
                EvidenceSource(
                    source_type=self.source_type,
                    identifier=self.pmid,
                    url=self.source_url,
                    access_statement=self.access_statement,
                )
            ]
        )
        if not _looks_like_sha256(self.source_snapshot_sha256):
            raise ValidationError(f"{self.study_id}: source_snapshot_sha256 is not SHA-256.")
        if self.article_identity_status != "pmid_verified":
            raise ValidationError(f"{self.study_id}: article identity must be PMID-verified.")
        if self.open_access_status != "verified_open_access":
            raise ValidationError(f"{self.study_id}: open-access status must be verified.")
        if not all(
            item.strip()
            for item in (
                self.figure_label,
                self.figure_page,
                self.outcome_label,
                self.active_treatment,
                self.control_treatment,
            )
        ):
            raise ValidationError(f"{self.study_id}: KM source requires figure, outcome, and arm labels.")
        if self.active_treatment == self.control_treatment:
            raise ValidationError(f"{self.study_id}: active and control treatments must differ.")
        if self.risk_table_status not in {"present_verified", "absent_declared"}:
            raise ValidationError(f"{self.study_id}: unsupported risk_table_status.")
        if self.reuse_origin != "wasserstein_method_pattern_only":
            raise ValidationError(f"{self.study_id}: reuse_origin must avoid importing uncertified outputs.")


def load_km_reconstruction_policy(path: str | Path) -> KMReconstructionPolicy:
    """Load a KM reconstruction policy TOML file."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return KMReconstructionPolicy.from_mapping(payload)


def screen_km_reconstruction_result(
    source: KMPaperSource | dict[str, Any],
    result: dict[str, Any],
    policy: KMReconstructionPolicy,
) -> dict[str, Any]:
    """Screen one extracted KM result before it can enter validation artifacts."""

    paper_source = source if isinstance(source, KMPaperSource) else KMPaperSource.from_mapping(source)
    policy.validate()
    missing = sorted(field for field in policy.required_result_fields if field not in result)
    if missing:
        raise ValidationError(f"{paper_source.study_id}: KM result missing required fields: {missing}")

    hr_method = str(result["hr_method"])
    orientation_method = str(result["orientation_method"])
    if hr_method in policy.blocked_hr_methods:
        raise ValidationError(f"{paper_source.study_id}: blocked KM HR method '{hr_method}'.")
    if orientation_method in policy.blocked_orientation_methods:
        raise ValidationError(f"{paper_source.study_id}: blocked KM orientation method '{orientation_method}'.")

    warnings = tuple(str(item) for item in result.get("warnings", ()))
    warning_text = "\n".join(warnings).lower()
    for term in policy.blocked_warning_terms:
        if term.lower() in warning_text:
            raise ValidationError(f"{paper_source.study_id}: KM result contains blocked warning term '{term}'.")

    curve1_n = _validate_curve(
        paper_source.study_id,
        "curve1",
        result["curve1_times"],
        result["curve1_survivals"],
        policy.min_curve_points,
    )
    curve2_n = _validate_curve(
        paper_source.study_id,
        "curve2",
        result["curve2_times"],
        result["curve2_survivals"],
        policy.min_curve_points,
    )
    total_ipd_records = int(result["total_ipd_records"])
    if total_ipd_records <= 0:
        raise ValidationError(f"{paper_source.study_id}: KM result has no reconstructed IPD records.")
    n_curves_found = int(result["n_curves_found"])
    if n_curves_found < 2:
        raise ValidationError(f"{paper_source.study_id}: KM result found fewer than two curves.")

    verification_level = str(result["verification_level"])
    if verification_level in {"", "unchecked"}:
        raise ValidationError(f"{paper_source.study_id}: KM result has no verification level.")
    verification_hash = str(result["verification_hash"])
    if not _looks_like_sha256(verification_hash):
        raise ValidationError(f"{paper_source.study_id}: KM result verification_hash is not SHA-256.")
    hr = float(result["hr"])
    if not math.isfinite(hr) or hr <= 0.0:
        raise ValidationError(f"{paper_source.study_id}: KM result HR must be positive and finite.")

    return {
        "schema_version": KM_RECONSTRUCTION_SCREEN_SCHEMA_VERSION,
        "study_id": paper_source.study_id,
        "status": "eligible_local_reconstruction",
        "certification_effect": "none",
        "source": asdict(paper_source),
        "extraction": {
            "hr": hr,
            "hr_method": hr_method,
            "orientation_method": orientation_method,
            "total_ipd_records": total_ipd_records,
            "curve1_points": curve1_n,
            "curve2_points": curve2_n,
            "n_curves_found": n_curves_found,
            "verification_level": verification_level,
            "verification_hash": verification_hash,
        },
    }


def km_from_ipd(times: Any, events: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute a right-continuous Kaplan-Meier step curve from IPD-like arrays."""

    times_array = np.asarray(times, dtype=float)
    event_values = np.asarray(events, dtype=float)
    if times_array.shape != event_values.shape:
        raise ValidationError("KM IPD times and events must have the same shape.")
    if times_array.ndim != 1:
        raise ValidationError("KM IPD times and events must be one-dimensional.")
    if times_array.size == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    if np.any(~np.isfinite(times_array)) or np.any(times_array < 0.0):
        raise ValidationError("KM IPD times must be finite and non-negative.")
    if np.any(~np.isfinite(event_values)) or not set(np.unique(event_values)).issubset({0.0, 1.0}):
        raise ValidationError("KM IPD events must be coded as 0/1.")
    events_array = event_values.astype(int)

    order = np.argsort(times_array, kind="mergesort")
    times_array = times_array[order]
    events_array = events_array[order]
    n_risk = times_array.size
    survival = 1.0
    event_times = [0.0]
    survivals = [1.0]
    for time in np.unique(times_array):
        at_time = times_array == time
        n_events = int(events_array[at_time].sum())
        n_censored = int(at_time.sum() - n_events)
        if n_risk > 0 and n_events > 0:
            survival *= 1.0 - n_events / n_risk
        event_times.append(float(time))
        survivals.append(float(survival))
        n_risk -= n_events + n_censored
    return np.asarray(event_times), np.asarray(survivals)


def evaluate_km_step(event_times: Any, survivals: Any, grid: Any) -> np.ndarray:
    """Evaluate a right-continuous KM step curve on a grid."""

    times_array, survival_array = _validated_km_curve_arrays(
        study_id="km_step",
        label="curve",
        raw_times=event_times,
        raw_survivals=survivals,
        min_curve_points=1,
    )
    grid_array = np.asarray(grid, dtype=float)
    if np.any(~np.isfinite(grid_array)) or np.any(grid_array < 0.0):
        raise ValidationError("KM evaluation grid must be finite and non-negative.")
    index = np.searchsorted(times_array, grid_array, side="right") - 1
    index = np.clip(index, 0, len(survival_array) - 1)
    return survival_array[index]


def restricted_mean_survival_time(event_times: Any, survivals: Any, tau: float) -> float:
    """Compute RMST by integrating a KM step curve over [0, tau]."""

    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValidationError("RMST tau must be positive and finite.")
    grid = np.linspace(0.0, tau, 2001)
    survival = evaluate_km_step(event_times, survivals, grid)
    return float(np.trapezoid(survival, grid))


def compare_km_curves(
    reference_times: Any,
    reference_survivals: Any,
    reconstructed_times: Any,
    reconstructed_survivals: Any,
    *,
    tau: float | None = None,
    grid_points: int = 100,
) -> dict[str, Any]:
    """Compare two externally oriented KM curves without choosing the orientation."""

    ref_times, ref_survivals = _validated_km_curve_arrays(
        study_id="reference",
        label="curve",
        raw_times=reference_times,
        raw_survivals=reference_survivals,
        min_curve_points=2,
    )
    recon_times, recon_survivals = _validated_km_curve_arrays(
        study_id="reconstructed",
        label="curve",
        raw_times=reconstructed_times,
        raw_survivals=reconstructed_survivals,
        min_curve_points=2,
    )
    if grid_points < 2:
        raise ValidationError("KM curve comparison requires at least two grid points.")
    if tau is None:
        tau = float(min(ref_times[-1], recon_times[-1]))
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValidationError("KM curve comparison tau must be positive and finite.")
    if tau > ref_times[-1] or tau > recon_times[-1]:
        raise ValidationError("KM curve comparison tau must stay within both observed curves.")

    grid = np.linspace(0.0, tau, grid_points)
    ref_curve = evaluate_km_step(ref_times, ref_survivals, grid)
    recon_curve = evaluate_km_step(recon_times, recon_survivals, grid)
    delta = np.abs(recon_curve - ref_curve)
    ref_rmst = restricted_mean_survival_time(ref_times, ref_survivals, tau)
    recon_rmst = restricted_mean_survival_time(recon_times, recon_survivals, tau)
    return {
        "schema_version": KM_CURVE_FIDELITY_SCHEMA_VERSION,
        "certification_effect": "none",
        "tau": tau,
        "grid_points": grid_points,
        "ae_median": float(np.median(delta)),
        "iae": float(np.trapezoid(delta, grid) / tau),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "ks": float(np.max(delta)),
        "rmst_abs_error": float(abs(recon_rmst - ref_rmst)),
        "rmst_relative_error": float(abs(recon_rmst - ref_rmst) / max(ref_rmst, 1e-12)),
    }


def _validate_curve(
    study_id: str,
    label: str,
    raw_times: Any,
    raw_survivals: Any,
    min_curve_points: int,
) -> int:
    if not isinstance(raw_times, list) or not isinstance(raw_survivals, list):
        raise ValidationError(f"{study_id}: {label} times and survivals must be lists.")
    if len(raw_times) != len(raw_survivals):
        raise ValidationError(f"{study_id}: {label} times and survivals have different lengths.")
    if len(raw_times) < min_curve_points:
        raise ValidationError(f"{study_id}: {label} has fewer than {min_curve_points} points.")

    previous_time: float | None = None
    previous_survival: float | None = None
    for time_raw, survival_raw in zip(raw_times, raw_survivals):
        time = float(time_raw)
        survival = float(survival_raw)
        if not math.isfinite(time) or time < 0.0:
            raise ValidationError(f"{study_id}: {label} contains an invalid time value.")
        if previous_time is not None and time < previous_time:
            raise ValidationError(f"{study_id}: {label} times must be nondecreasing.")
        if not math.isfinite(survival) or not 0.0 <= survival <= 1.0:
            raise ValidationError(f"{study_id}: {label} contains an invalid survival probability.")
        if previous_survival is not None and survival > previous_survival + 1e-12:
            raise ValidationError(f"{study_id}: {label} survival probabilities must be nonincreasing.")
        previous_time = time
        previous_survival = survival
    return len(raw_times)


def _validated_km_curve_arrays(
    *,
    study_id: str,
    label: str,
    raw_times: Any,
    raw_survivals: Any,
    min_curve_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        times_list = list(raw_times)
        survivals_list = list(raw_survivals)
    except TypeError as exc:
        raise ValidationError(f"{study_id}: {label} times and survivals must be iterable.") from exc
    _validate_curve(study_id, label, times_list, survivals_list, min_curve_points)
    return np.asarray(times_list, dtype=float), np.asarray(survivals_list, dtype=float)


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)
