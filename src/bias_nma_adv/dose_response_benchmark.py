"""Source-backed CT.gov dose-response benchmark support."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any
from urllib.parse import urlparse

import numpy as np

from bias_nma_adv.data import ValidationError
from bias_nma_adv.real_meta import sha256_file


DOSE_RESPONSE_MANIFEST_SCHEMA_VERSION = "dose_response_ctgov_manifest/v1"
DOSE_RESPONSE_VERIFICATION_SCHEMA_VERSION = "dose_response_source_verification/v1"
DOSE_RESPONSE_BENCHMARK_SCHEMA_VERSION = "dose_response_benchmark/v1"
_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass(frozen=True)
class DoseResponseArm:
    """One CT.gov-reported arm for a dose-response benchmark."""

    arm_id: str
    group_id: str
    treatment: str
    dose: float
    dose_unit: str
    dose_frequency: str
    escalation: str
    lsmean: float
    se: float
    source_terms: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DoseResponseArm":
        required = {
            "arm_id",
            "group_id",
            "treatment",
            "dose",
            "dose_unit",
            "dose_frequency",
            "escalation",
            "lsmean",
            "se",
            "source_terms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"dose-response arm missing required keys: {missing}")
        terms = raw["source_terms"]
        if not isinstance(terms, (list, tuple)):
            raise ValidationError("dose-response arm source_terms must be a list.")
        arm = cls(
            arm_id=str(raw["arm_id"]),
            group_id=str(raw["group_id"]),
            treatment=str(raw["treatment"]),
            dose=float(raw["dose"]),
            dose_unit=str(raw["dose_unit"]),
            dose_frequency=str(raw["dose_frequency"]),
            escalation=str(raw["escalation"]),
            lsmean=float(raw["lsmean"]),
            se=float(raw["se"]),
            source_terms=tuple(str(term) for term in terms),
        )
        arm.validate()
        return arm

    def validate(self) -> None:
        if not self.arm_id.strip() or not self.group_id.strip():
            raise ValidationError("dose-response arm identifiers must not be empty.")
        if not self.treatment.strip():
            raise ValidationError(f"{self.arm_id}: treatment must not be empty.")
        if self.dose < 0.0:
            raise ValidationError(f"{self.arm_id}: dose cannot be negative.")
        if not self.dose_unit.strip() or not self.dose_frequency.strip():
            raise ValidationError(f"{self.arm_id}: dose unit and frequency are required.")
        if self.se <= 0.0 or not math.isfinite(self.se):
            raise ValidationError(f"{self.arm_id}: standard error must be positive.")
        if not math.isfinite(self.lsmean):
            raise ValidationError(f"{self.arm_id}: LS mean must be finite.")
        if not self.source_terms or any(not term.strip() for term in self.source_terms):
            raise ValidationError(f"{self.arm_id}: source_terms must be non-empty strings.")


@dataclass(frozen=True)
class DoseResponseManifest:
    """Source-bounded CT.gov dose-response manifest."""

    benchmark_id: str
    source_policy: str
    evidence_mode: str
    status: str
    certification_effect: str
    trial: str
    nct_id: str
    pmid: str
    source_url: str
    pubmed_url: str
    access_statement: str
    outcome_id: str
    outcome_title: str
    outcome_param_type: str
    outcome_dispersion_type: str
    outcome_unit: str
    effect_scale: str
    reference_arm_id: str
    reuse_origin: str
    arms: tuple[DoseResponseArm, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        manifest_sha256: str | None = None,
    ) -> "DoseResponseManifest":
        required = {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "evidence_mode",
            "status",
            "certification_effect",
            "trial",
            "nct_id",
            "pmid",
            "source_url",
            "pubmed_url",
            "access_statement",
            "outcome_id",
            "outcome_title",
            "outcome_param_type",
            "outcome_dispersion_type",
            "outcome_unit",
            "effect_scale",
            "reference_arm_id",
            "reuse_origin",
            "arms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"dose-response manifest missing required keys: {missing}")
        if raw["schema_version"] != DOSE_RESPONSE_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(
                "dose-response manifest schema_version must be "
                f"{DOSE_RESPONSE_MANIFEST_SCHEMA_VERSION}."
            )
        manifest = cls(
            benchmark_id=str(raw["benchmark_id"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            trial=str(raw["trial"]),
            nct_id=str(raw["nct_id"]),
            pmid=str(raw["pmid"]),
            source_url=str(raw["source_url"]),
            pubmed_url=str(raw["pubmed_url"]),
            access_statement=str(raw["access_statement"]),
            outcome_id=str(raw["outcome_id"]),
            outcome_title=str(raw["outcome_title"]),
            outcome_param_type=str(raw["outcome_param_type"]),
            outcome_dispersion_type=str(raw["outcome_dispersion_type"]),
            outcome_unit=str(raw["outcome_unit"]),
            effect_scale=str(raw["effect_scale"]),
            reference_arm_id=str(raw["reference_arm_id"]),
            reuse_origin=str(raw["reuse_origin"]),
            arms=tuple(DoseResponseArm.from_mapping(item) for item in raw["arms"]),
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("dose-response source_policy is outside the evidence boundary.")
        if self.evidence_mode != "ctgov_dose_response_lsmean":
            raise ValidationError("dose-response evidence_mode is unsupported.")
        if self.status != "candidate_source_verified":
            raise ValidationError("dose-response manifest status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("dose-response manifests cannot certify model performance.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError("dose-response manifest has malformed NCT ID.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError("dose-response manifest has malformed PMID.")
        source_host = urlparse(self.source_url).hostname
        if source_host != "clinicaltrials.gov" or self.nct_id not in self.source_url:
            raise ValidationError("source_url must be a ClinicalTrials.gov URL containing the NCT ID.")
        pubmed_host = urlparse(self.pubmed_url).hostname
        if pubmed_host != "pubmed.ncbi.nlm.nih.gov" or self.pmid not in self.pubmed_url:
            raise ValidationError("pubmed_url must be a PubMed URL containing the PMID.")
        if "clinicaltrials.gov" not in self.access_statement.lower():
            raise ValidationError("access_statement must identify ClinicalTrials.gov.")
        if self.effect_scale != "percentage_point_change_vs_placebo":
            raise ValidationError("dose-response effect_scale is unsupported.")
        if self.reuse_origin != "source_backed_new_dose_response_lane":
            raise ValidationError("dose-response reuse_origin must describe a new source-backed lane.")
        if len(self.arms) < 3:
            raise ValidationError("dose-response manifest must include at least two active doses plus reference.")
        arm_ids = [arm.arm_id for arm in self.arms]
        if self.reference_arm_id not in arm_ids:
            raise ValidationError("reference_arm_id must match one manifest arm.")
        if len(set(arm_ids)) != len(arm_ids):
            raise ValidationError("dose-response arm IDs must be unique.")
        active = [arm for arm in self.arms if arm.arm_id != self.reference_arm_id]
        if any(arm.dose <= 0.0 for arm in active):
            raise ValidationError("active dose-response arms must have positive doses.")
        if len({arm.dose for arm in active}) < 2:
            raise ValidationError("dose-response manifest must include at least two active dose levels.")


@dataclass(frozen=True)
class DoseResponseVerificationRecord:
    """One source verification record for the dose-response benchmark."""

    source_type: str
    identifier: str
    evidence_scope: str
    response_sha256: str
    verified: bool
    details: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DoseResponseVerificationRecord":
        required = {
            "source_type",
            "identifier",
            "evidence_scope",
            "response_sha256",
            "verified",
            "details",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"dose-response verification record missing keys: {missing}")
        record = cls(
            source_type=str(raw["source_type"]),
            identifier=str(raw["identifier"]),
            evidence_scope=str(raw["evidence_scope"]),
            response_sha256=str(raw["response_sha256"]),
            verified=bool(raw["verified"]),
            details=dict(raw["details"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.source_type not in {"clinicaltrials_gov", "pubmed_abstract"}:
            raise ValidationError("dose-response verification source_type is unsupported.")
        if not self.identifier.strip():
            raise ValidationError("dose-response verification identifier must not be empty.")
        if len(self.response_sha256) != 64:
            raise ValidationError("dose-response verification response_sha256 must be SHA-256 length.")
        if not self.verified:
            raise ValidationError("dose-response verification record is not verified.")


@dataclass(frozen=True)
class DoseResponseVerificationReport:
    """Source verification report for a dose-response manifest."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[DoseResponseVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DoseResponseVerificationReport":
        required = {
            "schema_version",
            "benchmark_id",
            "checked_at",
            "manifest",
            "manifest_sha256",
            "status",
            "certification_effect",
            "records",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"dose-response verification report missing keys: {missing}")
        if raw["schema_version"] != DOSE_RESPONSE_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                "dose-response verification schema_version must be "
                f"{DOSE_RESPONSE_VERIFICATION_SCHEMA_VERSION}."
            )
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            manifest=str(raw["manifest"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            records=tuple(DoseResponseVerificationRecord.from_mapping(item) for item in raw["records"]),
        )
        report.validate()
        return report

    def validate(self) -> None:
        if self.status != "verified":
            raise ValidationError("dose-response verification report status must be verified.")
        if self.certification_effect != "none":
            raise ValidationError("dose-response verification report cannot certify.")
        if not self.records:
            raise ValidationError("dose-response verification report must include records.")
        source_types = {record.source_type for record in self.records}
        if not {"clinicaltrials_gov", "pubmed_abstract"} <= source_types:
            raise ValidationError("dose-response verification requires CT.gov and PubMed records.")


@dataclass(frozen=True)
class DoseResponseEffect:
    """One dose-level effect versus the reference arm."""

    study_id: str
    trial: str
    nct_id: str
    pmid: str
    arm_id: str
    group_id: str
    treatment: str
    dose: float
    dose_unit: str
    dose_frequency: str
    escalation: str
    active_lsmean: float
    active_se: float
    reference_arm_id: str
    reference_lsmean: float
    reference_se: float
    estimate: float
    se: float
    variance: float
    effect_scale: str
    variance_source: str


def load_dose_response_manifest(path: str | Path) -> DoseResponseManifest:
    manifest_path = Path(path)
    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return DoseResponseManifest.from_mapping(payload, manifest_sha256=sha256_file(manifest_path))


def load_dose_response_verification_report(path: str | Path) -> DoseResponseVerificationReport:
    return DoseResponseVerificationReport.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def validate_dose_response_source_bundle(
    manifest: DoseResponseManifest,
    report: DoseResponseVerificationReport,
) -> dict[str, Any]:
    if manifest.benchmark_id != report.benchmark_id:
        raise ValidationError("dose-response verification benchmark_id mismatch.")
    if manifest.manifest_sha256 != report.manifest_sha256:
        raise ValidationError("dose-response verification manifest SHA mismatch.")
    if report.status != "verified":
        raise ValidationError("dose-response source bundle is not verified.")
    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": "none",
        "n_records": len(report.records),
        "source_counts": _count_by(record.source_type for record in report.records),
    }


def dose_response_effects(manifest: DoseResponseManifest) -> tuple[DoseResponseEffect, ...]:
    reference = next(arm for arm in manifest.arms if arm.arm_id == manifest.reference_arm_id)
    effects: list[DoseResponseEffect] = []
    for arm in manifest.arms:
        if arm.arm_id == manifest.reference_arm_id:
            continue
        estimate = arm.lsmean - reference.lsmean
        variance = arm.se * arm.se + reference.se * reference.se
        effects.append(
            DoseResponseEffect(
                study_id=f"{manifest.nct_id}_{arm.arm_id}",
                trial=manifest.trial,
                nct_id=manifest.nct_id,
                pmid=manifest.pmid,
                arm_id=arm.arm_id,
                group_id=arm.group_id,
                treatment=arm.treatment,
                dose=_stable_float(arm.dose),
                dose_unit=arm.dose_unit,
                dose_frequency=arm.dose_frequency,
                escalation=arm.escalation,
                active_lsmean=_stable_float(arm.lsmean),
                active_se=_stable_float(arm.se),
                reference_arm_id=reference.arm_id,
                reference_lsmean=_stable_float(reference.lsmean),
                reference_se=_stable_float(reference.se),
                estimate=_stable_float(estimate),
                se=_stable_float(math.sqrt(variance)),
                variance=_stable_float(variance),
                effect_scale=manifest.effect_scale,
                variance_source="sqrt(active_se^2 + placebo_pool_se^2); shared placebo covariance not modeled",
            )
        )
    return tuple(effects)


def run_dose_response_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
) -> dict[str, Any]:
    manifest = load_dose_response_manifest(manifest_path)
    report = load_dose_response_verification_report(verification_report_path)
    source_bundle = validate_dose_response_source_bundle(manifest, report)
    effects = dose_response_effects(manifest)
    linear = _weighted_polynomial_fit(effects, degree=1)
    quadratic = _weighted_polynomial_fit(effects, degree=2)
    root = Path.cwd()
    manifest_rel = _relpath(Path(manifest_path), root)
    report_rel = _relpath(Path(verification_report_path), root)
    return {
        "schema_version": DOSE_RESPONSE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": manifest.benchmark_id,
        "status": "local_pass",
        "certification_effect": "none",
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "effect_scale": manifest.effect_scale,
        "source_manifest": manifest_rel,
        "source_manifest_sha256": manifest.manifest_sha256,
        "source_verification_report": report_rel,
        "source_verification_report_sha256": sha256_file(verification_report_path),
        "n_studies": 1,
        "n_dose_effects": len(effects),
        "limitations": [
            "single dose-ranging trial only",
            "shared placebo covariance not modeled",
            "not MBNMAdose reference matched",
            "not clinical superiority claims",
            "does not certify model performance",
        ],
        "source_bundle": source_bundle,
        "model_config": {
            "candidate_models": ["weighted_linear", "weighted_quadratic"],
            "reference_arm_id": manifest.reference_arm_id,
            "dose_unit": effects[0].dose_unit if effects else "",
            "dose_frequency": effects[0].dose_frequency if effects else "",
        },
        "study_effects": [_effect_to_dict(effect) for effect in effects],
        "candidate": {
            "weighted_linear": linear,
            "weighted_quadratic": quadratic,
        },
    }


def _weighted_polynomial_fit(effects: tuple[DoseResponseEffect, ...], *, degree: int) -> dict[str, Any]:
    doses = np.asarray([effect.dose for effect in effects], dtype=float)
    y = np.asarray([effect.estimate for effect in effects], dtype=float)
    variances = np.asarray([effect.variance for effect in effects], dtype=float)
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    design = np.column_stack([doses ** power for power in range(1, degree + 1)])
    weights = 1.0 / variances
    xtw = design.T * weights
    xtwx = xtw @ design
    beta = np.linalg.pinv(xtwx) @ xtw @ y
    fitted = design @ beta
    residual = y - fitted
    q = float(np.sum(weights * residual * residual))
    df = int(max(len(y) - design.shape[1], 0))
    covariance = np.linalg.pinv(xtwx)
    return {
        "model": f"weighted_polynomial_degree_{degree}_through_zero",
        "degree": int(degree),
        "coefficients": [_stable_float(item) for item in beta],
        "coefficient_ses": [
            _stable_float(math.sqrt(max(covariance[idx, idx], 0.0)))
            for idx in range(covariance.shape[0])
        ],
        "q": _stable_float(q),
        "df": df,
        "warnings": [
            "Local weighted polynomial fit is a source-backed smoke benchmark, not MBNMAdose parity.",
            "Shared placebo covariance is ignored in this simple artifact.",
        ],
        "fitted": [
            {
                "dose": _stable_float(dose),
                "observed": _stable_float(observed),
                "fitted": _stable_float(predicted),
                "residual": _stable_float(obs_residual),
            }
            for dose, observed, predicted, obs_residual in zip(doses, y, fitted, residual, strict=True)
        ],
    }


def _effect_to_dict(effect: DoseResponseEffect) -> dict[str, Any]:
    return {
        "study_id": effect.study_id,
        "trial": effect.trial,
        "nct_id": effect.nct_id,
        "pmid": effect.pmid,
        "arm_id": effect.arm_id,
        "group_id": effect.group_id,
        "treatment": effect.treatment,
        "dose": effect.dose,
        "dose_unit": effect.dose_unit,
        "dose_frequency": effect.dose_frequency,
        "escalation": effect.escalation,
        "active_lsmean": effect.active_lsmean,
        "active_se": effect.active_se,
        "reference_arm_id": effect.reference_arm_id,
        "reference_lsmean": effect.reference_lsmean,
        "reference_se": effect.reference_se,
        "estimate": effect.estimate,
        "se": effect.se,
        "variance": effect.variance,
        "effect_scale": effect.effect_scale,
        "variance_source": effect.variance_source,
    }


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_float(value: float) -> float:
    return float(f"{float(value):.12g}")
