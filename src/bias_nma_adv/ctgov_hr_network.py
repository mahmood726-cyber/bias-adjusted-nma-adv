"""ClinicalTrials.gov reported-HR network benchmark support."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import string
import tomllib
from typing import Any
from urllib.parse import urlparse

from bias_nma_adv.data import ValidationError
from bias_nma_adv.multiarm import ContrastRow, MultiArmNMAFit, fit_multiarm_gls
from bias_nma_adv.real_meta import sha256_file


CTGOV_HR_NETWORK_MANIFEST_SCHEMA_VERSION = "ctgov_hr_network_manifest/v1"
CTGOV_HR_NETWORK_VERIFICATION_SCHEMA_VERSION = "ctgov_hr_network_verification/v1"
NORMAL_975 = 1.959963984540054

_NCT_RE = re.compile(r"^NCT\d{8}$")


@dataclass(frozen=True)
class CTGovHRNetworkStudy:
    """One CT.gov reported-HR contrast for a network benchmark."""

    study_id: str
    trial: str
    nct_id: str
    outcome_id: str
    outcome_label: str
    active_drug: str
    analysis_treatment: str
    control_treatment: str
    effect_direction: str
    reported_hr: str
    ci_lower: str
    ci_upper: str
    source_type: str
    source_url: str
    access_statement: str
    evidence_mode: str
    source_terms: tuple[str, ...]
    outcome_search_terms: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovHRNetworkStudy":
        required = {
            "study_id",
            "trial",
            "nct_id",
            "outcome_id",
            "outcome_label",
            "active_drug",
            "analysis_treatment",
            "control_treatment",
            "effect_direction",
            "reported_hr",
            "ci_lower",
            "ci_upper",
            "source_type",
            "source_url",
            "access_statement",
            "evidence_mode",
            "source_terms",
            "outcome_search_terms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov HR network study missing required keys: {missing}")
        source_terms = raw["source_terms"]
        outcome_search_terms = raw["outcome_search_terms"]
        if not isinstance(source_terms, (list, tuple)):
            raise ValidationError("CT.gov HR network source_terms must be a list.")
        if not isinstance(outcome_search_terms, (list, tuple)):
            raise ValidationError("CT.gov HR network outcome_search_terms must be a list.")
        study = cls(
            study_id=str(raw["study_id"]),
            trial=str(raw["trial"]),
            nct_id=str(raw["nct_id"]),
            outcome_id=str(raw["outcome_id"]),
            outcome_label=str(raw["outcome_label"]),
            active_drug=str(raw["active_drug"]),
            analysis_treatment=str(raw["analysis_treatment"]),
            control_treatment=str(raw["control_treatment"]),
            effect_direction=str(raw["effect_direction"]),
            reported_hr=str(raw["reported_hr"]),
            ci_lower=str(raw["ci_lower"]),
            ci_upper=str(raw["ci_upper"]),
            source_type=str(raw["source_type"]),
            source_url=str(raw["source_url"]),
            access_statement=str(raw["access_statement"]),
            evidence_mode=str(raw["evidence_mode"]),
            source_terms=tuple(str(term) for term in source_terms),
            outcome_search_terms=tuple(str(term) for term in outcome_search_terms),
        )
        study.validate()
        return study

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("CT.gov HR network study_id must not be empty.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if self.source_type != "clinicaltrials_gov":
            raise ValidationError(f"{self.study_id}: source_type must be clinicaltrials_gov.")
        if self.evidence_mode != "reported_hr_clinicaltrials_gov_results":
            raise ValidationError(f"{self.study_id}: unsupported evidence_mode.")
        if self.effect_direction != "active_vs_control":
            raise ValidationError(f"{self.study_id}: unsupported effect_direction.")
        if self.analysis_treatment == self.control_treatment:
            raise ValidationError(f"{self.study_id}: analysis_treatment and control_treatment must differ.")
        if not self.active_drug.strip() or not self.analysis_treatment.strip() or not self.control_treatment.strip():
            raise ValidationError(f"{self.study_id}: treatment labels must not be empty.")
        if not self.source_terms or any(not term.strip() for term in self.source_terms):
            raise ValidationError(f"{self.study_id}: source_terms must be non-empty strings.")
        if not self.outcome_search_terms or any(not term.strip() for term in self.outcome_search_terms):
            raise ValidationError(f"{self.study_id}: outcome_search_terms must be non-empty strings.")
        parsed_url = urlparse(self.source_url)
        if parsed_url.hostname != "clinicaltrials.gov" or self.nct_id not in parsed_url.path:
            raise ValidationError(f"{self.study_id}: source_url must be a ClinicalTrials.gov URL containing the NCT ID.")
        if "clinicaltrials.gov" not in self.access_statement.lower():
            raise ValidationError(f"{self.study_id}: access_statement must identify ClinicalTrials.gov.")
        hr = _positive_decimal(self.reported_hr, self.study_id, "reported_hr")
        ci_low = _positive_decimal(self.ci_lower, self.study_id, "ci_lower")
        ci_high = _positive_decimal(self.ci_upper, self.study_id, "ci_upper")
        if ci_low > ci_high:
            raise ValidationError(f"{self.study_id}: CI lower bound exceeds upper bound.")
        if not (ci_low <= hr <= ci_high):
            raise ValidationError(f"{self.study_id}: reported HR is not contained in its confidence interval.")


@dataclass(frozen=True)
class CTGovHRNetworkManifest:
    """Source-bounded CT.gov reported-HR network manifest."""

    benchmark_id: str
    source_policy: str
    evidence_mode: str
    status: str
    certification_effect: str
    reference_treatment: str
    network_type: str
    effect_scale: str
    reuse_origin: str
    studies: tuple[CTGovHRNetworkStudy, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, manifest_sha256: str | None = None) -> "CTGovHRNetworkManifest":
        required = {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "evidence_mode",
            "status",
            "certification_effect",
            "reference_treatment",
            "network_type",
            "effect_scale",
            "reuse_origin",
            "studies",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov HR network manifest missing required keys: {missing}")
        if raw["schema_version"] != CTGOV_HR_NETWORK_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(
                "CT.gov HR network manifest schema_version must be "
                f"{CTGOV_HR_NETWORK_MANIFEST_SCHEMA_VERSION}."
            )
        studies = tuple(CTGovHRNetworkStudy.from_mapping(item) for item in raw["studies"])
        manifest = cls(
            benchmark_id=str(raw["benchmark_id"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            reference_treatment=str(raw["reference_treatment"]),
            network_type=str(raw["network_type"]),
            effect_scale=str(raw["effect_scale"]),
            reuse_origin=str(raw["reuse_origin"]),
            studies=studies,
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("CT.gov HR network manifest source_policy is outside the evidence boundary.")
        if self.evidence_mode != "reported_hr_clinicaltrials_gov_results":
            raise ValidationError("CT.gov HR network manifest evidence_mode is unsupported.")
        if self.status != "candidate_source_verified":
            raise ValidationError("CT.gov HR network manifest status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("CT.gov HR network manifests cannot certify model performance.")
        if self.network_type != "star_class_network":
            raise ValidationError("CT.gov HR network manifest network_type must be star_class_network.")
        if self.effect_scale != "log_hr":
            raise ValidationError("CT.gov HR network manifest effect_scale must be log_hr.")
        if self.reuse_origin != "complex_evidence_synthesis_map_pattern_only":
            raise ValidationError("CT.gov HR network manifest reuse_origin must avoid importing prior outputs.")
        if len(self.studies) < 2:
            raise ValidationError("CT.gov HR network manifest must contain at least two studies.")
        study_ids = [study.study_id for study in self.studies]
        duplicates = sorted({study_id for study_id in study_ids if study_ids.count(study_id) > 1})
        if duplicates:
            raise ValidationError(f"CT.gov HR network manifest contains duplicate study IDs: {duplicates}")
        treatments = {study.analysis_treatment for study in self.studies}
        if len(treatments) < 2:
            raise ValidationError("CT.gov HR network manifest must contain at least two active network treatments.")
        if any(study.control_treatment != self.reference_treatment for study in self.studies):
            raise ValidationError("all CT.gov HR network studies must use the manifest reference_treatment as control.")


@dataclass(frozen=True)
class CTGovHRNetworkVerificationRecord:
    """One CT.gov reported-HR analysis verification record."""

    study_id: str
    nct_id: str
    outcome_id: str
    evidence_scope: str
    response_sha256: str
    reported_hr: str
    ci_lower: str
    ci_upper: str
    source_terms: tuple[str, ...]
    outcome_search_terms: tuple[str, ...]
    nct_id_found: bool
    status_completed: bool
    hazard_ratio_analysis_found: bool
    ci_tokens_found: bool
    outcome_terms_found: bool
    source_terms_found: bool
    matched_outcome_title: str
    matched_param_type: str
    verified: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovHRNetworkVerificationRecord":
        required = {
            "study_id",
            "nct_id",
            "outcome_id",
            "evidence_scope",
            "response_sha256",
            "reported_hr",
            "ci_lower",
            "ci_upper",
            "source_terms",
            "outcome_search_terms",
            "nct_id_found",
            "status_completed",
            "hazard_ratio_analysis_found",
            "ci_tokens_found",
            "outcome_terms_found",
            "source_terms_found",
            "matched_outcome_title",
            "matched_param_type",
            "verified",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov HR verification record missing required keys: {missing}")
        source_terms = raw["source_terms"]
        outcome_search_terms = raw["outcome_search_terms"]
        if not isinstance(source_terms, (list, tuple)):
            raise ValidationError("CT.gov HR verification source_terms must be a list.")
        if not isinstance(outcome_search_terms, (list, tuple)):
            raise ValidationError("CT.gov HR verification outcome_search_terms must be a list.")
        record = cls(
            study_id=str(raw["study_id"]),
            nct_id=str(raw["nct_id"]),
            outcome_id=str(raw["outcome_id"]),
            evidence_scope=str(raw["evidence_scope"]),
            response_sha256=str(raw["response_sha256"]),
            reported_hr=str(raw["reported_hr"]),
            ci_lower=str(raw["ci_lower"]),
            ci_upper=str(raw["ci_upper"]),
            source_terms=tuple(str(term) for term in source_terms),
            outcome_search_terms=tuple(str(term) for term in outcome_search_terms),
            nct_id_found=bool(raw["nct_id_found"]),
            status_completed=bool(raw["status_completed"]),
            hazard_ratio_analysis_found=bool(raw["hazard_ratio_analysis_found"]),
            ci_tokens_found=bool(raw["ci_tokens_found"]),
            outcome_terms_found=bool(raw["outcome_terms_found"]),
            source_terms_found=bool(raw["source_terms_found"]),
            matched_outcome_title=str(raw["matched_outcome_title"]),
            matched_param_type=str(raw["matched_param_type"]),
            verified=bool(raw["verified"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if self.evidence_scope != "clinicaltrials_gov_reported_hr_analysis":
            raise ValidationError(f"{self.study_id}: unsupported evidence_scope.")
        if not _looks_like_sha256(self.response_sha256):
            raise ValidationError(f"{self.study_id}: response_sha256 is not a SHA-256 digest.")
        _positive_decimal(self.reported_hr, self.study_id, "reported_hr")
        _positive_decimal(self.ci_lower, self.study_id, "ci_lower")
        _positive_decimal(self.ci_upper, self.study_id, "ci_upper")
        if self.verified and not all(
            (
                self.nct_id_found,
                self.status_completed,
                self.hazard_ratio_analysis_found,
                self.ci_tokens_found,
                self.outcome_terms_found,
                self.source_terms_found,
            )
        ):
            raise ValidationError(f"{self.study_id}: verified CT.gov HR record is missing source evidence.")


@dataclass(frozen=True)
class CTGovHRNetworkVerificationReport:
    """CT.gov reported-HR verification snapshot."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[CTGovHRNetworkVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovHRNetworkVerificationReport":
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
            raise ValidationError(f"CT.gov HR verification report missing required keys: {missing}")
        if raw["schema_version"] != CTGOV_HR_NETWORK_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                "CT.gov HR verification schema_version must be "
                f"{CTGOV_HR_NETWORK_VERIFICATION_SCHEMA_VERSION}."
            )
        records = tuple(CTGovHRNetworkVerificationRecord.from_mapping(item) for item in raw["records"])
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            manifest=str(raw["manifest"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            records=records,
        )
        report.validate()
        return report

    def validate(self) -> None:
        if not _looks_like_sha256(self.manifest_sha256):
            raise ValidationError("CT.gov HR verification manifest_sha256 is not a SHA-256 digest.")
        if self.status not in {"verified", "failed"}:
            raise ValidationError(f"CT.gov HR verification status '{self.status}' is not supported.")
        if self.certification_effect != "none":
            raise ValidationError("CT.gov HR verification reports cannot certify model performance.")
        if not self.records:
            raise ValidationError("CT.gov HR verification report must contain records.")
        if self.status == "verified" and any(not record.verified for record in self.records):
            raise ValidationError("verified CT.gov HR report cannot contain unverified records.")


@dataclass(frozen=True)
class CTGovHRNetworkLogEffect:
    """One log-HR contrast derived from a verified CT.gov reported HR."""

    study_id: str
    trial: str
    nct_id: str
    outcome_id: str
    outcome_label: str
    active_drug: str
    analysis_treatment: str
    control_treatment: str
    effect_direction: str
    effect_scale: str
    reported_hr: float
    ci_lower: float
    ci_upper: float
    estimate: float
    variance: float
    se: float
    variance_source: str

    @classmethod
    def from_study(cls, study: CTGovHRNetworkStudy) -> "CTGovHRNetworkLogEffect":
        hr = float(study.reported_hr)
        ci_lower = float(study.ci_lower)
        ci_upper = float(study.ci_upper)
        estimate = math.log(hr)
        se = (math.log(ci_upper) - math.log(ci_lower)) / (2.0 * NORMAL_975)
        if se <= 0.0 or not math.isfinite(se):
            raise ValidationError(f"{study.study_id}: reported HR confidence interval yields invalid SE.")
        return cls(
            study_id=study.study_id,
            trial=study.trial,
            nct_id=study.nct_id,
            outcome_id=study.outcome_id,
            outcome_label=study.outcome_label,
            active_drug=study.active_drug,
            analysis_treatment=study.analysis_treatment,
            control_treatment=study.control_treatment,
            effect_direction=study.effect_direction,
            effect_scale="log_hr",
            reported_hr=hr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            estimate=estimate,
            variance=se * se,
            se=se,
            variance_source="ctgov_reported_95_ci_log_scale",
        )


def load_ctgov_hr_network_manifest(path: str | Path) -> CTGovHRNetworkManifest:
    """Load and validate one CT.gov reported-HR network manifest."""

    path = Path(path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return CTGovHRNetworkManifest.from_mapping(payload, manifest_sha256=sha256_file(path))


def load_ctgov_hr_network_verification_report(path: str | Path) -> CTGovHRNetworkVerificationReport:
    """Load and validate one CT.gov HR verification report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CTGovHRNetworkVerificationReport.from_mapping(payload)


def validate_ctgov_hr_network_source_bundle(
    manifest: CTGovHRNetworkManifest,
    report: CTGovHRNetworkVerificationReport,
) -> dict[str, Any]:
    """Validate that a CT.gov HR network manifest is backed by a source report."""

    if report.benchmark_id != manifest.benchmark_id:
        raise ValidationError("CT.gov HR verification benchmark_id does not match manifest.")
    if manifest.manifest_sha256 is not None and report.manifest_sha256 != manifest.manifest_sha256:
        raise ValidationError("CT.gov HR verification manifest_sha256 does not match manifest file.")
    if report.certification_effect != "none":
        raise ValidationError("CT.gov HR source verification cannot certify model performance.")
    if report.status != "verified":
        raise ValidationError("CT.gov HR source verification must be verified before benchmarking.")
    report_by_study = {record.study_id: record for record in report.records}
    manifest_ids = {study.study_id for study in manifest.studies}
    if set(report_by_study) != manifest_ids:
        raise ValidationError("CT.gov HR verification studies do not match manifest studies.")

    for study in manifest.studies:
        record = report_by_study[study.study_id]
        expected = (
            study.nct_id,
            study.outcome_id,
            study.reported_hr,
            study.ci_lower,
            study.ci_upper,
            study.source_terms,
            study.outcome_search_terms,
        )
        observed = (
            record.nct_id,
            record.outcome_id,
            record.reported_hr,
            record.ci_lower,
            record.ci_upper,
            record.source_terms,
            record.outcome_search_terms,
        )
        if observed != expected:
            raise ValidationError(f"{study.study_id}: CT.gov HR verification record does not match manifest.")

    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": report.certification_effect,
        "n_studies": len(manifest.studies),
        "source_counts": {"clinicaltrials_gov": len(manifest.studies)},
    }


def ctgov_hr_network_log_effects(manifest: CTGovHRNetworkManifest) -> list[CTGovHRNetworkLogEffect]:
    """Derive study-level log-HR contrasts from a CT.gov HR manifest."""

    return [CTGovHRNetworkLogEffect.from_study(study) for study in manifest.studies]


def run_ctgov_hr_network_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
) -> dict[str, Any]:
    """Run deterministic fixed/random GLS NMA on verified CT.gov HR contrasts."""

    manifest = load_ctgov_hr_network_manifest(manifest_path)
    report = load_ctgov_hr_network_verification_report(verification_report_path)
    source_bundle = validate_ctgov_hr_network_source_bundle(manifest, report)
    effects = ctgov_hr_network_log_effects(manifest)
    rows = [
        ContrastRow(
            study=effect.study_id,
            t1=effect.control_treatment,
            t2=effect.analysis_treatment,
            est=effect.estimate,
            se=effect.se,
        )
        for effect in effects
    ]
    fixed = fit_multiarm_gls(rows, reference_treatment=manifest.reference_treatment, model="fixed")
    random = fit_multiarm_gls(rows, reference_treatment=manifest.reference_treatment, model="random")
    return {
        "schema_version": "ctgov_hr_network_benchmark/v1",
        "benchmark_id": manifest.benchmark_id,
        "status": "local_pass",
        "certification_effect": "none",
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "network_type": manifest.network_type,
        "effect_scale": manifest.effect_scale,
        "reference_treatment": manifest.reference_treatment,
        "source_manifest": _path_as_posix(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_verification_report": _path_as_posix(verification_report_path),
        "source_verification_report_sha256": sha256_file(verification_report_path),
        "source_bundle": source_bundle,
        "model_config": {
            "fixed_effect_method": "contrast_gls",
            "random_effect_method": "generalized_dl",
            "level": 0.95,
            "reference_treatment": manifest.reference_treatment,
        },
        "n_studies": len(effects),
        "n_treatments": len({manifest.reference_treatment} | {effect.analysis_treatment for effect in effects}),
        "study_effects": [asdict(effect) for effect in effects],
        "candidate": {
            "fixed_gls": _fit_payload(fixed),
            "random_gls": _fit_payload(random),
        },
        "limitations": [
            "ClinicalTrials.gov reported HR and CI values are verified, but no external NMA package parity is claimed",
            "the network is a placebo-centered star, so closed-loop inconsistency cannot be assessed",
            "class labels are analyst-defined treatment groupings and are not clinical superiority claims",
            "source verification does not certify model performance or tier-one parity",
        ],
    }


def _fit_payload(fit: MultiArmNMAFit) -> dict[str, Any]:
    effects: list[dict[str, Any]] = []
    for treatment in fit.nonreference_treatments:
        estimate, se = fit.effect_vs_reference(treatment)
        ci_low = estimate - NORMAL_975 * se
        ci_high = estimate + NORMAL_975 * se
        effects.append(
            {
                "treatment": treatment,
                "reference_treatment": fit.reference_treatment,
                "estimate": estimate,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "hr": math.exp(estimate),
                "hr_ci_low": math.exp(ci_low),
                "hr_ci_high": math.exp(ci_high),
            }
        )
    return {
        "model": fit.model,
        "reference_treatment": fit.reference_treatment,
        "treatments": list(fit.treatments),
        "nonreference_treatments": list(fit.nonreference_treatments),
        "tau2": fit.tau2,
        "q": fit.q if fit.q is not None else "",
        "df": fit.df if fit.df is not None else "",
        "multi_arm_studies": list(fit.multi_arm_studies),
        "warnings": list(fit.warnings),
        "effects": effects,
    }


def _path_as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _positive_decimal(value: str, study_id: str, field_name: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"{study_id}: {field_name} must be a decimal string.") from exc
    if parsed <= 0:
        raise ValidationError(f"{study_id}: {field_name} must be positive.")
    return parsed


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)
