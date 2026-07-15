"""Source-backed survival benchmark manifests and verification reports."""

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

import numpy as np

from bias_nma_adv.data import ValidationError
from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources
from bias_nma_adv.pairwise import PairwiseMetaResult, fit_pairwise_meta
from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.source_verification import SourceVerificationReport
from bias_nma_adv.source_verification import load_source_verification_report
from bias_nma_adv.source_verification import summarize_source_verification


SURVIVAL_HR_MANIFEST_SCHEMA_VERSION = "survival_hr_manifest/v1"
SURVIVAL_HR_VERIFICATION_SCHEMA_VERSION = "survival_hr_verification/v1"
NORMAL_975 = 1.959963984540054

_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass(frozen=True)
class SurvivalHRStudy:
    """One source-backed reported hazard-ratio study target."""

    study_id: str
    trial: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    active_treatment: str
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
    km_reconstruction_status: str
    reuse_origin: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SurvivalHRStudy":
        required = {
            "study_id",
            "trial",
            "nct_id",
            "pmid",
            "outcome_id",
            "outcome_label",
            "active_treatment",
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
            "km_reconstruction_status",
            "reuse_origin",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"survival HR study missing required keys: {missing}")
        source_terms = raw["source_terms"]
        if not isinstance(source_terms, (list, tuple)):
            raise ValidationError("survival HR source_terms must be a list.")
        study = cls(
            study_id=str(raw["study_id"]),
            trial=str(raw["trial"]),
            nct_id=str(raw["nct_id"]),
            pmid=str(raw["pmid"]),
            outcome_id=str(raw["outcome_id"]),
            outcome_label=str(raw["outcome_label"]),
            active_treatment=str(raw["active_treatment"]),
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
            km_reconstruction_status=str(raw["km_reconstruction_status"]),
            reuse_origin=str(raw["reuse_origin"]),
        )
        study.validate()
        return study

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("survival HR study_id must not be empty.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError(f"{self.study_id}: malformed PMID.")
        if self.effect_direction != "active_vs_control":
            raise ValidationError(f"{self.study_id}: unsupported effect_direction.")
        if self.evidence_mode != "reported_hr_pubmed_abstract":
            raise ValidationError(f"{self.study_id}: unsupported survival HR evidence_mode.")
        if self.source_type != "pubmed_abstract":
            raise ValidationError(f"{self.study_id}: reported HR manifest currently requires PubMed abstracts.")
        if not self.source_terms or any(not term.strip() for term in self.source_terms):
            raise ValidationError(f"{self.study_id}: source_terms must be non-empty strings.")
        if self.km_reconstruction_status != "not_digitized":
            raise ValidationError(
                f"{self.study_id}: KM reconstruction cannot be marked complete in a PubMed-only HR manifest."
            )
        if self.reuse_origin != "wasserstein_method_pattern_only":
            raise ValidationError(f"{self.study_id}: reuse_origin must avoid importing uncertified outputs.")

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
        parsed_url = urlparse(self.source_url)
        if parsed_url.hostname != "pubmed.ncbi.nlm.nih.gov" or self.pmid not in parsed_url.path:
            raise ValidationError(f"{self.study_id}: PubMed source URL must use pubmed.ncbi.nlm.nih.gov and contain PMID.")
        hr = _positive_decimal(self.reported_hr, self.study_id, "reported_hr")
        ci_low = _positive_decimal(self.ci_lower, self.study_id, "ci_lower")
        ci_high = _positive_decimal(self.ci_upper, self.study_id, "ci_upper")
        if ci_low > ci_high:
            raise ValidationError(f"{self.study_id}: CI lower bound exceeds upper bound.")
        if not (ci_low <= hr <= ci_high):
            raise ValidationError(f"{self.study_id}: reported HR is not contained in its confidence interval.")


@dataclass(frozen=True)
class SurvivalHRLogEffect:
    """One study-level log-HR effect derived from a source-verified reported HR."""

    study_id: str
    trial: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    active_treatment: str
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
    def from_study(cls, study: SurvivalHRStudy) -> "SurvivalHRLogEffect":
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
            pmid=study.pmid,
            outcome_id=study.outcome_id,
            outcome_label=study.outcome_label,
            active_treatment=study.active_treatment,
            control_treatment=study.control_treatment,
            effect_direction=study.effect_direction,
            effect_scale="log_hr",
            reported_hr=hr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            estimate=estimate,
            variance=se * se,
            se=se,
            variance_source="reported_95_ci_log_scale",
        )


@dataclass(frozen=True)
class SurvivalHRManifest:
    """A survival reported-HR benchmark manifest."""

    benchmark_id: str
    source_policy: str
    evidence_mode: str
    status: str
    certification_effect: str
    studies: tuple[SurvivalHRStudy, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, manifest_sha256: str | None = None) -> "SurvivalHRManifest":
        required = {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "evidence_mode",
            "status",
            "certification_effect",
            "studies",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"survival HR manifest missing required keys: {missing}")
        if raw["schema_version"] != SURVIVAL_HR_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(f"survival HR manifest schema_version must be {SURVIVAL_HR_MANIFEST_SCHEMA_VERSION}.")
        studies = tuple(SurvivalHRStudy.from_mapping(item) for item in raw["studies"])
        manifest = cls(
            benchmark_id=str(raw["benchmark_id"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            studies=studies,
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.benchmark_id.strip():
            raise ValidationError("survival HR manifest benchmark_id must not be empty.")
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("survival HR manifest source_policy is outside the project evidence boundary.")
        if self.evidence_mode != "reported_hr_pubmed_abstract":
            raise ValidationError("survival HR manifest evidence_mode must be reported_hr_pubmed_abstract.")
        if self.status != "candidate_source_verified":
            raise ValidationError("survival HR manifest status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("survival HR manifests cannot certify model performance.")
        if not self.studies:
            raise ValidationError("survival HR manifest must contain at least one study.")
        study_ids = [study.study_id for study in self.studies]
        duplicates = sorted({study_id for study_id in study_ids if study_ids.count(study_id) > 1})
        if duplicates:
            raise ValidationError(f"survival HR manifest contains duplicate study IDs: {duplicates}")


@dataclass(frozen=True)
class SurvivalHRVerificationRecord:
    """One PubMed abstract verification record for a reported HR."""

    study_id: str
    pmid: str
    outcome_id: str
    evidence_scope: str
    abstract_sha256: str
    reported_hr: str
    ci_lower: str
    ci_upper: str
    source_terms: tuple[str, ...]
    hr_token_found: bool
    ci_lower_token_found: bool
    ci_upper_token_found: bool
    hazard_ratio_anchor_found: bool
    confidence_interval_anchor_found: bool
    tokens_near_hazard_ratio_anchor: bool
    source_terms_near_hazard_ratio_anchor: bool
    verified: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SurvivalHRVerificationRecord":
        required = {
            "study_id",
            "pmid",
            "outcome_id",
            "evidence_scope",
            "abstract_sha256",
            "reported_hr",
            "ci_lower",
            "ci_upper",
            "source_terms",
            "hr_token_found",
            "ci_lower_token_found",
            "ci_upper_token_found",
            "hazard_ratio_anchor_found",
            "confidence_interval_anchor_found",
            "tokens_near_hazard_ratio_anchor",
            "source_terms_near_hazard_ratio_anchor",
            "verified",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"survival HR verification record missing required keys: {missing}")
        source_terms = raw["source_terms"]
        if not isinstance(source_terms, (list, tuple)):
            raise ValidationError("survival HR verification source_terms must be a list.")
        record = cls(
            study_id=str(raw["study_id"]),
            pmid=str(raw["pmid"]),
            outcome_id=str(raw["outcome_id"]),
            evidence_scope=str(raw["evidence_scope"]),
            abstract_sha256=str(raw["abstract_sha256"]),
            reported_hr=str(raw["reported_hr"]),
            ci_lower=str(raw["ci_lower"]),
            ci_upper=str(raw["ci_upper"]),
            source_terms=tuple(str(term) for term in source_terms),
            hr_token_found=bool(raw["hr_token_found"]),
            ci_lower_token_found=bool(raw["ci_lower_token_found"]),
            ci_upper_token_found=bool(raw["ci_upper_token_found"]),
            hazard_ratio_anchor_found=bool(raw["hazard_ratio_anchor_found"]),
            confidence_interval_anchor_found=bool(raw["confidence_interval_anchor_found"]),
            tokens_near_hazard_ratio_anchor=bool(raw["tokens_near_hazard_ratio_anchor"]),
            source_terms_near_hazard_ratio_anchor=bool(raw["source_terms_near_hazard_ratio_anchor"]),
            verified=bool(raw["verified"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("survival HR verification study_id must not be empty.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError(f"{self.study_id}: malformed PMID.")
        if self.evidence_scope != "pubmed_abstract_reported_hr_tokens":
            raise ValidationError(f"{self.study_id}: unsupported evidence_scope '{self.evidence_scope}'.")
        if not _looks_like_sha256(self.abstract_sha256):
            raise ValidationError(f"{self.study_id}: abstract_sha256 is not a SHA-256 digest.")
        _positive_decimal(self.reported_hr, self.study_id, "reported_hr")
        _positive_decimal(self.ci_lower, self.study_id, "ci_lower")
        _positive_decimal(self.ci_upper, self.study_id, "ci_upper")
        if not self.source_terms or any(not term.strip() for term in self.source_terms):
            raise ValidationError(f"{self.study_id}: source_terms must be non-empty strings.")
        if self.verified and not all(
            (
                self.hr_token_found,
                self.ci_lower_token_found,
                self.ci_upper_token_found,
                self.hazard_ratio_anchor_found,
                self.confidence_interval_anchor_found,
                self.tokens_near_hazard_ratio_anchor,
                self.source_terms_near_hazard_ratio_anchor,
            )
        ):
            raise ValidationError(f"{self.study_id}: verified HR record is missing abstract token evidence.")


@dataclass(frozen=True)
class SurvivalHRVerificationReport:
    """One PubMed abstract reported-HR verification snapshot."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[SurvivalHRVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SurvivalHRVerificationReport":
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
            raise ValidationError(f"survival HR verification report missing required keys: {missing}")
        if raw["schema_version"] != SURVIVAL_HR_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                f"survival HR verification schema_version must be {SURVIVAL_HR_VERIFICATION_SCHEMA_VERSION}."
            )
        records = tuple(SurvivalHRVerificationRecord.from_mapping(item) for item in raw["records"])
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
        if not self.benchmark_id.strip():
            raise ValidationError("survival HR verification benchmark_id must not be empty.")
        if not self.checked_at.strip():
            raise ValidationError("survival HR verification checked_at must not be empty.")
        if not _looks_like_sha256(self.manifest_sha256):
            raise ValidationError("survival HR verification manifest_sha256 is not a SHA-256 digest.")
        if self.status not in {"verified", "failed"}:
            raise ValidationError(f"survival HR verification status '{self.status}' is not supported.")
        if self.certification_effect != "none":
            raise ValidationError("survival HR verification reports cannot certify model performance.")
        if not self.records:
            raise ValidationError("survival HR verification report must contain records.")
        if self.status == "verified" and any(not record.verified for record in self.records):
            raise ValidationError("verified survival HR report cannot contain unverified records.")


def load_survival_hr_manifest(path: str | Path) -> SurvivalHRManifest:
    """Load and validate one survival HR manifest."""

    path = Path(path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return SurvivalHRManifest.from_mapping(payload, manifest_sha256=sha256_file(path))


def load_survival_hr_verification_report(path: str | Path) -> SurvivalHRVerificationReport:
    """Load and validate one survival HR verification report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SurvivalHRVerificationReport.from_mapping(payload)


def validate_survival_hr_source_bundle(
    manifest: SurvivalHRManifest,
    report: SurvivalHRVerificationReport,
) -> dict[str, Any]:
    """Validate that a reported-HR manifest is backed by its source-token report."""

    if report.benchmark_id != manifest.benchmark_id:
        raise ValidationError("survival HR verification benchmark_id does not match manifest.")
    if manifest.manifest_sha256 is not None and report.manifest_sha256 != manifest.manifest_sha256:
        raise ValidationError("survival HR verification manifest_sha256 does not match manifest file.")
    if report.certification_effect != "none":
        raise ValidationError("survival HR source verification cannot certify model performance.")
    if report.status != "verified":
        raise ValidationError("survival HR source verification must be verified before benchmarking.")
    report_by_study = {record.study_id: record for record in report.records}
    manifest_ids = {study.study_id for study in manifest.studies}
    if set(report_by_study) != manifest_ids:
        raise ValidationError("survival HR verification studies do not match manifest studies.")

    for study in manifest.studies:
        record = report_by_study[study.study_id]
        if not record.verified:
            raise ValidationError(f"{study.study_id}: survival HR source record is not verified.")
        expected = (
            study.pmid,
            study.outcome_id,
            study.reported_hr,
            study.ci_lower,
            study.ci_upper,
        )
        observed = (
            record.pmid,
            record.outcome_id,
            record.reported_hr,
            record.ci_lower,
            record.ci_upper,
        )
        if observed != expected:
            raise ValidationError(f"{study.study_id}: survival HR verification record does not match manifest.")

    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": report.certification_effect,
        "n_studies": len(manifest.studies),
    }


def validate_survival_hr_identity_bundle(
    manifest: SurvivalHRManifest,
    report: SourceVerificationReport,
) -> dict[str, Any]:
    """Validate that survival HR entries are tied to CT.gov and PubMed identities."""

    if report.benchmark_id != manifest.benchmark_id:
        raise ValidationError("survival HR identity benchmark_id does not match manifest.")
    if manifest.manifest_sha256 is not None and report.source_manifest_sha256 != manifest.manifest_sha256:
        raise ValidationError("survival HR identity source_manifest_sha256 does not match manifest file.")
    if report.certification_effect != "none":
        raise ValidationError("survival HR identity verification cannot certify model performance.")
    if report.status != "verified":
        raise ValidationError("survival HR identity verification must be verified before benchmarking.")

    expected = {
        (study.study_id, "clinicaltrials_gov", study.nct_id)
        for study in manifest.studies
    } | {
        (study.study_id, "pubmed_abstract", study.pmid)
        for study in manifest.studies
    }
    observed = {
        (record.study_id, record.source_type, record.identifier)
        for record in report.records
    }
    if observed != expected:
        raise ValidationError("survival HR identity records do not match manifest studies.")

    for record in report.records:
        if record.source_type == "clinicaltrials_gov" and record.details.get("nct_id") != record.identifier:
            raise ValidationError(f"{record.study_id}: CT.gov identity details do not match identifier.")
        if record.source_type == "pubmed_abstract" and record.details.get("pmid") != record.identifier:
            raise ValidationError(f"{record.study_id}: PubMed identity details do not match identifier.")

    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": report.certification_effect,
        "source_counts": summarize_source_verification(report),
    }


def survival_hr_log_effects(manifest: SurvivalHRManifest) -> list[SurvivalHRLogEffect]:
    """Derive study-level log-HR effects from source-verified reported HR entries."""

    return [SurvivalHRLogEffect.from_study(study) for study in manifest.studies]


def run_survival_hr_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
    identity_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic pairwise pooling on source-verified reported survival HRs."""

    manifest = load_survival_hr_manifest(manifest_path)
    verification_report = load_survival_hr_verification_report(verification_report_path)
    source_bundle = validate_survival_hr_source_bundle(manifest, verification_report)
    identity_bundle = None
    if identity_report_path is not None:
        identity_report = load_source_verification_report(identity_report_path)
        identity_bundle = validate_survival_hr_identity_bundle(manifest, identity_report)
    effects = survival_hr_log_effects(manifest)
    estimates = np.asarray([effect.estimate for effect in effects], dtype=float)
    variances = np.asarray([effect.variance for effect in effects], dtype=float)
    fixed = fit_pairwise_meta(estimates, variances, method="FE")
    reml_hksj = fit_pairwise_meta(
        estimates,
        variances,
        method="REML",
        hksj=True,
        hksj_floor=True,
        prediction_interval=True,
    )
    return {
        "schema_version": "survival_hr_benchmark/v1",
        "benchmark_id": manifest.benchmark_id,
        "status": "local_pass",
        "certification_effect": "none",
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "effect_scale": "log_hr",
        "source_manifest": _path_as_posix(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_verification_report": _path_as_posix(verification_report_path),
        "source_verification_report_sha256": sha256_file(verification_report_path),
        "source_bundle": source_bundle,
        "source_identity_report": _path_as_posix(identity_report_path) if identity_report_path is not None else "",
        "source_identity_report_sha256": sha256_file(identity_report_path) if identity_report_path is not None else "",
        "source_identity_bundle": identity_bundle or {},
        "model_config": {
            "pairwise_fixed_effect_method": "FE",
            "pairwise_random_effect_method": "REML",
            "pairwise_hksj": True,
            "pairwise_hksj_floor": True,
            "pairwise_prediction_interval": True,
            "pairwise_prediction_interval_df": "k_minus_1",
            "level": 0.95,
        },
        "n_studies": len(effects),
        "study_effects": [asdict(effect) for effect in effects],
        "candidate": {
            "pairwise_fixed_effect": _pairwise_result_payload(fixed),
            "pairwise_reml_hksj": _pairwise_result_payload(reml_hksj),
        },
        "limitations": [
            "reported PubMed abstract HR tokens are verified, but Kaplan-Meier curves are not digitized",
            "this is a pairwise class meta-analysis, not a multi-treatment survival NMA",
            "source-token verification does not certify model performance or tier-one parity",
        ],
    }


def _pairwise_result_payload(result: PairwiseMetaResult) -> dict[str, Any]:
    payload = {
        "method": result.method,
        "estimate": result.estimate,
        "se": result.se,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "tau2": result.tau2,
        "q": result.q,
        "df": result.df,
        "hksj": result.hksj,
        "hksj_q_factor": result.hksj_q_factor,
        "weights": list(result.weights),
        "warnings": list(result.warnings),
    }
    if result.prediction_interval is not None:
        payload["pi_low"] = float(result.prediction_interval[0])
        payload["pi_high"] = float(result.prediction_interval[1])
    return payload


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
