"""Source-backed cross-design RCT/NRS benchmark support."""

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
from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources
from bias_nma_adv.real_meta import sha256_file


CROSS_DESIGN_HR_MANIFEST_SCHEMA_VERSION = "cross_design_hr_manifest/v1"
CROSS_DESIGN_HR_VERIFICATION_SCHEMA_VERSION = "cross_design_hr_verification/v1"
CROSS_DESIGN_HR_BENCHMARK_SCHEMA_VERSION = "cross_design_hr_benchmark/v1"
NORMAL_975 = 1.959963984540054

_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass(frozen=True)
class CrossDesignHRStudy:
    """One PubMed-reported HR row for a cross-design benchmark."""

    study_id: str
    trial: str
    design: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    target_population: str
    active_treatment: str
    control_treatment: str
    comparator_class: str
    effect_direction: str
    reported_hr: str
    ci_lower: str
    ci_upper: str
    source_type: str
    source_url: str
    access_statement: str
    evidence_mode: str
    source_terms: tuple[str, ...]
    estimand_tags: tuple[str, ...]
    reuse_origin: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CrossDesignHRStudy":
        required = {
            "study_id",
            "trial",
            "design",
            "nct_id",
            "pmid",
            "outcome_id",
            "outcome_label",
            "target_population",
            "active_treatment",
            "control_treatment",
            "comparator_class",
            "effect_direction",
            "reported_hr",
            "ci_lower",
            "ci_upper",
            "source_type",
            "source_url",
            "access_statement",
            "evidence_mode",
            "source_terms",
            "estimand_tags",
            "reuse_origin",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"cross-design HR study missing required keys: {missing}")
        source_terms = raw["source_terms"]
        estimand_tags = raw["estimand_tags"]
        if not isinstance(source_terms, (list, tuple)):
            raise ValidationError("cross-design HR source_terms must be a list.")
        if not isinstance(estimand_tags, (list, tuple)):
            raise ValidationError("cross-design HR estimand_tags must be a list.")
        study = cls(
            study_id=str(raw["study_id"]),
            trial=str(raw["trial"]),
            design=str(raw["design"]),
            nct_id=str(raw["nct_id"]),
            pmid=str(raw["pmid"]),
            outcome_id=str(raw["outcome_id"]),
            outcome_label=str(raw["outcome_label"]),
            target_population=str(raw["target_population"]),
            active_treatment=str(raw["active_treatment"]),
            control_treatment=str(raw["control_treatment"]),
            comparator_class=str(raw["comparator_class"]),
            effect_direction=str(raw["effect_direction"]),
            reported_hr=str(raw["reported_hr"]),
            ci_lower=str(raw["ci_lower"]),
            ci_upper=str(raw["ci_upper"]),
            source_type=str(raw["source_type"]),
            source_url=str(raw["source_url"]),
            access_statement=str(raw["access_statement"]),
            evidence_mode=str(raw["evidence_mode"]),
            source_terms=tuple(str(term) for term in source_terms),
            estimand_tags=tuple(str(term) for term in estimand_tags),
            reuse_origin=str(raw["reuse_origin"]),
        )
        study.validate()
        return study

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("cross-design HR study_id must not be empty.")
        if self.design not in {"rct", "nrs"}:
            raise ValidationError(f"{self.study_id}: design must be rct or nrs.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError(f"{self.study_id}: malformed PMID.")
        if self.source_type != "pubmed_abstract":
            raise ValidationError(f"{self.study_id}: source_type must be pubmed_abstract.")
        if self.evidence_mode != "reported_hr_pubmed_abstract_cross_design":
            raise ValidationError(f"{self.study_id}: unsupported evidence_mode.")
        if self.effect_direction != "active_vs_control":
            raise ValidationError(f"{self.study_id}: unsupported effect_direction.")
        if not self.source_terms or any(not term.strip() for term in self.source_terms):
            raise ValidationError(f"{self.study_id}: source_terms must be non-empty strings.")
        if not self.estimand_tags or any(not term.strip() for term in self.estimand_tags):
            raise ValidationError(f"{self.study_id}: estimand_tags must be non-empty strings.")
        if self.reuse_origin != "source_backed_new_cross_design_lane":
            raise ValidationError(f"{self.study_id}: unsupported reuse_origin.")
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
            raise ValidationError(f"{self.study_id}: source_url must be a PubMed URL containing the PMID.")
        hr = _positive_decimal(self.reported_hr, self.study_id, "reported_hr")
        ci_low = _positive_decimal(self.ci_lower, self.study_id, "ci_lower")
        ci_high = _positive_decimal(self.ci_upper, self.study_id, "ci_upper")
        if ci_low > ci_high:
            raise ValidationError(f"{self.study_id}: CI lower bound exceeds upper bound.")
        if not (ci_low <= hr <= ci_high):
            raise ValidationError(f"{self.study_id}: reported HR is not contained in its confidence interval.")


@dataclass(frozen=True)
class CrossDesignHRManifest:
    """Cross-design RCT/NRS PubMed-reported HR manifest."""

    benchmark_id: str
    source_policy: str
    evidence_mode: str
    status: str
    certification_effect: str
    effect_scale: str
    cross_design_strategy: str
    studies: tuple[CrossDesignHRStudy, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        manifest_sha256: str | None = None,
    ) -> "CrossDesignHRManifest":
        required = {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "evidence_mode",
            "status",
            "certification_effect",
            "effect_scale",
            "cross_design_strategy",
            "studies",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"cross-design manifest missing required keys: {missing}")
        if raw["schema_version"] != CROSS_DESIGN_HR_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(
                "cross-design manifest schema_version must be "
                f"{CROSS_DESIGN_HR_MANIFEST_SCHEMA_VERSION}."
            )
        studies = tuple(CrossDesignHRStudy.from_mapping(item) for item in raw["studies"])
        manifest = cls(
            benchmark_id=str(raw["benchmark_id"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            effect_scale=str(raw["effect_scale"]),
            cross_design_strategy=str(raw["cross_design_strategy"]),
            studies=studies,
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("cross-design manifest source_policy is outside the evidence boundary.")
        if self.evidence_mode != "reported_hr_pubmed_abstract_cross_design":
            raise ValidationError("cross-design manifest evidence_mode is unsupported.")
        if self.status != "candidate_source_verified":
            raise ValidationError("cross-design manifest status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("cross-design manifests cannot certify model performance.")
        if self.effect_scale != "log_hr":
            raise ValidationError("cross-design manifest effect_scale must be log_hr.")
        if self.cross_design_strategy != "separated_reporting_first":
            raise ValidationError("cross-design strategy must preserve separated reporting first.")
        if len(self.studies) < 2:
            raise ValidationError("cross-design manifest must contain at least two studies.")
        designs = {study.design for study in self.studies}
        if designs != {"rct", "nrs"}:
            raise ValidationError("cross-design manifest must include both rct and nrs rows.")
        study_ids = [study.study_id for study in self.studies]
        duplicates = sorted({study_id for study_id in study_ids if study_ids.count(study_id) > 1})
        if duplicates:
            raise ValidationError(f"cross-design manifest contains duplicate study IDs: {duplicates}")


@dataclass(frozen=True)
class CrossDesignHRVerificationRecord:
    """One PubMed abstract token verification record."""

    study_id: str
    pmid: str
    nct_id: str
    design: str
    outcome_id: str
    evidence_scope: str
    abstract_sha256: str
    reported_hr: str
    ci_lower: str
    ci_upper: str
    source_terms: tuple[str, ...]
    pmid_found: bool
    nct_id_found: bool
    hr_token_found: bool
    ci_lower_token_found: bool
    ci_upper_token_found: bool
    hazard_ratio_anchor_found: bool
    confidence_interval_anchor_found: bool
    source_terms_found: bool
    tokens_near_hazard_ratio_anchor: bool
    verified: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CrossDesignHRVerificationRecord":
        required = {
            "study_id",
            "pmid",
            "nct_id",
            "design",
            "outcome_id",
            "evidence_scope",
            "abstract_sha256",
            "reported_hr",
            "ci_lower",
            "ci_upper",
            "source_terms",
            "pmid_found",
            "nct_id_found",
            "hr_token_found",
            "ci_lower_token_found",
            "ci_upper_token_found",
            "hazard_ratio_anchor_found",
            "confidence_interval_anchor_found",
            "source_terms_found",
            "tokens_near_hazard_ratio_anchor",
            "verified",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"cross-design verification record missing required keys: {missing}")
        source_terms = raw["source_terms"]
        if not isinstance(source_terms, (list, tuple)):
            raise ValidationError("cross-design verification source_terms must be a list.")
        record = cls(
            study_id=str(raw["study_id"]),
            pmid=str(raw["pmid"]),
            nct_id=str(raw["nct_id"]),
            design=str(raw["design"]),
            outcome_id=str(raw["outcome_id"]),
            evidence_scope=str(raw["evidence_scope"]),
            abstract_sha256=str(raw["abstract_sha256"]),
            reported_hr=str(raw["reported_hr"]),
            ci_lower=str(raw["ci_lower"]),
            ci_upper=str(raw["ci_upper"]),
            source_terms=tuple(str(term) for term in source_terms),
            pmid_found=bool(raw["pmid_found"]),
            nct_id_found=bool(raw["nct_id_found"]),
            hr_token_found=bool(raw["hr_token_found"]),
            ci_lower_token_found=bool(raw["ci_lower_token_found"]),
            ci_upper_token_found=bool(raw["ci_upper_token_found"]),
            hazard_ratio_anchor_found=bool(raw["hazard_ratio_anchor_found"]),
            confidence_interval_anchor_found=bool(raw["confidence_interval_anchor_found"]),
            source_terms_found=bool(raw["source_terms_found"]),
            tokens_near_hazard_ratio_anchor=bool(raw["tokens_near_hazard_ratio_anchor"]),
            verified=bool(raw["verified"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not _PMID_RE.match(self.pmid):
            raise ValidationError(f"{self.study_id}: malformed PMID.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if self.design not in {"rct", "nrs"}:
            raise ValidationError(f"{self.study_id}: design must be rct or nrs.")
        if self.evidence_scope != "pubmed_abstract_cross_design_reported_hr_tokens":
            raise ValidationError(f"{self.study_id}: unsupported evidence_scope.")
        if not _looks_like_sha256(self.abstract_sha256):
            raise ValidationError(f"{self.study_id}: abstract_sha256 is not a SHA-256 digest.")
        _positive_decimal(self.reported_hr, self.study_id, "reported_hr")
        _positive_decimal(self.ci_lower, self.study_id, "ci_lower")
        _positive_decimal(self.ci_upper, self.study_id, "ci_upper")
        if self.verified and not all(
            (
                self.pmid_found,
                self.nct_id_found,
                self.hr_token_found,
                self.ci_lower_token_found,
                self.ci_upper_token_found,
                self.hazard_ratio_anchor_found,
                self.confidence_interval_anchor_found,
                self.source_terms_found,
                self.tokens_near_hazard_ratio_anchor,
            )
        ):
            raise ValidationError(f"{self.study_id}: verified cross-design record is missing source evidence.")


@dataclass(frozen=True)
class CrossDesignHRVerificationReport:
    """PubMed abstract token verification report for cross-design HRs."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[CrossDesignHRVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CrossDesignHRVerificationReport":
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
            raise ValidationError(f"cross-design verification report missing required keys: {missing}")
        if raw["schema_version"] != CROSS_DESIGN_HR_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                "cross-design verification schema_version must be "
                f"{CROSS_DESIGN_HR_VERIFICATION_SCHEMA_VERSION}."
            )
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            manifest=str(raw["manifest"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            records=tuple(CrossDesignHRVerificationRecord.from_mapping(item) for item in raw["records"]),
        )
        report.validate()
        return report

    def validate(self) -> None:
        if not _looks_like_sha256(self.manifest_sha256):
            raise ValidationError("cross-design verification manifest_sha256 is not a SHA-256 digest.")
        if self.status not in {"verified", "failed"}:
            raise ValidationError("cross-design verification status must be verified or failed.")
        if self.certification_effect != "none":
            raise ValidationError("cross-design verification cannot certify model performance.")
        if not self.records:
            raise ValidationError("cross-design verification report must contain records.")
        if self.status == "verified" and any(not record.verified for record in self.records):
            raise ValidationError("verified cross-design report cannot contain unverified records.")


@dataclass(frozen=True)
class CrossDesignLogEffect:
    """One log-HR effect derived from a verified cross-design row."""

    study_id: str
    trial: str
    design: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    target_population: str
    active_treatment: str
    control_treatment: str
    comparator_class: str
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
    def from_study(cls, study: CrossDesignHRStudy) -> "CrossDesignLogEffect":
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
            design=study.design,
            nct_id=study.nct_id,
            pmid=study.pmid,
            outcome_id=study.outcome_id,
            outcome_label=study.outcome_label,
            target_population=study.target_population,
            active_treatment=study.active_treatment,
            control_treatment=study.control_treatment,
            comparator_class=study.comparator_class,
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


def load_cross_design_manifest(path: str | Path) -> CrossDesignHRManifest:
    """Load and validate one cross-design HR manifest."""

    path = Path(path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return CrossDesignHRManifest.from_mapping(payload, manifest_sha256=sha256_file(path))


def load_cross_design_verification_report(path: str | Path) -> CrossDesignHRVerificationReport:
    """Load and validate a cross-design source verification report."""

    return CrossDesignHRVerificationReport.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def validate_cross_design_source_bundle(
    manifest: CrossDesignHRManifest,
    report: CrossDesignHRVerificationReport,
) -> dict[str, Any]:
    """Validate that the cross-design manifest is backed by a source report."""

    if report.benchmark_id != manifest.benchmark_id:
        raise ValidationError("cross-design verification benchmark_id does not match manifest.")
    if manifest.manifest_sha256 is not None and report.manifest_sha256 != manifest.manifest_sha256:
        raise ValidationError("cross-design verification manifest_sha256 does not match manifest file.")
    if report.status != "verified":
        raise ValidationError("cross-design source verification must be verified before benchmarking.")
    report_by_study = {record.study_id: record for record in report.records}
    manifest_ids = {study.study_id for study in manifest.studies}
    if set(report_by_study) != manifest_ids:
        raise ValidationError("cross-design verification studies do not match manifest studies.")
    design_counts: dict[str, int] = {}
    for study in manifest.studies:
        record = report_by_study[study.study_id]
        observed = (
            record.pmid,
            record.nct_id,
            record.design,
            record.outcome_id,
            record.reported_hr,
            record.ci_lower,
            record.ci_upper,
            record.source_terms,
        )
        expected = (
            study.pmid,
            study.nct_id,
            study.design,
            study.outcome_id,
            study.reported_hr,
            study.ci_lower,
            study.ci_upper,
            study.source_terms,
        )
        if observed != expected:
            raise ValidationError(f"{study.study_id}: cross-design verification does not match manifest.")
        design_counts[study.design] = design_counts.get(study.design, 0) + 1
    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": report.certification_effect,
        "n_studies": len(manifest.studies),
        "design_counts": dict(sorted(design_counts.items())),
        "source_counts": {"pubmed_abstract": len(manifest.studies)},
    }


def cross_design_log_effects(manifest: CrossDesignHRManifest) -> list[CrossDesignLogEffect]:
    """Derive log-HR effects from a cross-design manifest."""

    return [CrossDesignLogEffect.from_study(study) for study in manifest.studies]


def run_cross_design_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
) -> dict[str, Any]:
    """Run separated RCT/NRS summaries for a source-verified cross-design fixture."""

    manifest = load_cross_design_manifest(manifest_path)
    report = load_cross_design_verification_report(verification_report_path)
    source_bundle = validate_cross_design_source_bundle(manifest, report)
    effects = cross_design_log_effects(manifest)
    design_summaries = {
        design: _fixed_effect_summary([effect for effect in effects if effect.design == design])
        for design in ("rct", "nrs")
    }
    compatibility = _estimand_compatibility(effects)
    design_delta = _design_delta(design_summaries)
    return {
        "schema_version": CROSS_DESIGN_HR_BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": manifest.benchmark_id,
        "status": "local_pass",
        "certification_effect": "none",
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "effect_scale": manifest.effect_scale,
        "cross_design_strategy": manifest.cross_design_strategy,
        "source_manifest": _path_as_posix(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_verification_report": _path_as_posix(verification_report_path),
        "source_verification_report_sha256": sha256_file(verification_report_path),
        "source_bundle": source_bundle,
        "model_config": {
            "summary_method": "separated_inverse_variance_fixed_effect",
            "combined_borrowing": "blocked_unless_estimands_match",
            "level": 0.95,
        },
        "compatibility": compatibility,
        "n_studies": len(effects),
        "study_effects": [asdict(effect) for effect in effects],
        "candidate": {
            "separated_by_design": design_summaries,
            "nrs_minus_rct_diagnostic": design_delta,
        },
        "limitations": [
            "RCT and NRS rows are source verified but not assumed exchangeable",
            "combined borrowing is blocked when target population, comparator, or outcome estimands differ",
            "this is a cross-design routing benchmark, not crossnma reference matching",
            "does not certify model performance",
        ],
    }


def _fixed_effect_summary(effects: list[CrossDesignLogEffect]) -> dict[str, Any]:
    if not effects:
        return {
            "n_studies": 0,
            "estimate": "",
            "se": "",
            "ci_low": "",
            "ci_high": "",
            "hr": "",
            "hr_ci_low": "",
            "hr_ci_high": "",
        }
    weights = [1.0 / effect.variance for effect in effects]
    weight_sum = sum(weights)
    estimate = sum(weight * effect.estimate for weight, effect in zip(weights, effects)) / weight_sum
    se = math.sqrt(1.0 / weight_sum)
    ci_low = estimate - NORMAL_975 * se
    ci_high = estimate + NORMAL_975 * se
    return {
        "n_studies": len(effects),
        "estimate": estimate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "hr": math.exp(estimate),
        "hr_ci_low": math.exp(ci_low),
        "hr_ci_high": math.exp(ci_high),
    }


def _design_delta(design_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rct = design_summaries["rct"]
    nrs = design_summaries["nrs"]
    if not rct["n_studies"] or not nrs["n_studies"]:
        return {"status": "unavailable_missing_design", "certification_effect": "none"}
    estimate = float(nrs["estimate"]) - float(rct["estimate"])
    se = math.sqrt(float(nrs["se"]) ** 2 + float(rct["se"]) ** 2)
    ci_low = estimate - NORMAL_975 * se
    ci_high = estimate + NORMAL_975 * se
    return {
        "status": "diagnostic_only",
        "certification_effect": "none",
        "estimate": estimate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _estimand_compatibility(effects: list[CrossDesignLogEffect]) -> dict[str, Any]:
    fields = {
        "outcome_id": sorted({effect.outcome_id for effect in effects}),
        "target_population": sorted({effect.target_population for effect in effects}),
        "control_treatment": sorted({effect.control_treatment for effect in effects}),
        "comparator_class": sorted({effect.comparator_class for effect in effects}),
    }
    mismatches = {
        key: values
        for key, values in fields.items()
        if len(values) > 1
    }
    if mismatches:
        status = "separated_only_estimand_mismatch"
        combined_borrowing_allowed = False
    else:
        status = "compatible_for_prespecified_cross_design_model"
        combined_borrowing_allowed = True
    return {
        "status": status,
        "combined_borrowing_allowed": combined_borrowing_allowed,
        "mismatched_fields": mismatches,
        "certification_effect": "none",
    }


def _positive_decimal(value: str, study_id: str, field_name: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"{study_id}: {field_name} must be a decimal string.") from exc
    if parsed <= 0:
        raise ValidationError(f"{study_id}: {field_name} must be positive.")
    return parsed


def _path_as_posix(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)
