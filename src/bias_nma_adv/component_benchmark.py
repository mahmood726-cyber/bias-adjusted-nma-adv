"""Source-backed component-NMA smoke benchmark support."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any
from urllib.parse import urlparse

from bias_nma_adv.component_nma import fit_additive_component_nma
from bias_nma_adv.data import ValidationError
from bias_nma_adv.real_meta import sha256_file


COMPONENT_MANIFEST_SCHEMA_VERSION = "component_nma_ctgov_manifest/v1"
COMPONENT_VERIFICATION_SCHEMA_VERSION = "component_nma_source_verification/v1"
COMPONENT_BENCHMARK_SCHEMA_VERSION = "component_nma_source_benchmark/v1"
_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")
_Z_975 = 1.959963984540054


@dataclass(frozen=True)
class ComponentArm:
    """One CT.gov arm for a component-NMA source benchmark."""

    arm_id: str
    group_id: str
    treatment_label: str
    components: tuple[str, ...]
    n: int
    lsmean: float
    lower: float
    upper: float
    source_terms: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ComponentArm":
        required = {
            "arm_id",
            "group_id",
            "treatment_label",
            "components",
            "n",
            "lsmean",
            "lower",
            "upper",
            "source_terms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"component arm missing required keys: {missing}")
        arm = cls(
            arm_id=str(raw["arm_id"]),
            group_id=str(raw["group_id"]),
            treatment_label=str(raw["treatment_label"]),
            components=tuple(str(item) for item in raw["components"]),
            n=int(raw["n"]),
            lsmean=float(raw["lsmean"]),
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            source_terms=tuple(str(item) for item in raw["source_terms"]),
        )
        arm.validate()
        return arm

    @property
    def se(self) -> float:
        return (self.upper - self.lower) / (2.0 * _Z_975)

    @property
    def component_treatment(self) -> str:
        return " + ".join(self.components)

    def validate(self) -> None:
        if not self.arm_id.strip() or not self.group_id.strip():
            raise ValidationError("component arm identifiers must not be empty.")
        if not self.treatment_label.strip():
            raise ValidationError(f"{self.arm_id}: treatment_label must not be empty.")
        if not self.components or any(not item.strip() for item in self.components):
            raise ValidationError(f"{self.arm_id}: components must be non-empty strings.")
        if len(set(self.components)) != len(self.components):
            raise ValidationError(f"{self.arm_id}: duplicate components are not allowed.")
        if self.n <= 0:
            raise ValidationError(f"{self.arm_id}: n must be positive.")
        if not math.isfinite(self.lsmean):
            raise ValidationError(f"{self.arm_id}: lsmean must be finite.")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper) or self.upper <= self.lower:
            raise ValidationError(f"{self.arm_id}: confidence limits must be finite and ordered.")
        if self.se <= 0.0:
            raise ValidationError(f"{self.arm_id}: derived standard error must be positive.")
        if not self.source_terms or any(not term.strip() for term in self.source_terms):
            raise ValidationError(f"{self.arm_id}: source_terms must be non-empty strings.")


@dataclass(frozen=True)
class ComponentManifest:
    """Source-bounded CT.gov component-NMA manifest."""

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
    reuse_origin: str
    arms: tuple[ComponentArm, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        manifest_sha256: str | None = None,
    ) -> "ComponentManifest":
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
            "reuse_origin",
            "arms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"component manifest missing required keys: {missing}")
        if raw["schema_version"] != COMPONENT_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(
                f"component manifest schema_version must be {COMPONENT_MANIFEST_SCHEMA_VERSION}."
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
            reuse_origin=str(raw["reuse_origin"]),
            arms=tuple(ComponentArm.from_mapping(item) for item in raw["arms"]),
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("component source_policy is outside the evidence boundary.")
        if self.evidence_mode != "ctgov_component_lsmean":
            raise ValidationError("component evidence_mode is unsupported.")
        if self.status != "candidate_source_verified":
            raise ValidationError("component manifest status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("component manifests cannot certify model performance.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError("component manifest has malformed NCT ID.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError("component manifest has malformed PMID.")
        source_host = urlparse(self.source_url).hostname
        if source_host != "clinicaltrials.gov" or self.nct_id not in self.source_url:
            raise ValidationError("source_url must be a ClinicalTrials.gov URL containing the NCT ID.")
        pubmed_host = urlparse(self.pubmed_url).hostname
        if pubmed_host != "pubmed.ncbi.nlm.nih.gov" or self.pmid not in self.pubmed_url:
            raise ValidationError("pubmed_url must be a PubMed URL containing the PMID.")
        if "clinicaltrials.gov" not in self.access_statement.lower():
            raise ValidationError("access_statement must identify ClinicalTrials.gov.")
        if self.effect_scale != "percentage_point_change_a1c":
            raise ValidationError("component effect_scale is unsupported.")
        if self.reuse_origin != "source_backed_new_component_lane":
            raise ValidationError("component reuse_origin must describe a new source-backed lane.")
        if len(self.arms) < 4:
            raise ValidationError("component manifest must include at least four arms.")
        arm_ids = [arm.arm_id for arm in self.arms]
        group_ids = [arm.group_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids) or len(set(group_ids)) != len(group_ids):
            raise ValidationError("component arm and group IDs must be unique.")
        if not any(len(arm.components) > 1 for arm in self.arms):
            raise ValidationError("component manifest must include at least one combination arm.")
        component_counts: dict[str, int] = {}
        for arm in self.arms:
            for component in arm.components:
                component_counts[component] = component_counts.get(component, 0) + 1
        if not any(count >= 2 for count in component_counts.values()):
            raise ValidationError("at least one component must appear in multiple arms.")


@dataclass(frozen=True)
class ComponentVerificationRecord:
    """One source verification record for the component benchmark."""

    source_type: str
    identifier: str
    evidence_scope: str
    response_sha256: str
    verified: bool
    details: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ComponentVerificationRecord":
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
            raise ValidationError(f"component verification record missing keys: {missing}")
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
            raise ValidationError("component verification source_type is unsupported.")
        if not self.identifier.strip():
            raise ValidationError("component verification identifier must not be empty.")
        if len(self.response_sha256) != 64:
            raise ValidationError("component verification response_sha256 must be SHA-256 length.")
        if not self.verified:
            raise ValidationError("component verification record is not verified.")


@dataclass(frozen=True)
class ComponentVerificationReport:
    """Source verification report for a component manifest."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[ComponentVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ComponentVerificationReport":
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
            raise ValidationError(f"component verification report missing keys: {missing}")
        if raw["schema_version"] != COMPONENT_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                f"component verification schema_version must be {COMPONENT_VERIFICATION_SCHEMA_VERSION}."
            )
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            manifest=str(raw["manifest"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            records=tuple(ComponentVerificationRecord.from_mapping(item) for item in raw["records"]),
        )
        report.validate()
        return report

    def validate(self) -> None:
        if self.status != "verified":
            raise ValidationError("component verification report status must be verified.")
        if self.certification_effect != "none":
            raise ValidationError("component verification report cannot certify.")
        source_types = {record.source_type for record in self.records}
        if not {"clinicaltrials_gov", "pubmed_abstract"} <= source_types:
            raise ValidationError("component verification requires CT.gov and PubMed records.")


def load_component_manifest(path: str | Path) -> ComponentManifest:
    manifest_path = Path(path)
    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return ComponentManifest.from_mapping(payload, manifest_sha256=sha256_file(manifest_path))


def load_component_verification_report(path: str | Path) -> ComponentVerificationReport:
    return ComponentVerificationReport.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def validate_component_source_bundle(
    manifest: ComponentManifest,
    report: ComponentVerificationReport,
) -> dict[str, Any]:
    if manifest.benchmark_id != report.benchmark_id:
        raise ValidationError("component verification benchmark_id mismatch.")
    if manifest.manifest_sha256 != report.manifest_sha256:
        raise ValidationError("component verification manifest SHA mismatch.")
    if report.status != "verified":
        raise ValidationError("component source bundle is not verified.")
    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": "none",
        "n_records": len(report.records),
        "source_counts": _count_by(record.source_type for record in report.records),
    }


def component_contrasts(manifest: ComponentManifest) -> tuple[dict[str, Any], ...]:
    contrasts: list[dict[str, Any]] = []
    for left, right in itertools.combinations(manifest.arms, 2):
        estimate = right.lsmean - left.lsmean
        variance = right.se * right.se + left.se * left.se
        contrasts.append(
            {
                "study_id": f"{manifest.nct_id}_{right.arm_id}_vs_{left.arm_id}",
                "trial": manifest.trial,
                "nct_id": manifest.nct_id,
                "pmid": manifest.pmid,
                "treat1": right.component_treatment,
                "treat2": left.component_treatment,
                "arm1_id": right.arm_id,
                "arm2_id": left.arm_id,
                "arm1_group_id": right.group_id,
                "arm2_group_id": left.group_id,
                "arm1_n": right.n,
                "arm2_n": left.n,
                "arm1_lsmean": _stable_float(right.lsmean),
                "arm2_lsmean": _stable_float(left.lsmean),
                "arm1_se": _stable_float(right.se),
                "arm2_se": _stable_float(left.se),
                "estimate": _stable_float(estimate),
                "se": _stable_float(math.sqrt(variance)),
                "variance": _stable_float(variance),
                "effect_scale": manifest.effect_scale,
                "variance_source": "sqrt(arm1_ci_se^2 + arm2_ci_se^2); same-trial covariance not modeled",
            }
        )
    return tuple(contrasts)


def run_component_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
) -> dict[str, Any]:
    manifest = load_component_manifest(manifest_path)
    report = load_component_verification_report(verification_report_path)
    source_bundle = validate_component_source_bundle(manifest, report)
    contrasts = component_contrasts(manifest)
    fit = fit_additive_component_nma(
        [
            {
                "study_id": item["study_id"],
                "treat1": item["treat1"],
                "treat2": item["treat2"],
                "estimate": item["estimate"],
                "se": item["se"],
            }
            for item in contrasts
        ],
        inactive_treatment="inactive_reference_not_observed",
    )
    root = Path.cwd()
    manifest_rel = _relpath(Path(manifest_path), root)
    report_rel = _relpath(Path(verification_report_path), root)
    return {
        "schema_version": COMPONENT_BENCHMARK_SCHEMA_VERSION,
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
        "n_component_contrasts": len(contrasts),
        "limitations": [
            "single CT.gov factorial trial only",
            "arm-level LS mean contrasts are derived from reported confidence intervals",
            "same-trial covariance is not modeled",
            "not broad netmeta CNMA parity",
            "not clinical superiority claims",
            "does not certify model performance",
        ],
        "source_bundle": source_bundle,
        "model_config": {
            "candidate_model": "fixed_effect_additive_component_wls",
            "outcome_title": manifest.outcome_title,
            "component_separator": " + ",
        },
        "study_effects": list(contrasts),
        "candidate": {
            "additive_component_wls": {
                "rank": fit.rank,
                "df": fit.df,
                "q": _stable_float(fit.q),
                "component_effects": [
                    {
                        "name": effect.name,
                        "estimate": _stable_float(effect.estimate),
                        "se": _stable_float(effect.se),
                        "estimable": effect.estimable,
                    }
                    for effect in fit.component_effects
                ],
                "warnings": list(fit.warnings)
                + [
                    "Source-backed component smoke benchmark only; not netmeta CNMA parity.",
                    "Same-trial arm covariance is ignored.",
                ],
            }
        },
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
