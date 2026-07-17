"""ClinicalTrials.gov arm-count binary closed-loop network benchmark support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import string
import tomllib
from typing import Any
from urllib.parse import urlparse

from bias_nma_adv.data import ValidationError
from bias_nma_adv.multiarm import ContrastRow, fit_multiarm_gls
from bias_nma_adv.node_splitting import fixed_effect_node_splitting
from bias_nma_adv.real_meta import sha256_file


CTGOV_BINARY_NETWORK_MANIFEST_SCHEMA_VERSION = "ctgov_binary_network_manifest/v1"
CTGOV_BINARY_NETWORK_VERIFICATION_SCHEMA_VERSION = "ctgov_binary_network_verification/v1"

_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class CTGovBinaryArm:
    """One source-bound binary arm count from a CT.gov result table."""

    arm_id: str
    group_id: str
    treatment: str
    group_title: str
    events: int
    n: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovBinaryArm":
        required = {"arm_id", "group_id", "treatment", "group_title", "events", "n"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov binary arm missing required keys: {missing}")
        arm = cls(
            arm_id=str(raw["arm_id"]),
            group_id=str(raw["group_id"]),
            treatment=str(raw["treatment"]),
            group_title=str(raw["group_title"]),
            events=int(raw["events"]),
            n=int(raw["n"]),
        )
        arm.validate()
        return arm

    def validate(self) -> None:
        if not self.arm_id.strip() or not self.group_id.strip() or not self.treatment.strip():
            raise ValidationError("CT.gov binary arm identifiers and treatment must not be empty.")
        if self.events <= 0 or self.n <= 0 or self.events >= self.n:
            raise ValidationError(
                f"{self.arm_id}: event count must satisfy 0 < events < n for no-correction log-OR contrasts."
            )


@dataclass(frozen=True)
class CTGovBinaryNetworkStudy:
    """One CT.gov trial supplying binary arm counts for the closed-loop benchmark."""

    study_id: str
    trial: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    source_url: str
    pubmed_url: str
    access_statement: str
    outcome_search_terms: tuple[str, ...]
    source_terms: tuple[str, ...]
    pubmed_title_terms: tuple[str, ...]
    arms: tuple[CTGovBinaryArm, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovBinaryNetworkStudy":
        required = {
            "study_id",
            "trial",
            "nct_id",
            "pmid",
            "outcome_id",
            "outcome_label",
            "source_url",
            "pubmed_url",
            "access_statement",
            "outcome_search_terms",
            "source_terms",
            "pubmed_title_terms",
            "arms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov binary network study missing required keys: {missing}")
        for key in ("outcome_search_terms", "source_terms", "pubmed_title_terms", "arms"):
            if not isinstance(raw[key], list | tuple) or not raw[key]:
                raise ValidationError(f"{raw.get('study_id', 'study')}: {key} must be a non-empty list.")
        study = cls(
            study_id=str(raw["study_id"]),
            trial=str(raw["trial"]),
            nct_id=str(raw["nct_id"]),
            pmid=str(raw["pmid"]),
            outcome_id=str(raw["outcome_id"]),
            outcome_label=str(raw["outcome_label"]),
            source_url=str(raw["source_url"]),
            pubmed_url=str(raw["pubmed_url"]),
            access_statement=str(raw["access_statement"]),
            outcome_search_terms=tuple(str(item) for item in raw["outcome_search_terms"]),
            source_terms=tuple(str(item) for item in raw["source_terms"]),
            pubmed_title_terms=tuple(str(item) for item in raw["pubmed_title_terms"]),
            arms=tuple(CTGovBinaryArm.from_mapping(item) for item in raw["arms"]),
        )
        study.validate()
        return study

    def validate(self) -> None:
        if not self.study_id.strip() or not self.trial.strip():
            raise ValidationError("CT.gov binary network study_id and trial must not be empty.")
        if not _NCT_RE.match(self.nct_id):
            raise ValidationError(f"{self.study_id}: malformed NCT ID.")
        if not _PMID_RE.match(self.pmid):
            raise ValidationError(f"{self.study_id}: malformed PMID.")
        parsed_ctgov = urlparse(self.source_url)
        if parsed_ctgov.hostname != "clinicaltrials.gov" or self.nct_id not in parsed_ctgov.path:
            raise ValidationError(f"{self.study_id}: source_url must be a CT.gov study URL containing the NCT ID.")
        parsed_pubmed = urlparse(self.pubmed_url)
        if parsed_pubmed.hostname != "pubmed.ncbi.nlm.nih.gov" or self.pmid not in parsed_pubmed.path:
            raise ValidationError(f"{self.study_id}: pubmed_url must be a PubMed URL containing the PMID.")
        if "clinicaltrials.gov" not in self.access_statement.lower():
            raise ValidationError(f"{self.study_id}: access_statement must identify ClinicalTrials.gov.")
        if len(self.arms) < 3:
            raise ValidationError(f"{self.study_id}: at least three arms are required for a closed-loop study.")
        treatments = [arm.treatment for arm in self.arms]
        if len(set(treatments)) != len(treatments):
            raise ValidationError(f"{self.study_id}: arm treatments must be unique within a study.")
        group_ids = [arm.group_id for arm in self.arms]
        if len(set(group_ids)) != len(group_ids):
            raise ValidationError(f"{self.study_id}: CT.gov group IDs must be unique within a study.")


@dataclass(frozen=True)
class CTGovBinaryNetworkManifest:
    """Source-bounded CT.gov arm-count binary closed-loop network manifest."""

    benchmark_id: str
    source_policy: str
    evidence_mode: str
    status: str
    certification_effect: str
    reference_treatment: str
    network_type: str
    effect_scale: str
    continuity_policy: str
    reuse_origin: str
    studies: tuple[CTGovBinaryNetworkStudy, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        manifest_sha256: str | None = None,
    ) -> "CTGovBinaryNetworkManifest":
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
            "continuity_policy",
            "reuse_origin",
            "studies",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov binary network manifest missing required keys: {missing}")
        if raw["schema_version"] != CTGOV_BINARY_NETWORK_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(
                "CT.gov binary network manifest schema_version must be "
                f"{CTGOV_BINARY_NETWORK_MANIFEST_SCHEMA_VERSION}."
            )
        studies = tuple(CTGovBinaryNetworkStudy.from_mapping(item) for item in raw["studies"])
        manifest = cls(
            benchmark_id=str(raw["benchmark_id"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            reference_treatment=str(raw["reference_treatment"]),
            network_type=str(raw["network_type"]),
            effect_scale=str(raw["effect_scale"]),
            continuity_policy=str(raw["continuity_policy"]),
            reuse_origin=str(raw["reuse_origin"]),
            studies=studies,
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("CT.gov binary network source_policy is outside the evidence boundary.")
        if self.evidence_mode != "ctgov_arm_level_binary_counts":
            raise ValidationError("CT.gov binary network evidence_mode is unsupported.")
        if self.status != "candidate_source_verified":
            raise ValidationError("CT.gov binary network status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("CT.gov binary network manifests cannot certify model performance.")
        if self.network_type != "closed_loop_arm_level_binary_network":
            raise ValidationError("CT.gov binary network network_type must be closed_loop_arm_level_binary_network.")
        if self.effect_scale != "log_or":
            raise ValidationError("CT.gov binary network effect_scale must be log_or.")
        if self.continuity_policy != "no_continuity_correction_zero_cells_fail_closed":
            raise ValidationError("CT.gov binary network continuity policy drifted.")
        if len(self.studies) < 2:
            raise ValidationError("CT.gov binary network manifest must contain at least two studies.")
        study_ids = [study.study_id for study in self.studies]
        duplicates = sorted({study_id for study_id in study_ids if study_ids.count(study_id) > 1})
        if duplicates:
            raise ValidationError(f"CT.gov binary network manifest contains duplicate study IDs: {duplicates}")
        if self.reference_treatment not in {arm.treatment for study in self.studies for arm in study.arms}:
            raise ValidationError("reference_treatment is absent from the arm-count network.")
        if _cycle_rank(self.contrast_rows()) <= 0:
            raise ValidationError("CT.gov binary network manifest must contain at least one closed loop.")

    def contrast_rows(self) -> tuple[ContrastRow, ...]:
        return contrast_rows_from_manifest(self)


@dataclass(frozen=True)
class CTGovBinaryNetworkVerificationRecord:
    """One source verification record for the binary closed-loop benchmark."""

    study_id: str
    source_type: str
    identifier: str
    evidence_scope: str
    response_sha256: str
    verified: bool
    details: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovBinaryNetworkVerificationRecord":
        required = {
            "study_id",
            "source_type",
            "identifier",
            "evidence_scope",
            "response_sha256",
            "verified",
            "details",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"CT.gov binary verification record missing required keys: {missing}")
        record = cls(
            study_id=str(raw["study_id"]),
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
        if not self.study_id.strip():
            raise ValidationError("CT.gov binary verification study_id must not be empty.")
        if self.source_type not in {"clinicaltrials_gov", "pubmed_abstract"}:
            raise ValidationError(f"{self.study_id}: unsupported binary-network source_type.")
        if self.source_type == "clinicaltrials_gov":
            if not _NCT_RE.match(self.identifier):
                raise ValidationError(f"{self.study_id}: malformed NCT identifier.")
            if self.evidence_scope != "clinicaltrials_gov_arm_level_binary_counts":
                raise ValidationError(f"{self.study_id}: unsupported CT.gov evidence scope.")
        if self.source_type == "pubmed_abstract":
            if not _PMID_RE.match(self.identifier):
                raise ValidationError(f"{self.study_id}: malformed PMID identifier.")
            if self.evidence_scope != "pubmed_abstract_binary_network_identity":
                raise ValidationError(f"{self.study_id}: unsupported PubMed evidence scope.")
        if not _looks_like_sha256(self.response_sha256):
            raise ValidationError(f"{self.study_id}: response_sha256 is not a SHA-256 digest.")
        if not self.verified:
            raise ValidationError(f"{self.study_id}: verification record is not verified.")


@dataclass(frozen=True)
class CTGovBinaryNetworkVerificationReport:
    """Source verification snapshot for a CT.gov binary network manifest."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[CTGovBinaryNetworkVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CTGovBinaryNetworkVerificationReport":
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
            raise ValidationError(f"CT.gov binary verification report missing required keys: {missing}")
        if raw["schema_version"] != CTGOV_BINARY_NETWORK_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                "CT.gov binary verification schema_version must be "
                f"{CTGOV_BINARY_NETWORK_VERIFICATION_SCHEMA_VERSION}."
            )
        records = tuple(CTGovBinaryNetworkVerificationRecord.from_mapping(item) for item in raw["records"])
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
        if not self.benchmark_id.strip() or not self.checked_at.strip():
            raise ValidationError("CT.gov binary verification report identity fields must not be empty.")
        if not _looks_like_sha256(self.manifest_sha256):
            raise ValidationError("CT.gov binary verification manifest_sha256 is not a SHA-256 digest.")
        if self.status != "verified":
            raise ValidationError("CT.gov binary verification report status must be verified.")
        if self.certification_effect != "none":
            raise ValidationError("CT.gov binary verification reports cannot certify model performance.")
        if not self.records:
            raise ValidationError("CT.gov binary verification report must contain records.")


def load_ctgov_binary_network_manifest(path: str | Path) -> CTGovBinaryNetworkManifest:
    source = Path(path)
    with source.open("rb") as handle:
        payload = tomllib.load(handle)
    return CTGovBinaryNetworkManifest.from_mapping(payload, manifest_sha256=sha256_file(source))


def load_ctgov_binary_network_verification_report(
    path: str | Path,
) -> CTGovBinaryNetworkVerificationReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CTGovBinaryNetworkVerificationReport.from_mapping(payload)


def validate_ctgov_binary_network_source_bundle(
    manifest: CTGovBinaryNetworkManifest,
    report: CTGovBinaryNetworkVerificationReport,
) -> None:
    """Validate a manifest/report pair without making network calls."""

    if report.benchmark_id != manifest.benchmark_id:
        raise ValidationError("CT.gov binary verification benchmark_id mismatch.")
    if manifest.manifest_sha256 and report.manifest_sha256 != manifest.manifest_sha256:
        raise ValidationError("CT.gov binary verification manifest SHA-256 mismatch.")
    records_by_study: dict[str, list[CTGovBinaryNetworkVerificationRecord]] = {}
    for record in report.records:
        records_by_study.setdefault(record.study_id, []).append(record)
    for study in manifest.studies:
        records = records_by_study.get(study.study_id, [])
        source_types = {record.source_type for record in records}
        if source_types != {"clinicaltrials_gov", "pubmed_abstract"}:
            raise ValidationError(f"{study.study_id}: CT.gov and PubMed verification records are required.")
        ctgov = next(record for record in records if record.source_type == "clinicaltrials_gov")
        pubmed = next(record for record in records if record.source_type == "pubmed_abstract")
        if ctgov.identifier != study.nct_id:
            raise ValidationError(f"{study.study_id}: CT.gov verification identifier mismatch.")
        if pubmed.identifier != study.pmid:
            raise ValidationError(f"{study.study_id}: PubMed verification identifier mismatch.")
        _require_true(ctgov.details, study.study_id, "nct_id_found")
        _require_true(ctgov.details, study.study_id, "status_completed")
        _require_true(ctgov.details, study.study_id, "outcome_found")
        _require_true(ctgov.details, study.study_id, "arm_counts_found")
        _require_true(ctgov.details, study.study_id, "source_terms_found")
        _require_true(pubmed.details, study.study_id, "pmid_found")
        _require_true(pubmed.details, study.study_id, "title_terms_found")


def contrast_rows_from_manifest(manifest: CTGovBinaryNetworkManifest) -> tuple[ContrastRow, ...]:
    rows: list[ContrastRow] = []
    for study in manifest.studies:
        arms = study.arms
        for index, arm_1 in enumerate(arms):
            for arm_2 in arms[index + 1 :]:
                rows.append(
                    ContrastRow(
                        study=study.study_id,
                        t1=arm_1.treatment,
                        t2=arm_2.treatment,
                        est=_log_odds(arm_2.events, arm_2.n) - _log_odds(arm_1.events, arm_1.n),
                        se=math.sqrt(
                            1.0 / arm_1.events
                            + 1.0 / (arm_1.n - arm_1.events)
                            + 1.0 / arm_2.events
                            + 1.0 / (arm_2.n - arm_2.events)
                        ),
                    )
                )
    return tuple(rows)


def run_ctgov_binary_network_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
) -> dict[str, Any]:
    manifest = load_ctgov_binary_network_manifest(manifest_path)
    report = load_ctgov_binary_network_verification_report(verification_report_path)
    validate_ctgov_binary_network_source_bundle(manifest, report)

    rows = contrast_rows_from_manifest(manifest)
    fixed = fit_multiarm_gls(rows, reference_treatment=manifest.reference_treatment, model="fixed")
    random = fit_multiarm_gls(rows, reference_treatment=manifest.reference_treatment, model="random")
    node_splits = fixed_effect_node_splitting(rows, reference_treatment=manifest.reference_treatment)
    cycle_rank = _cycle_rank(rows)

    return {
        "schema_version": "ctgov_binary_network_benchmark/v1",
        "benchmark_id": manifest.benchmark_id,
        "status": "local_pass",
        "certification_effect": "none",
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "network_type": manifest.network_type,
        "effect_scale": manifest.effect_scale,
        "continuity_policy": manifest.continuity_policy,
        "reference_treatment": manifest.reference_treatment,
        "source_manifest": Path(manifest_path).as_posix(),
        "source_manifest_sha256": manifest.manifest_sha256,
        "source_verification_report": Path(verification_report_path).as_posix(),
        "source_verification_report_sha256": sha256_file(verification_report_path),
        "n_studies": len(manifest.studies),
        "n_treatments": len({arm.treatment for study in manifest.studies for arm in study.arms}),
        "n_arms": sum(len(study.arms) for study in manifest.studies),
        "n_contrast_rows": len(rows),
        "closed_loop_cycle_rank": cycle_rank,
        "limitations": [
            "single dermatology endpoint family only",
            "closed-loop source-backed smoke benchmark, not broad inconsistency performance",
            "node-splitting diagnostics are local fixed-effect smoke checks, not netmeta or multinma reference parity",
            "same-trial multi-arm correlations are modeled through the contrast covariance but this is not CINeMA or ROB-MEN certainty",
            "does not certify model performance",
        ],
        "source_bundle": {
            "manifest_status": manifest.status,
            "verification_status": report.status,
            "nct_ids": sorted({study.nct_id for study in manifest.studies}),
            "pmids": sorted({study.pmid for study in manifest.studies}),
            "source_types": ["clinicaltrials_gov", "pubmed_abstract"],
        },
        "model_config": {
            "contrast_generation": "all_pairwise_log_or_contrasts_from_arm_counts",
            "continuity_correction": "none",
            "zero_cell_policy": "fail_closed",
            "fixed_effect_engine": "multiarm_gls",
            "random_effect_engine": "multiarm_generalized_dl",
            "node_splitting_engine": "fixed_effect_direct_indirect_smoke",
        },
        "arm_counts": _arm_count_payload(manifest),
        "study_effects": _study_effect_payload(manifest, rows),
        "candidate": {
            "fixed_effect": _fit_payload(fixed),
            "random_effect": _fit_payload(random),
            "node_splitting": {
                "n_diagnostics": len(node_splits),
                "n_estimable": sum(1 for row in node_splits if row.status == "estimable"),
                "diagnostics": [asdict(row) for row in node_splits],
                "claim_limit": "Local fixed-effect smoke diagnostic only; not reference-matched node-splitting parity.",
            },
        },
    }


def _fit_payload(fit: Any) -> dict[str, Any]:
    effects = []
    for treatment in fit.nonreference_treatments:
        estimate, se = fit.effect_vs_reference(treatment)
        effects.append(
            {
                "treatment": treatment,
                "reference_treatment": fit.reference_treatment,
                "estimate": float(estimate),
                "se": float(se),
                "ci_low": float(estimate - 1.959963984540054 * se),
                "ci_high": float(estimate + 1.959963984540054 * se),
            }
        )
    heatmap = fit.study_contribution_heatmap()
    return {
        "model": fit.model,
        "tau2": float(fit.tau2),
        "q": float(fit.q) if fit.q is not None else None,
        "df": int(fit.df) if fit.df is not None else None,
        "multi_arm_studies": list(fit.multi_arm_studies),
        "warnings": list(fit.warnings),
        "study_contribution_heatmap": {
            "target_treatments": list(heatmap.target_treatments),
            "studies": list(heatmap.studies),
            "values": [list(row) for row in heatmap.values],
            "warnings": list(heatmap.warnings),
        },
        "effects": effects,
    }


def _arm_count_payload(manifest: CTGovBinaryNetworkManifest) -> list[dict[str, Any]]:
    rows = []
    for study in manifest.studies:
        for arm in study.arms:
            rows.append(
                {
                    "study_id": study.study_id,
                    "trial": study.trial,
                    "nct_id": study.nct_id,
                    "pmid": study.pmid,
                    "outcome_id": study.outcome_id,
                    "outcome_label": study.outcome_label,
                    "arm_id": arm.arm_id,
                    "group_id": arm.group_id,
                    "treatment": arm.treatment,
                    "group_title": arm.group_title,
                    "events": arm.events,
                    "n": arm.n,
                }
            )
    return rows


def _study_effect_payload(
    manifest: CTGovBinaryNetworkManifest,
    rows: tuple[ContrastRow, ...],
) -> list[dict[str, Any]]:
    studies = {study.study_id: study for study in manifest.studies}
    arms_by_study = {
        study.study_id: {arm.treatment: arm for arm in study.arms}
        for study in manifest.studies
    }
    payload = []
    for row in rows:
        study = studies[row.study]
        arm_1 = arms_by_study[row.study][row.t1]
        arm_2 = arms_by_study[row.study][row.t2]
        payload.append(
            {
                "study_id": row.study,
                "trial": study.trial,
                "nct_id": study.nct_id,
                "pmid": study.pmid,
                "outcome_id": study.outcome_id,
                "outcome_label": study.outcome_label,
                "treatment_from": row.t1,
                "treatment_to": row.t2,
                "events_from": arm_1.events,
                "n_from": arm_1.n,
                "events_to": arm_2.events,
                "n_to": arm_2.n,
                "effect_direction": "treatment_to_minus_treatment_from",
                "effect_scale": "log_or",
                "continuity_correction": "none_zero_cells_fail_closed",
                "estimate": float(row.est),
                "variance": float(row.se * row.se),
                "se": float(row.se),
            }
        )
    return payload


def _cycle_rank(rows: tuple[ContrastRow, ...]) -> int:
    treatments = sorted({row.t1 for row in rows} | {row.t2 for row in rows})
    edges = {tuple(sorted((row.t1, row.t2))) for row in rows}
    if not treatments:
        return 0
    parent = {treatment: treatment for treatment in treatments}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)
    components = len({find(treatment) for treatment in treatments})
    return len(edges) - len(treatments) + components


def _log_odds(events: int, n: int) -> float:
    if events <= 0 or events >= n:
        raise ValidationError("log-OR conversion requires 0 < events < n without continuity correction.")
    return math.log(events / (n - events))


def _require_true(details: dict[str, Any], study_id: str, key: str) -> None:
    if not bool(details.get(key)):
        raise ValidationError(f"{study_id}: verification detail {key} is not true.")


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)
