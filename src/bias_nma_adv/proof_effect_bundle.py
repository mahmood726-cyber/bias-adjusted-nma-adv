"""Proof-carrying extracted-effect bundles for source-backed validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from bias_nma_adv.ingestion import (
    EvidenceIngestionRecord,
    ExtractionProvenance,
    IngestionProvenanceError,
    ProofCarryingEffectRecord,
    validate_proof_carrying_effects,
)
from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.source_verification import load_source_verification_report
from bias_nma_adv.survival_benchmark import (
    load_survival_hr_manifest,
    load_survival_hr_verification_report,
    validate_survival_hr_identity_bundle,
    validate_survival_hr_source_bundle,
)


PROOF_EFFECT_BUNDLE_SCHEMA_VERSION = "proof_effect_bundle/v1"


class ProofEffectBundleError(ValueError):
    """Raised when a proof-carrying extracted-effect bundle is malformed."""


def load_proof_effect_bundle(path: str | Path) -> dict[str, Any]:
    """Load a proof-carrying extracted-effect bundle JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_proof_effect_bundle(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate a proof-carrying extracted-effect bundle and its source reports."""

    bundle_path = Path(path)
    root = Path(repo_root)
    bundle = load_proof_effect_bundle(bundle_path)
    _require_keys(
        bundle,
        {
            "schema_version",
            "bundle_id",
            "source_policy",
            "evidence_mode",
            "status",
            "certification_effect",
            "source_manifest",
            "source_manifest_sha256",
            "token_verification_report",
            "token_verification_sha256",
            "identity_verification_report",
            "identity_verification_sha256",
            "records",
            "limitations",
        },
        label="proof-effect bundle",
    )
    if bundle["schema_version"] != PROOF_EFFECT_BUNDLE_SCHEMA_VERSION:
        raise ProofEffectBundleError(
            f"proof-effect bundle schema_version must be {PROOF_EFFECT_BUNDLE_SCHEMA_VERSION}."
        )
    if bundle["certification_effect"] != "none":
        raise ProofEffectBundleError("proof-effect bundles cannot certify model performance.")
    if bundle["status"] != "local_pass":
        raise ProofEffectBundleError("proof-effect bundle status must be local_pass.")
    if bundle["evidence_mode"] != "reported_hr_pubmed_abstract":
        raise ProofEffectBundleError("proof-effect bundle currently supports reported_hr_pubmed_abstract.")
    if not bundle["limitations"]:
        raise ProofEffectBundleError("proof-effect bundle must declare limitations.")

    manifest_path = root / str(bundle["source_manifest"])
    token_report_path = root / str(bundle["token_verification_report"])
    identity_report_path = root / str(bundle["identity_verification_report"])
    _assert_hash(manifest_path, str(bundle["source_manifest_sha256"]), "source manifest")
    _assert_hash(token_report_path, str(bundle["token_verification_sha256"]), "token verification report")
    _assert_hash(
        identity_report_path,
        str(bundle["identity_verification_sha256"]),
        "identity verification report",
    )

    manifest = load_survival_hr_manifest(manifest_path)
    token_report = load_survival_hr_verification_report(token_report_path)
    identity_report = load_source_verification_report(identity_report_path)
    validate_survival_hr_source_bundle(manifest, token_report)
    validate_survival_hr_identity_bundle(manifest, identity_report)

    records = [_proof_effect_from_mapping(item) for item in bundle["records"]]
    validate_proof_carrying_effects(records)
    _validate_record_alignment(records, manifest, token_report)

    return {
        "schema_version": PROOF_EFFECT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": str(bundle["bundle_id"]),
        "status": str(bundle["status"]),
        "n_records": len(records),
        "effect_type_counts": _counts(record.effect_type for record in records),
        "source_type_counts": _counts(record.source.source_type for record in records),
        "certification_effect": str(bundle["certification_effect"]),
    }


def summarize_proof_effect_bundle(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Return a compact validation-status summary for one bundle."""

    return validate_proof_effect_bundle(path, repo_root=repo_root)


def build_reported_hr_proof_effect_bundle(
    *,
    manifest_path: str | Path,
    token_report_path: str | Path,
    identity_report_path: str | Path,
    abstracts_by_pmid: dict[str, str],
    checked_at: str,
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Build a proof-carrying bundle from a survival reported-HR manifest."""

    manifest_path = Path(manifest_path)
    token_report_path = Path(token_report_path)
    identity_report_path = Path(identity_report_path)
    manifest = load_survival_hr_manifest(manifest_path)
    token_report = load_survival_hr_verification_report(token_report_path)
    identity_report = load_source_verification_report(identity_report_path)
    validate_survival_hr_source_bundle(manifest, token_report)
    validate_survival_hr_identity_bundle(manifest, identity_report)
    token_by_study = {record.study_id: record for record in token_report.records}

    records = []
    for study in manifest.studies:
        abstract = abstracts_by_pmid.get(study.pmid, "")
        if not abstract:
            raise ProofEffectBundleError(f"{study.study_id}: missing PubMed abstract text.")
        abstract_hash = hashlib.sha256(abstract.encode("utf-8")).hexdigest()
        token_record = token_by_study[study.study_id]
        if abstract_hash != token_record.abstract_sha256:
            raise ProofEffectBundleError(f"{study.study_id}: abstract SHA-256 does not match token report.")
        snippet, char_start, char_end = _extract_hr_snippet(
            abstract=abstract,
            hr=study.reported_hr,
            ci_lower=study.ci_lower,
            ci_upper=study.ci_upper,
        )
        record = ProofCarryingEffectRecord(
            record_id=f"{manifest.benchmark_id}:{study.study_id}:reported_hr",
            study_id=study.study_id,
            outcome_name=study.outcome_label,
            effect_type="HR",
            point_estimate=float(study.reported_hr),
            ci_lower=float(study.ci_lower),
            ci_upper=float(study.ci_upper),
            standard_error=None,
            p_value=None,
            timepoint=None,
            is_primary=None,
            is_subgroup=False,
            computation_origin="reported",
            source=EvidenceIngestionRecord(
                row_id=f"{manifest.benchmark_id}:{study.study_id}:pubmed_abstract",
                source_type="pubmed_abstract",
                url=study.source_url,
                access_statement=study.access_statement,
                pmid=study.pmid,
                nct_id=study.nct_id,
            ),
            provenance=ExtractionProvenance(
                source_text=snippet,
                source_type="text",
                char_start=char_start,
                char_end=char_end,
            ),
        )
        records.append(record.to_dict())

    return {
        "schema_version": PROOF_EFFECT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id or f"{manifest.benchmark_id}_proof_effects",
        "checked_at": checked_at,
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "status": "local_pass",
        "certification_effect": "none",
        "source_manifest": manifest_path.as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "token_verification_report": token_report_path.as_posix(),
        "token_verification_sha256": sha256_file(token_report_path),
        "identity_verification_report": identity_report_path.as_posix(),
        "identity_verification_sha256": sha256_file(identity_report_path),
        "records": records,
        "limitations": [
            "proof-carrying records validate source snippets and uncertainty only",
            "reported PubMed abstract HRs are not Kaplan-Meier reconstruction",
            "bundle does not certify model performance or tier-one parity",
        ],
    }


def _proof_effect_from_mapping(raw: dict[str, Any]) -> ProofCarryingEffectRecord:
    _require_keys(
        raw,
        {
            "record_id",
            "study_id",
            "outcome_name",
            "effect_type",
            "point_estimate",
            "source",
            "provenance",
        },
        label="proof-effect record",
    )
    source = raw["source"]
    provenance = raw["provenance"]
    if not isinstance(source, dict) or not isinstance(provenance, dict):
        raise ProofEffectBundleError("proof-effect source and provenance must be objects.")
    return ProofCarryingEffectRecord(
        record_id=str(raw["record_id"]),
        study_id=str(raw["study_id"]),
        outcome_name=str(raw["outcome_name"]),
        effect_type=str(raw["effect_type"]),
        point_estimate=float(raw["point_estimate"]),
        ci_lower=_optional_float(raw.get("ci_lower")),
        ci_upper=_optional_float(raw.get("ci_upper")),
        standard_error=_optional_float(raw.get("standard_error")),
        p_value=_optional_float(raw.get("p_value")),
        timepoint=_optional_str(raw.get("timepoint")),
        is_primary=raw.get("is_primary") if isinstance(raw.get("is_primary"), bool) else None,
        is_subgroup=bool(raw.get("is_subgroup", False)),
        computation_origin=str(raw.get("computation_origin") or "reported"),
        source=EvidenceIngestionRecord(
            row_id=str(source["row_id"]),
            source_type=str(source["source_type"]),
            url=str(source["url"]),
            access_statement=str(source["access_statement"]),
            pmid=_optional_str(source.get("pmid")),
            nct_id=_optional_str(source.get("nct_id")),
            pmcid=_optional_str(source.get("pmcid")),
            doi=_optional_str(source.get("doi")),
            registry_id=_optional_str(source.get("registry_id")),
            source_text=_optional_str(source.get("source_text")) or "",
        ),
        provenance=ExtractionProvenance(
            source_text=str(provenance["source_text"]),
            source_type=str(provenance["source_type"]),
            page_number=_optional_int(provenance.get("page_number")),
            char_start=_optional_int(provenance.get("char_start")),
            char_end=_optional_int(provenance.get("char_end")),
            figure_label=_optional_str(provenance.get("figure_label")),
            table_label=_optional_str(provenance.get("table_label")),
        ),
    )


def _validate_record_alignment(records: list[ProofCarryingEffectRecord], manifest: Any, token_report: Any) -> None:
    studies_by_id = {study.study_id: study for study in manifest.studies}
    token_by_id = {record.study_id: record for record in token_report.records}
    if {record.study_id for record in records} != set(studies_by_id):
        raise ProofEffectBundleError("proof-effect records do not match manifest studies.")
    for record in records:
        study = studies_by_id[record.study_id]
        token = token_by_id[record.study_id]
        if record.source.pmid != study.pmid:
            raise ProofEffectBundleError(f"{record.study_id}: PMID does not match manifest.")
        if record.source.nct_id != study.nct_id:
            raise ProofEffectBundleError(f"{record.study_id}: NCT ID does not match manifest.")
        if record.outcome_name != study.outcome_label:
            raise ProofEffectBundleError(f"{record.study_id}: outcome label does not match manifest.")
        if record.effect_type != "HR":
            raise ProofEffectBundleError(f"{record.study_id}: reported survival bundle requires HR effects.")
        if str(record.point_estimate) != str(float(study.reported_hr)):
            raise ProofEffectBundleError(f"{record.study_id}: point estimate does not match manifest HR.")
        if str(record.ci_lower) != str(float(study.ci_lower)) or str(record.ci_upper) != str(float(study.ci_upper)):
            raise ProofEffectBundleError(f"{record.study_id}: confidence interval does not match manifest.")
        snippet_norm = _normalise_for_search(record.provenance.source_text)
        for token_value in (token.reported_hr, token.ci_lower, token.ci_upper):
            if _normalise_for_search(token_value) not in snippet_norm:
                raise ProofEffectBundleError(f"{record.study_id}: proof snippet is missing token {token_value}.")


def _extract_hr_snippet(*, abstract: str, hr: str, ci_lower: str, ci_upper: str) -> tuple[str, int, int]:
    text = " ".join(abstract.split())
    lowered = text.lower()
    anchor_positions = [match.start() for match in re.finditer("hazard ratio", lowered)]
    for anchor in anchor_positions:
        window_end = min(len(text), anchor + 240)
        token_bounds = [
            _find_token_after(lowered, token.lower(), anchor, window_end)
            for token in (hr, ci_lower, ci_upper)
        ]
        if any(bounds is None for bounds in token_bounds):
            continue
        start = anchor
        end = _advance_to_word_boundary(text, max(bounds[1] for bounds in token_bounds if bounds))
        snippet = text[start:end].strip()
        normalized = _normalise_for_search(snippet)
        if all(_normalise_for_search(token) in normalized for token in (hr, ci_lower, ci_upper)):
            return snippet, start, end
    raise ProofEffectBundleError("could not find a hazard-ratio snippet containing HR and CI tokens.")


def _find_token_after(text: str, token: str, start: int, end: int) -> tuple[int, int] | None:
    index = text.find(token, start, end)
    if index < 0:
        return None
    return index, index + len(token)


def _advance_to_word_boundary(text: str, end: int) -> int:
    while end < len(text) and not text[end].isspace():
        end += 1
    return min(end, len(text))


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ProofEffectBundleError(f"{label} does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ProofEffectBundleError(f"{label} SHA-256 mismatch: {observed} != {expected}")


def _require_keys(raw: dict[str, Any], required: set[str], *, label: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        raise ProofEffectBundleError(f"{label} missing required keys: {missing}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", "")).strip()


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
