"""Coverage atlas for current source-backed real benchmark artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.benchmark_registry import (
    assert_registry_covers_source_backed_artifacts,
    validate_source_benchmark_registry,
)
from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES, PROTOCOL_ONLY_SOURCE_TYPES


REAL_BENCHMARK_ATLAS_SCHEMA_VERSION = "real_benchmark_atlas/v1"


def build_real_benchmark_atlas(
    repo_root: str | Path,
    *,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Build a non-certifying coverage atlas from registered real benchmarks."""

    root = Path(repo_root).resolve()
    registry_path = root / "validation" / "benchmark_registry.toml"
    registry = validate_source_benchmark_registry(registry_path, repo_root=root)
    assert_registry_covers_source_backed_artifacts(registry, repo_root=root)

    domain_counts: dict[str, int] = {}
    evidence_mode_counts: dict[str, int] = {}
    source_policy_counts: dict[str, int] = {}
    effect_scale_counts: dict[str, int] = {}
    source_check_schema_counts: dict[str, int] = {}
    source_check_scope_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    unique_nct_ids: set[str] = set()
    unique_pmids: set[str] = set()
    unique_study_ids: set[str] = set()
    n_study_effects = 0
    benchmark_summaries: list[dict[str, Any]] = []

    for entry in registry.benchmarks:
        artifact = _load_toml(root / entry.artifact_path)
        _count(domain_counts, entry.domain)
        _count(evidence_mode_counts, entry.evidence_mode)
        _count(source_policy_counts, entry.source_policy)
        _count(effect_scale_counts, str(artifact.get("effect_scale", "not_reported")))

        study_effects = artifact.get("study_effects", [])
        if isinstance(study_effects, list):
            n_study_effects += len(study_effects)
            for study in study_effects:
                if not isinstance(study, dict):
                    continue
                _add_if_present(unique_study_ids, study.get("study_id"))
                _add_if_present(unique_nct_ids, study.get("nct_id"))
                _add_if_present(unique_pmids, study.get("pmid"))

        for source_check in entry.source_checks:
            payload = _load_json(root / source_check)
            schema_version = str(payload.get("schema_version", "not_reported"))
            _count(source_check_schema_counts, schema_version)
            for record in payload.get("records", []):
                if not isinstance(record, dict):
                    continue
                scope = str(record.get("evidence_scope", "not_reported"))
                _count(source_check_scope_counts, scope)
                source_type = _source_type_from_record(record)
                if source_type:
                    _count(source_type_counts, source_type)

        benchmark_summaries.append(
            {
                "id": entry.id,
                "domain": entry.domain,
                "artifact": entry.artifact_path,
                "artifact_sha256": entry.artifact_sha256,
                "artifact_schema_version": entry.artifact_schema_version,
                "artifact_status": entry.artifact_status,
                "source_policy": entry.source_policy,
                "evidence_mode": entry.evidence_mode,
                "effect_scale": str(artifact.get("effect_scale", "not_reported")),
                "n_studies": entry.n_studies,
                "n_study_effects": len(study_effects) if isinstance(study_effects, list) else 0,
                "source_manifests": list(entry.source_manifests),
                "source_checks": list(entry.source_checks),
                "required_limitations": list(entry.required_limitations),
                "certification_effect": entry.certification_effect,
            }
        )

    return {
        "schema_version": REAL_BENCHMARK_ATLAS_SCHEMA_VERSION,
        "status": "passed",
        "checked_at": checked_at or _utc_now(),
        "certification_effect": "none",
        "allowed_effect_evidence_sources": sorted(EFFECT_EVIDENCE_SOURCE_TYPES),
        "allowed_protocol_only_sources": sorted(PROTOCOL_ONLY_SOURCE_TYPES),
        "protocol_registry_rule": (
            "Protocol-only registry sources may verify registration, planned outcomes, "
            "eligibility, and dates, but cannot supply model-ready effects; downloaded "
            "ICTRP or PACTR result rows may supply effects only when public numeric "
            "result text is source-bound."
        ),
        "registry": "validation/benchmark_registry.toml",
        "n_benchmarks": len(registry.benchmarks),
        "n_benchmark_study_effects": n_study_effects,
        "n_unique_study_ids": len(unique_study_ids),
        "n_unique_nct_ids": len(unique_nct_ids),
        "n_unique_pmids": len(unique_pmids),
        "domain_counts": _sorted_counts(domain_counts),
        "evidence_mode_counts": _sorted_counts(evidence_mode_counts),
        "source_policy_counts": _sorted_counts(source_policy_counts),
        "effect_scale_counts": _sorted_counts(effect_scale_counts),
        "source_check_schema_counts": _sorted_counts(source_check_schema_counts),
        "source_check_scope_counts": _sorted_counts(source_check_scope_counts),
        "source_type_counts": _sorted_counts(source_type_counts),
        "benchmarks": benchmark_summaries,
        "coverage_claim": (
            "This atlas proves only that registered local real-data benchmark artifacts "
            "are source-governed and reproducible within their stated scope."
        ),
        "does_not_prove": [
            "tier-one parity",
            "clinical superiority",
            "production certification",
            "closed-loop inconsistency performance",
            "Kaplan-Meier reconstruction accuracy",
            "ML-NMR, broad dose-response NMA, component, or cross-design synthesis performance",
        ],
        "required_next_gates": [
            "external reference runs for every applicable module",
            "open-access Kaplan-Meier curve artifacts before KM reconstruction claims",
            "closed-loop source-backed networks before inconsistency-performance claims",
            "prespecified simulation expansion before statistical superiority claims",
        ],
    }


def summarize_real_benchmark_atlas(atlas: dict[str, Any]) -> dict[str, Any]:
    """Return validation-status-friendly atlas fields."""

    return {
        "schema_version": atlas["schema_version"],
        "status": atlas["status"],
        "n_benchmarks": atlas["n_benchmarks"],
        "n_benchmark_study_effects": atlas["n_benchmark_study_effects"],
        "n_unique_nct_ids": atlas["n_unique_nct_ids"],
        "n_unique_pmids": atlas["n_unique_pmids"],
        "domain_counts": atlas["domain_counts"],
        "source_type_counts": atlas["source_type_counts"],
        "certification_effect": atlas["certification_effect"],
    }


def write_real_benchmark_atlas(
    repo_root: str | Path,
    output: str | Path,
    *,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Build and write the atlas JSON with deterministic formatting."""

    atlas = build_real_benchmark_atlas(repo_root, checked_at=checked_at)
    Path(output).write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return atlas


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_type_from_record(record: dict[str, Any]) -> str | None:
    source_type = record.get("source_type")
    if isinstance(source_type, str) and source_type:
        return source_type
    scope = str(record.get("evidence_scope", ""))
    if scope.startswith("pubmed_abstract"):
        return "pubmed_abstract"
    if scope.startswith("clinicaltrials_gov"):
        return "clinicaltrials_gov"
    return None


def _add_if_present(values: set[str], value: Any) -> None:
    if value is not None and str(value).strip():
        values.add(str(value))


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
