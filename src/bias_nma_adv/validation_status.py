"""Unified validation-status reporting for certification gates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bias_nma_adv.benchmark_registry import (
    assert_registry_covers_source_backed_artifacts,
    validate_source_benchmark_registry,
)
from bias_nma_adv.certification import (
    assert_no_unsupported_production_claims,
    assert_reference_runs_target_known,
    certification_candidate_artifacts,
    load_reference_run_reports,
    load_reference_targets,
    summarize_reference_run_reports,
    summarize_reference_targets,
)
from bias_nma_adv.grand_benchmark_plan import (
    summarize_grand_benchmark_plan,
    validate_grand_benchmark_plan,
)


VALIDATION_STATUS_SCHEMA_VERSION = "validation_status/v1"
ALLOWED_EVIDENCE_SOURCES = (
    "clinicaltrials_gov",
    "pubmed_abstract",
    "open_access_paper",
)
NO_PRODUCTION_CERTIFIED_MESSAGE = (
    "No modules in this build currently hold Production Certified status. "
    "Clinical and HTA reporting is disabled."
)


def build_validation_status(
    repo_root: str | Path,
    *,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable report from all current validation gates.

    The status report is a CI/Overmind-friendly summary. It does not certify any
    model by itself; certification still requires passed external reference runs
    and the staged evidence recorded in ``validation/reference_targets.toml``.
    """

    root = Path(repo_root).resolve()
    registry_path = root / "validation" / "benchmark_registry.toml"
    grand_benchmark_plan_path = root / "validation" / "grand_benchmark_plan.toml"
    reference_targets_path = root / "validation" / "reference_targets.toml"
    reference_runs_path = root / "validation" / "reference_runs"

    registry = validate_source_benchmark_registry(registry_path, repo_root=root)
    assert_registry_covers_source_backed_artifacts(registry, repo_root=root)
    grand_benchmark_plan = validate_grand_benchmark_plan(
        grand_benchmark_plan_path,
        source_registry=registry,
    )

    targets = load_reference_targets(reference_targets_path)
    assert_no_unsupported_production_claims(targets)

    reports = load_reference_run_reports(reference_runs_path)
    assert_reference_runs_target_known(targets, reports)

    target_status_counts = summarize_reference_targets(targets)
    reference_run_status_counts = summarize_reference_run_reports(reports)
    candidate_artifacts = certification_candidate_artifacts(reports)
    production_targets = tuple(
        target for target in targets if target.status == "production_certified"
    )
    production_modules = tuple(sorted({target.module for target in production_targets}))

    return {
        "schema_version": VALIDATION_STATUS_SCHEMA_VERSION,
        "status": "passed",
        "checked_at": checked_at or _utc_now(),
        "repository": root.name,
        "allowed_evidence_sources": list(ALLOWED_EVIDENCE_SOURCES),
        "certification_effect": "none",
        "clinical_hta_reporting_enabled": bool(production_targets),
        "clinical_hta_reporting_reason": _clinical_hta_reason(production_targets),
        "production_certified_modules": list(production_modules),
        "source_benchmark_registry": _source_registry_summary(
            registry,
            registry_path=registry_path,
            repo_root=root,
        ),
        "grand_benchmark_plan": {
            "plan": _relpath(grand_benchmark_plan_path, root),
            **summarize_grand_benchmark_plan(grand_benchmark_plan),
        },
        "reference_targets": {
            "registry": _relpath(reference_targets_path, root),
            "n_targets": len(targets),
            "status_counts": target_status_counts,
            "production_certified_target_ids": [
                target.id for target in production_targets
            ],
        },
        "reference_runs": {
            "directory": _relpath(reference_runs_path, root),
            "n_reports": len(reports),
            "status_counts": reference_run_status_counts,
            "certification_candidate_artifacts": list(candidate_artifacts),
            "reports": [
                {
                    "target_id": report.target_id,
                    "adapter_id": report.adapter_id,
                    "reference_method": report.reference_method,
                    "status": report.status,
                    "certification_effect": report.certification_effect,
                    "skip_reason": report.skip_reason,
                }
                for report in reports
            ],
        },
        "limitations": [
            "Local source verification checks public source identity, source tokens, or governed manifests; it is not external reference matching.",
            "Unavailable or failed external reference adapters cannot support tier-one parity, clinical, regulatory, or HTA claims.",
            "Source-backed real-data benchmarks remain regression evidence until a separate external reference run passes with versions, hashes, and tolerance.",
        ],
    }


def _source_registry_summary(
    registry: Any,
    *,
    registry_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    domains: dict[str, int] = {}
    for entry in registry.benchmarks:
        domains[entry.domain] = domains.get(entry.domain, 0) + 1
    return {
        "registry": _relpath(registry_path, repo_root),
        "checked_at": registry.checked_at,
        "n_benchmarks": len(registry.benchmarks),
        "benchmark_ids": [entry.id for entry in registry.benchmarks],
        "domains": dict(sorted(domains.items())),
        "certification_effect": "none",
    }


def _clinical_hta_reason(production_targets: tuple[Any, ...]) -> str:
    if not production_targets:
        return NO_PRODUCTION_CERTIFIED_MESSAGE
    return "At least one module has Production Certified status in the reference-target registry."


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
