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
from bias_nma_adv.dose_response_coverage import (
    load_dose_response_source_coverage,
    summarize_dose_response_source_coverage,
)
from bias_nma_adv.grand_benchmark_plan import (
    summarize_grand_benchmark_plan,
    validate_grand_benchmark_plan,
)
from bias_nma_adv.html_delivery_contract import (
    load_html_delivery_contract,
    summarize_html_delivery_contract,
)
from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES, PROTOCOL_ONLY_SOURCE_TYPES
from bias_nma_adv.improvement_review import (
    load_improvement_review,
    summarize_improvement_review,
)
from bias_nma_adv.ingestion import summarize_proof_carrying_ingestion_contract
from bias_nma_adv.portfolio_reuse import (
    load_portfolio_reuse_registry,
    summarize_portfolio_reuse_registry,
)
from bias_nma_adv.proof_effect_bundle import summarize_proof_effect_bundle
from bias_nma_adv.real_benchmark_atlas import (
    build_real_benchmark_atlas,
    summarize_real_benchmark_atlas,
)
from bias_nma_adv.review_ledger import summarize_review_ledger
from bias_nma_adv.simulation_matrix import (
    summarize_simulation_matrix,
    validate_simulation_matrix,
)
from bias_nma_adv.tier1_gap_register import (
    load_tier1_gap_register,
    summarize_tier1_gap_register,
)


VALIDATION_STATUS_SCHEMA_VERSION = "validation_status/v1"
ALLOWED_EVIDENCE_SOURCES = tuple(sorted(EFFECT_EVIDENCE_SOURCE_TYPES))
ALLOWED_PROTOCOL_ONLY_SOURCES = tuple(sorted(PROTOCOL_ONLY_SOURCE_TYPES))
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
    simulation_matrix_path = root / "validation" / "simulation_matrix.toml"
    portfolio_reuse_registry_path = root / "validation" / "portfolio_reuse_sources.toml"
    proof_effect_bundle_path = (
        root / "validation" / "ingestion" / "sglt2_hf_reported_hr_proof_effects.json"
    )
    review_ledger_path = (
        root / "validation" / "reviews" / "multiperson_review_2026_07_15.toml"
    )
    improvement_review_path = (
        root / "validation" / "reviews" / "improvement_review_2026_07_15.toml"
    )
    reference_targets_path = root / "validation" / "reference_targets.toml"
    reference_runs_path = root / "validation" / "reference_runs"
    real_benchmark_atlas_path = root / "validation" / "real_benchmark_atlas.json"
    tier1_gap_register_path = root / "validation" / "tier1_gap_register.toml"
    html_delivery_contract_path = root / "validation" / "html_delivery_contract.toml"
    dose_response_coverage_path = root / "validation" / "dose_response_source_coverage.toml"

    registry = validate_source_benchmark_registry(registry_path, repo_root=root)
    assert_registry_covers_source_backed_artifacts(registry, repo_root=root)
    grand_benchmark_plan = validate_grand_benchmark_plan(
        grand_benchmark_plan_path,
        source_registry=registry,
    )
    real_benchmark_atlas = build_real_benchmark_atlas(root, checked_at=checked_at)
    simulation_matrix = validate_simulation_matrix(
        simulation_matrix_path,
        grand_benchmark_plan_path=grand_benchmark_plan_path,
    )
    portfolio_reuse_registry = load_portfolio_reuse_registry(portfolio_reuse_registry_path)
    tier1_gap_register = load_tier1_gap_register(tier1_gap_register_path)
    html_delivery_contract = load_html_delivery_contract(html_delivery_contract_path)
    dose_response_coverage = load_dose_response_source_coverage(
        dose_response_coverage_path
    )
    improvement_review = load_improvement_review(improvement_review_path)

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
        "allowed_effect_evidence_sources": list(ALLOWED_EVIDENCE_SOURCES),
        "allowed_protocol_only_sources": list(ALLOWED_PROTOCOL_ONLY_SOURCES),
        "protocol_registry_rule": (
            "Protocol-only registry sources may verify registration, planned outcomes, "
            "eligibility, and dates, but cannot supply model-ready effects."
        ),
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
        "real_benchmark_atlas": {
            "atlas": _relpath(real_benchmark_atlas_path, root),
            **summarize_real_benchmark_atlas(real_benchmark_atlas),
        },
        "simulation_matrix": {
            "matrix": _relpath(simulation_matrix_path, root),
            **summarize_simulation_matrix(simulation_matrix),
        },
        "portfolio_reuse": {
            "registry": _relpath(portfolio_reuse_registry_path, root),
            **summarize_portfolio_reuse_registry(portfolio_reuse_registry),
        },
        "tier1_gap_register": {
            "register": _relpath(tier1_gap_register_path, root),
            **summarize_tier1_gap_register(tier1_gap_register),
        },
        "html_delivery_contract": {
            "contract": _relpath(html_delivery_contract_path, root),
            **summarize_html_delivery_contract(html_delivery_contract),
        },
        "dose_response_source_coverage": {
            "coverage": _relpath(dose_response_coverage_path, root),
            **summarize_dose_response_source_coverage(dose_response_coverage),
        },
        "ingestion_contract": summarize_proof_carrying_ingestion_contract(),
        "proof_effect_bundle": {
            "bundle": _relpath(proof_effect_bundle_path, root),
            **summarize_proof_effect_bundle(proof_effect_bundle_path, repo_root=root),
        },
        "multiperson_review": {
            "ledger": _relpath(review_ledger_path, root),
            **summarize_review_ledger(review_ledger_path),
        },
        "improvement_review": {
            "ledger": _relpath(improvement_review_path, root),
            **summarize_improvement_review(improvement_review),
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
