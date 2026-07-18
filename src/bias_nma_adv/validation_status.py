"""Unified validation-status reporting for certification gates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
from bias_nma_adv.dta_coverage import (
    load_dta_source_coverage,
    summarize_dta_source_coverage,
)
from bias_nma_adv.feature_parity_matrix import (
    load_feature_parity_matrix,
    summarize_feature_parity_matrix,
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
from bias_nma_adv.large_scale_validation import (
    load_large_scale_validation_gate,
    summarize_large_scale_validation,
)
from bias_nma_adv.mlnmr_coverage import (
    load_mlnmr_source_coverage,
    summarize_mlnmr_source_coverage,
)
from bias_nma_adv.portfolio_reuse import (
    load_portfolio_reuse_registry,
    summarize_portfolio_reuse_registry,
)
from bias_nma_adv.proof_effect_bundle import summarize_proof_effect_bundle
from bias_nma_adv.real_benchmark_atlas import (
    build_real_benchmark_atlas,
    summarize_real_benchmark_atlas,
)
from bias_nma_adv.reversal_yardstick import (
    load_reversal_yardstick,
    summarize_reversal_yardstick,
)
from bias_nma_adv.review_ledger import summarize_review_ledger
from bias_nma_adv.simulation_matrix import (
    load_simulation_matrix_report,
    summarize_simulation_matrix,
    summarize_simulation_matrix_report,
    validate_simulation_matrix_report,
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
    source_artifact_paths: Mapping[str, str | Path] | None = None,
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
    simulation_full_report_path = root / "validation" / "simulation_full_report.json"
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
    dta_coverage_path = root / "validation" / "dta_source_coverage.toml"
    mlnmr_coverage_path = root / "validation" / "mlnmr_source_coverage.toml"
    reversal_yardstick_path = root / "validation" / "reversal_yardstick.toml"
    feature_parity_matrix_path = root / "validation" / "feature_parity_matrix.toml"
    large_scale_validation_path = root / "validation" / "large_scale_validation.toml"

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
    simulation_full_report = None
    if simulation_full_report_path.exists():
        simulation_full_report = validate_simulation_matrix_report(
            load_simulation_matrix_report(simulation_full_report_path),
            simulation_matrix,
        )
    portfolio_reuse_registry = load_portfolio_reuse_registry(portfolio_reuse_registry_path)
    tier1_gap_register = load_tier1_gap_register(tier1_gap_register_path)
    html_delivery_contract = load_html_delivery_contract(html_delivery_contract_path)
    dose_response_coverage = load_dose_response_source_coverage(
        dose_response_coverage_path
    )
    dta_coverage = load_dta_source_coverage(dta_coverage_path)
    mlnmr_coverage = load_mlnmr_source_coverage(mlnmr_coverage_path)
    feature_parity_matrix = load_feature_parity_matrix(feature_parity_matrix_path)
    large_scale_validation_gate = load_large_scale_validation_gate(
        large_scale_validation_path
    )
    reversal_yardstick = load_reversal_yardstick(reversal_yardstick_path)
    reversal_pin_report = (
        reversal_yardstick.verify_source_artifact_pins(source_artifact_paths)
        if source_artifact_paths is not None
        else None
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
            "eligibility, dates, registered-primary-outcome anchors, and completeness "
            "denominators, but cannot supply model-ready effects; downloaded ICTRP or "
            "PACTR result rows and public FDA/EMA regulatory review rows may supply "
            "effects only when public numeric result text is source-bound."
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
        "simulation_full_report": _simulation_full_report_summary(
            simulation_full_report,
            report_path=simulation_full_report_path,
            repo_root=root,
        ),
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
        "dta_source_coverage": {
            "coverage": _relpath(dta_coverage_path, root),
            **summarize_dta_source_coverage(dta_coverage),
        },
        "mlnmr_source_coverage": {
            "coverage": _relpath(mlnmr_coverage_path, root),
            **summarize_mlnmr_source_coverage(mlnmr_coverage),
        },
        "feature_parity_matrix": {
            "matrix": _relpath(feature_parity_matrix_path, root),
            **summarize_feature_parity_matrix(feature_parity_matrix),
        },
        "large_scale_validation": {
            "gate": _relpath(large_scale_validation_path, root),
            **summarize_large_scale_validation(
                large_scale_validation_gate,
                real_benchmark_atlas=real_benchmark_atlas,
                simulation_matrix=simulation_matrix,
                reference_reports=reports,
                simulation_report=simulation_full_report,
                formal_required_domain_exclusions=(
                    mlnmr_coverage.large_scale_domain_exclusions()
                ),
            ),
        },
        "reversal_yardstick": {
            "yardstick": _relpath(reversal_yardstick_path, root),
            **summarize_reversal_yardstick(reversal_yardstick),
            **(
                {"source_artifact_pins": reversal_pin_report}
                if reversal_pin_report is not None
                else {}
            ),
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


def _simulation_full_report_summary(
    report: dict[str, Any] | None,
    *,
    report_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if report is None:
        return {
            "report": _relpath(report_path, repo_root),
            "status": "absent",
            "full_validation_jobs": 0,
            "full_validation_iterations_successful": 0,
            "certification_effect": "none",
        }
    return {
        "report": _relpath(report_path, repo_root),
        **summarize_simulation_matrix_report(report),
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
