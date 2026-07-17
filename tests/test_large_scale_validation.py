from pathlib import Path

import pytest

from bias_nma_adv.certification import load_reference_run_reports
from bias_nma_adv.large_scale_validation import (
    LARGE_SCALE_VALIDATION_SCHEMA_VERSION,
    LargeScaleValidationError,
    LargeScaleValidationGate,
    load_large_scale_validation_gate,
    summarize_large_scale_validation,
)
from bias_nma_adv.real_benchmark_atlas import build_real_benchmark_atlas
from bias_nma_adv.simulation_matrix import validate_simulation_matrix
from bias_nma_adv.simulation_matrix import (
    load_simulation_matrix_report,
    validate_simulation_matrix_report,
)


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "validation" / "large_scale_validation.toml"


def test_large_scale_validation_gate_reports_partial_current_evidence():
    gate = load_large_scale_validation_gate(GATE)
    atlas = build_real_benchmark_atlas(ROOT, checked_at="2026-07-16T00:00:00Z")
    matrix = validate_simulation_matrix(
        ROOT / "validation" / "simulation_matrix.toml",
        grand_benchmark_plan_path=ROOT / "validation" / "grand_benchmark_plan.toml",
    )
    simulation_report = validate_simulation_matrix_report(
        load_simulation_matrix_report(ROOT / "validation" / "simulation_full_report.json"),
        matrix,
    )
    reports = load_reference_run_reports(ROOT / "validation" / "reference_runs")

    summary = summarize_large_scale_validation(
        gate,
        real_benchmark_atlas=atlas,
        simulation_matrix=matrix,
        reference_reports=reports,
        simulation_report=simulation_report,
    )

    assert summary["schema_version"] == LARGE_SCALE_VALIDATION_SCHEMA_VERSION
    assert summary["status"] == "partial_not_large_scale"
    assert summary["global_large_scale_validation_complete"] is False
    assert summary["dynamic_counts"]["source_backed_benchmarks"] == {
        "observed": 20,
        "required": 20,
    }
    assert summary["dynamic_counts"]["benchmark_study_effects"] == {
        "observed": 121,
        "required": 200,
    }
    assert summary["dynamic_counts"]["unique_nct_ids"] == {
        "observed": 65,
        "required": 100,
    }
    assert summary["dynamic_counts"]["unique_pmids"] == {
        "observed": 60,
        "required": 50,
    }
    assert summary["dynamic_counts"]["passed_reference_reports"] == {
        "observed": 12,
        "required": 10,
    }
    assert summary["dynamic_counts"]["tau2_positive_benchmarks"] == {
        "observed": 9,
        "required": 1,
    }
    assert summary["dynamic_counts"]["simulation_jobs"] == {
        "observed": 25,
        "required": 25,
    }
    assert summary["dynamic_counts"]["simulation_iterations"] == {
        "observed": 10588,
        "required": 10000,
    }
    assert "Only passed full jobs from a validated simulation report count" in summary[
        "simulation_counting_rule"
    ]
    assert summary["simulation_evidence_status"] == "passed"
    assert "diagnostic_test_accuracy" not in summary["missing_required_real_domains"]
    assert "component_nma" not in summary["missing_required_real_domains"]
    assert "cross_design_nma" not in summary["missing_required_real_domains"]
    assert summary["missing_required_real_domains"] == ["mlnmr"]
    assert "source_backed_benchmarks" not in summary["failed_checks"]
    assert "benchmark_study_effects" in summary["failed_checks"]
    assert "unique_nct_ids" in summary["failed_checks"]
    assert summary["certification_effect"] == "none"


def test_large_scale_validation_gate_rejects_static_completion_claim():
    gate = load_large_scale_validation_gate(GATE)
    raw = {
        "schema_version": LARGE_SCALE_VALIDATION_SCHEMA_VERSION,
        "checked_at": gate.checked_at,
        "purpose": gate.purpose,
        "source_boundary": gate.source_boundary,
        "certification_effect": gate.certification_effect,
        "global_large_scale_validation_complete": True,
        "claim_limit": gate.claim_limit,
        "thresholds": {
            "minimum_source_backed_benchmarks": gate.thresholds.minimum_source_backed_benchmarks,
            "minimum_benchmark_study_effects": gate.thresholds.minimum_benchmark_study_effects,
            "minimum_unique_nct_ids": gate.thresholds.minimum_unique_nct_ids,
            "minimum_unique_pmids": gate.thresholds.minimum_unique_pmids,
            "minimum_passed_reference_reports": gate.thresholds.minimum_passed_reference_reports,
            "minimum_tau2_positive_benchmarks": gate.thresholds.minimum_tau2_positive_benchmarks,
            "minimum_simulation_jobs": gate.thresholds.minimum_simulation_jobs,
            "minimum_simulation_iterations": gate.thresholds.minimum_simulation_iterations,
            "required_real_domains": list(gate.thresholds.required_real_domains),
        },
    }

    with pytest.raises(LargeScaleValidationError, match="dynamic evidence"):
        LargeScaleValidationGate.from_mapping(raw)
