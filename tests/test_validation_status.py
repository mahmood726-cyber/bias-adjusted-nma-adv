import json
from pathlib import Path
import subprocess
import sys

from bias_nma_adv.validation_status import (
    NO_PRODUCTION_CERTIFIED_MESSAGE,
    VALIDATION_STATUS_SCHEMA_VERSION,
    build_validation_status,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_validation_status.py"


def test_validation_status_composes_all_current_gates():
    report = build_validation_status(ROOT, checked_at="2026-07-15T00:00:00Z")

    assert report["schema_version"] == VALIDATION_STATUS_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["checked_at"] == "2026-07-15T00:00:00Z"
    assert report["allowed_evidence_sources"] == [
        "clinicaltrials_gov",
        "open_access_paper",
        "pubmed_abstract",
    ]
    assert report["allowed_effect_evidence_sources"] == report["allowed_evidence_sources"]
    assert report["allowed_protocol_only_sources"] == [
        "other_trial_registry_protocol",
        "who_ictrp_protocol",
    ]
    assert "cannot supply model-ready effects" in report["protocol_registry_rule"]
    assert report["certification_effect"] == "none"

    assert report["clinical_hta_reporting_enabled"] is False
    assert report["clinical_hta_reporting_reason"] == NO_PRODUCTION_CERTIFIED_MESSAGE
    assert report["production_certified_modules"] == []

    source_registry = report["source_benchmark_registry"]
    assert source_registry["registry"] == "validation/benchmark_registry.toml"
    assert source_registry["n_benchmarks"] == 4
    assert source_registry["certification_effect"] == "none"
    assert set(source_registry["benchmark_ids"]) == {
        "sglt2_hf_primary_log_or",
        "sglt2_hf_reported_hr",
        "pcsk9_mace_reported_hr",
        "t2d_mace_ctgov_hr_network",
    }

    grand_plan = report["grand_benchmark_plan"]
    assert grand_plan["plan"] == "validation/grand_benchmark_plan.toml"
    assert grand_plan["n_real_data_lanes"] == 3
    assert grand_plan["real_data_lane_status_counts"] == {"active": 3}
    assert grand_plan["n_simulation_scenarios"] == 3
    assert grand_plan["simulation_scenario_status_counts"] == {"planned": 3}
    assert grand_plan["allowed_protocol_only_sources"] == [
        "other_trial_registry_protocol",
        "who_ictrp_protocol",
    ]
    assert grand_plan["certification_effect"] == "none"

    real_atlas = report["real_benchmark_atlas"]
    assert real_atlas["atlas"] == "validation/real_benchmark_atlas.json"
    assert real_atlas["schema_version"] == "real_benchmark_atlas/v1"
    assert real_atlas["status"] == "passed"
    assert real_atlas["n_benchmarks"] == 4
    assert real_atlas["n_benchmark_study_effects"] == 20
    assert real_atlas["n_unique_nct_ids"] == 16
    assert real_atlas["n_unique_pmids"] == 6
    assert real_atlas["domain_counts"] == {
        "binary_pairwise_meta": 1,
        "reported_hr_star_network": 1,
        "reported_survival_hr_pairwise": 2,
    }
    assert real_atlas["source_type_counts"] == {
        "clinicaltrials_gov": 20,
        "pubmed_abstract": 20,
    }
    assert real_atlas["certification_effect"] == "none"

    simulation_matrix = report["simulation_matrix"]
    assert simulation_matrix["matrix"] == "validation/simulation_matrix.toml"
    assert simulation_matrix["n_jobs"] == 1
    assert simulation_matrix["job_status_counts"] == {"active": 1}
    assert simulation_matrix["execution_mode_counts"] == {"smoke": 1}
    assert simulation_matrix["certification_effect"] == "none"

    portfolio_reuse = report["portfolio_reuse"]
    assert portfolio_reuse["registry"] == "validation/portfolio_reuse_sources.toml"
    assert portfolio_reuse["n_sources"] == 8
    assert portfolio_reuse["priority_counts"] == {"high": 4, "medium": 4}
    assert portfolio_reuse["certification_effect"] == "none"
    assert portfolio_reuse["required_review_rounds"] == [
        "source_boundary_review",
        "statistical_methods_review",
        "implementation_contract_review",
        "claims_governance_review",
    ]

    tier1_gaps = report["tier1_gap_register"]
    assert tier1_gaps["register"] == "validation/tier1_gap_register.toml"
    assert tier1_gaps["schema_version"] == "tier1_gap_register/v1"
    assert tier1_gaps["n_gaps"] == 3
    assert tier1_gaps["status_counts"] == {"blocking": 3}
    assert tier1_gaps["gap_ids"] == [
        "feature_completeness",
        "numerical_stability",
        "bayesian_ecosystem_integration",
    ]
    assert tier1_gaps["implemented_capabilities"] == {
        "feature_completeness": [
            "multiarm_prefit_design_diagnostic",
            "multiarm_gls_influence_leverage_diagnostics",
            "multiarm_gls_absolute_mapping_contribution_diagnostics",
            "multiarm_study_contribution_matrix_diagnostic",
            "multiarm_heatmap_ready_contribution_matrix",
            "pairwise_leave_one_out_outlier_space_diagnostic",
            "pairwise_exhaustive_gosh_subset_diagnostic",
            "bounded_trim_and_fill_sensitivity_screen",
            "fixed_effect_node_splitting_smoke_diagnostics",
            "egger_small_study_effect_diagnostic",
            "selection_weight_publication_bias_sensitivity",
        ],
        "numerical_stability": [
            "positive_definite_covariance_fail_closed_policy",
            "multiarm_deterministic_failure_reports",
            "pairwise_alternative_tau2_cross_checks",
            "pairwise_sparse_dominant_study_stress_report",
            "pairwise_reml_local_minimum_profile_diagnostic",
            "pairwise_optimizer_stress_matrix",
        ],
        "bayesian_ecosystem_integration": [
            "local_mcmc_rhat_ess_mcse_diagnostic_warnings",
            "cmdstan_preflight_fail_closed_contract",
            "local_prior_predictive_check",
            "local_posterior_predictive_check",
            "joint_posterior_ranking_draws_from_local_mcmc",
        ],
    }
    assert "tier_one_superiority" in tier1_gaps["blocked_claims"]
    assert tier1_gaps["certification_effect"] == "none"

    html_contract = report["html_delivery_contract"]
    assert html_contract["contract"] == "validation/html_delivery_contract.toml"
    assert html_contract["schema_version"] == "html_delivery_contract/v1"
    assert html_contract["n_capabilities"] == 6
    assert html_contract["delivery_mode_counts"] == {
        "backend_required": 4,
        "static_html_allowed": 2,
    }
    assert html_contract["status_counts"] == {
        "allowed": 2,
        "blocked_for_html_only": 4,
    }
    assert "statistical_estimation_engine" in html_contract["html_only_blocked_ids"]
    assert html_contract["certification_effect"] == "none"

    dose_response_coverage = report["dose_response_source_coverage"]
    assert dose_response_coverage["coverage"] == (
        "validation/dose_response_source_coverage.toml"
    )
    assert dose_response_coverage["schema_version"] == "dose_response_source_coverage/v1"
    assert dose_response_coverage["status"] == "missing_source_backed_dose_response_data"
    assert dose_response_coverage["registered_benchmark_ids"] == []
    assert dose_response_coverage["registered_source_counts"] == {
        "clinicaltrials_gov": 0,
        "open_access_paper": 0,
        "pubmed_abstract": 0,
    }
    assert dose_response_coverage["has_source_backed_dose_response_data"] is False
    assert "MBNMAdose_reference_run_before_certification" in dose_response_coverage[
        "required_next_artifacts"
    ]
    assert dose_response_coverage["certification_effect"] == "none"

    ingestion_contract = report["ingestion_contract"]
    assert ingestion_contract["schema_version"] == "proof_carrying_effect/v1"
    assert ingestion_contract["requires_source_identity"] is True
    assert ingestion_contract["requires_source_snippet"] is True
    assert ingestion_contract["required_uncertainty"] == "complete_ci_or_standard_error"
    assert ingestion_contract["certification_effect"] == "none"
    assert ingestion_contract["allowed_effect_source_types"] == [
        "clinicaltrials_gov",
        "open_access_paper",
        "pubmed_abstract",
    ]
    assert ingestion_contract["protocol_only_source_types"] == [
        "other_trial_registry_protocol",
        "who_ictrp_protocol",
    ]
    assert ingestion_contract["protocol_sources_can_supply_model_effects"] is False
    assert "HR" in ingestion_contract["allowed_effect_types"]
    assert "HR" in ingestion_contract["ratio_effect_types"]

    proof_effect_bundle = report["proof_effect_bundle"]
    assert proof_effect_bundle["bundle"] == (
        "validation/ingestion/sglt2_hf_reported_hr_proof_effects.json"
    )
    assert proof_effect_bundle["schema_version"] == "proof_effect_bundle/v1"
    assert proof_effect_bundle["bundle_id"] == "sglt2_hf_reported_hr_proof_effects"
    assert proof_effect_bundle["status"] == "local_pass"
    assert proof_effect_bundle["n_records"] == 4
    assert proof_effect_bundle["effect_type_counts"] == {"HR": 4}
    assert proof_effect_bundle["source_type_counts"] == {"pubmed_abstract": 4}
    assert proof_effect_bundle["certification_effect"] == "none"

    multiperson_review = report["multiperson_review"]
    assert multiperson_review["ledger"] == (
        "validation/reviews/multiperson_review_2026_07_15.toml"
    )
    assert multiperson_review["schema_version"] == "multiperson_review/v1"
    assert multiperson_review["n_rounds"] == 4
    assert multiperson_review["status_counts"] == {
        "actioned": 2,
        "tracked_next_gate": 2,
    }
    assert multiperson_review["certification_effect"] == "none"

    improvement_review = report["improvement_review"]
    assert improvement_review["ledger"] == (
        "validation/reviews/improvement_review_2026_07_15.toml"
    )
    assert improvement_review["schema_version"] == "improvement_review/v1"
    assert improvement_review["overall_status"] == (
        "passed_current_milestone_with_global_goal_blockers"
    )
    assert improvement_review["n_rounds"] == 4
    assert improvement_review["status_counts"] == {"passed_current_milestone": 4}
    assert improvement_review["global_goal_complete"] is False
    assert improvement_review["certification_effect"] == "none"

    reference_targets = report["reference_targets"]
    assert reference_targets["registry"] == "validation/reference_targets.toml"
    assert reference_targets["status_counts"] == {
        "planned": reference_targets["n_targets"],
    }
    assert reference_targets["production_certified_target_ids"] == []

    reference_runs = report["reference_runs"]
    assert reference_runs["directory"] == "validation/reference_runs"
    assert reference_runs["n_reports"] == 4
    assert reference_runs["status_counts"] == {"failed": 2, "passed": 2}
    assert set(reference_runs["certification_candidate_artifacts"]) == {
        "validation/reference_runs/pairwise_metafor_meta_output.json",
        "validation/reference_runs/multiarm_netmeta_output.json",
    }
    assert {item["certification_effect"] for item in reference_runs["reports"]} == {
        "none",
        "evidence_candidate",
    }
    assert {
        (item["adapter_id"], item["status"])
        for item in reference_runs["reports"]
    } == {
        ("r_metafor_meta_pairwise_preflight", "failed"),
        ("r_metafor_meta_pairwise_output_validation", "passed"),
        ("r_netmeta_multiarm_preflight", "failed"),
        ("r_netmeta_multiarm_output_validation", "passed"),
    }


def test_write_validation_status_script_outputs_machine_readable_json(tmp_path):
    output = tmp_path / "validation_status.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(ROOT),
            "--checked-at",
            "2026-07-15T00:00:00Z",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validation status written" in completed.stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == VALIDATION_STATUS_SCHEMA_VERSION
    assert payload["status"] == "passed"
    assert payload["checked_at"] == "2026-07-15T00:00:00Z"
    assert payload["clinical_hta_reporting_enabled"] is False
    assert payload["source_benchmark_registry"]["n_benchmarks"] == 4
    assert payload["real_benchmark_atlas"]["n_benchmark_study_effects"] == 20
    assert payload["tier1_gap_register"]["status_counts"] == {"blocking": 3}
    assert payload["html_delivery_contract"]["status_counts"] == {
        "allowed": 2,
        "blocked_for_html_only": 4,
    }
    assert payload["dose_response_source_coverage"][
        "has_source_backed_dose_response_data"
    ] is False
    assert payload["proof_effect_bundle"]["n_records"] == 4
    assert payload["multiperson_review"]["n_rounds"] == 4
    assert payload["improvement_review"]["global_goal_complete"] is False
