import copy
from pathlib import Path

import pytest

from bias_nma_adv.tier1_gap_register import (
    REQUIRED_BLOCKED_CLAIMS,
    REQUIRED_GAP_IDS,
    TIER1_GAP_REGISTER_SCHEMA_VERSION,
    Tier1GapRegister,
    Tier1GapRegisterError,
    load_tier1_gap_register,
    summarize_tier1_gap_register,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "validation" / "tier1_gap_register.toml"


def test_tier1_gap_register_keeps_current_shortcomings_blocking():
    register = load_tier1_gap_register(REGISTER)

    assert register.certification_effect == "none"
    assert set(register.blocked_claims) >= REQUIRED_BLOCKED_CLAIMS
    assert {gap.id for gap in register.gaps} == REQUIRED_GAP_IDS
    assert {gap.status for gap in register.gaps} == {"blocking"}

    by_id = {gap.id: gap for gap in register.gaps}
    assert "reference_matched_node_splitting" in by_id["feature_completeness"].missing_capabilities
    assert (
        "redescending_robust_fraud_containment_core"
        in by_id["feature_completeness"].missing_capabilities
    )
    assert (
        "multiarm_prefit_design_diagnostic"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "multiarm_gls_influence_leverage_diagnostics"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "multiarm_gls_absolute_mapping_contribution_diagnostics"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "multiarm_study_contribution_matrix_diagnostic"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "multiarm_heatmap_ready_contribution_matrix"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "pairwise_leave_one_out_outlier_space_diagnostic"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "pairwise_exhaustive_gosh_subset_diagnostic"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "bounded_trim_and_fill_sensitivity_screen"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "fixed_effect_node_splitting_smoke_diagnostics"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "egger_small_study_effect_diagnostic"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "metafor_regtest_source_backed_publication_bias_reference_candidate"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "selection_weight_publication_bias_sensitivity"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "native_python_guyot_reconstruction_check"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "source_backed_dose_response_smoke_benchmark"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "mbnmadose_source_backed_linear_reference_candidate"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "aact_ctgov_ingestion_contract"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "ictrp_pactr_result_source_ingestion_contract"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "fda_ema_regulatory_review_source_tier_contract"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "protocol_only_registry_metadata_ledger"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "config_driven_study_design_policy"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "input_verified_reversal_yardstick_gate"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "reversal_aggregate_answer_key_runner"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert "dta_source_coverage_gate" in by_id["feature_completeness"].implemented_capabilities
    assert "mlnmr_source_coverage_gate" in by_id["feature_completeness"].implemented_capabilities
    assert (
        "dta_bivariate_logitnormal_reml_prototype"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "dta_mada_reitsma_algorithmic_reference_adapter"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "rapidmeta_app_index_fail_closed_adapter_contract"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "evalue_and_binary_fragility_sensitivity"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "pairwise_redescending_outlier_sensitivity"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "feature_parity_matrix_gate"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "component_nma_additive_core_with_estimability_checks"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "netmeta_discomb_component_reference_adapter"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "source_backed_component_smoke_benchmark"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "source_backed_closed_loop_binary_network_benchmark"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "netmeta_source_backed_closed_loop_reference_candidate"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert (
        "reference_matched_optimizer_stress_matrix"
        in by_id["numerical_stability"].missing_capabilities
    )
    assert (
        "positive_definite_covariance_fail_closed_policy"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "multiarm_deterministic_failure_reports"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "pairwise_alternative_tau2_cross_checks"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "pairwise_tau2_sign_and_null_crossing_warnings"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "mixed_measure_outcome_fail_closed"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "nma_dropped_study_warning_counts"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "nma_tau_method_result_reporting"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "sponsor_bias_missing_attrition_conservative_policy"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "pairwise_sparse_dominant_study_stress_report"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "pairwise_reml_local_minimum_profile_diagnostic"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "pairwise_optimizer_stress_matrix"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "ctgov_sparse_design_bias_guard"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "source_backed_cross_design_routing_benchmark"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "crossnma_source_fixture_compatibility_preflight"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "source_backed_closed_loop_multiarm_netmeta_reference_candidate"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "large_scale_validation_evidence_gate"
        in by_id["numerical_stability"].implemented_capabilities
    )
    assert (
        "multinma_reference_matching"
        in by_id["bayesian_ecosystem_integration"].missing_capabilities
    )
    assert (
        "local_mcmc_rhat_ess_mcse_diagnostic_warnings"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "cmdstan_preflight_fail_closed_contract"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "local_prior_predictive_check"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "local_posterior_predictive_check"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "joint_posterior_ranking_draws_from_local_mcmc"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "standard_binary_stan_model_source"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "stan_nuts_cmdstan_reference_preflight_report"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "source_backed_cmdstan_nuts_sglt2i_reference_candidate"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert (
        "stan_nuts_rhat_ess_divergence_treedepth_mcse_exports"
        in by_id["bayesian_ecosystem_integration"].implemented_capabilities
    )
    assert "multinma" in by_id["bayesian_ecosystem_integration"].tier_one_references


def test_tier1_gap_register_summary_is_validation_status_ready():
    summary = summarize_tier1_gap_register(load_tier1_gap_register(REGISTER))

    assert summary == {
        "schema_version": TIER1_GAP_REGISTER_SCHEMA_VERSION,
        "checked_at": "2026-07-17",
        "n_gaps": 3,
        "gap_ids": [
            "feature_completeness",
            "numerical_stability",
            "bayesian_ecosystem_integration",
        ],
        "status_counts": {"blocking": 3},
        "implemented_capabilities": {
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
            "metafor_regtest_source_backed_publication_bias_reference_candidate",
            "selection_weight_publication_bias_sensitivity",
            "native_python_guyot_reconstruction_check",
            "source_backed_dose_response_smoke_benchmark",
            "mbnmadose_source_backed_linear_reference_candidate",
            "aact_ctgov_ingestion_contract",
            "ictrp_pactr_result_source_ingestion_contract",
            "fda_ema_regulatory_review_source_tier_contract",
            "protocol_only_registry_metadata_ledger",
            "config_driven_study_design_policy",
            "input_verified_reversal_yardstick_gate",
            "reversal_aggregate_answer_key_runner",
            "dta_source_coverage_gate",
            "mlnmr_source_coverage_gate",
            "dta_bivariate_logitnormal_reml_prototype",
            "dta_mada_reitsma_algorithmic_reference_adapter",
            "rapidmeta_app_index_fail_closed_adapter_contract",
            "evalue_and_binary_fragility_sensitivity",
            "pairwise_redescending_outlier_sensitivity",
            "feature_parity_matrix_gate",
            "component_nma_additive_core_with_estimability_checks",
            "netmeta_discomb_component_reference_adapter",
            "source_backed_component_smoke_benchmark",
            "source_backed_closed_loop_binary_network_benchmark",
            "netmeta_source_backed_closed_loop_reference_candidate",
        ],
            "numerical_stability": [
                "positive_definite_covariance_fail_closed_policy",
                "multiarm_deterministic_failure_reports",
                "pairwise_alternative_tau2_cross_checks",
                "pairwise_tau2_sign_and_null_crossing_warnings",
                "pairwise_sparse_dominant_study_stress_report",
                "pairwise_reml_local_minimum_profile_diagnostic",
                "pairwise_optimizer_stress_matrix",
                "mixed_measure_outcome_fail_closed",
                "nma_dropped_study_warning_counts",
                "nma_tau_method_result_reporting",
                "sponsor_bias_missing_attrition_conservative_policy",
                "ctgov_sparse_design_bias_guard",
                "source_backed_cross_design_routing_benchmark",
                "crossnma_source_fixture_compatibility_preflight",
                "source_backed_closed_loop_multiarm_netmeta_reference_candidate",
                "large_scale_validation_evidence_gate",
            ],
            "bayesian_ecosystem_integration": [
                "local_mcmc_rhat_ess_mcse_diagnostic_warnings",
                "cmdstan_preflight_fail_closed_contract",
                "local_prior_predictive_check",
                "local_posterior_predictive_check",
                "joint_posterior_ranking_draws_from_local_mcmc",
                "standard_binary_stan_model_source",
                "stan_nuts_cmdstan_reference_preflight_report",
                "source_backed_cmdstan_nuts_sglt2i_reference_candidate",
                "stan_nuts_rhat_ess_divergence_treedepth_mcse_exports",
                "source_backed_multinma_sglt2i_reference_candidate",
                "multinma_rstan_rhat_ess_divergence_treedepth_exports",
            ],
        },
        "blocked_claims": [
            "tier_one_parity",
            "tier_one_superiority",
            "production_certification",
            "clinical_reporting",
            "hta_reporting",
        ],
        "certification_effect": "none",
    }


def test_tier1_gap_register_rejects_removed_gap_or_softened_status():
    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["gaps"] = [gap for gap in raw["gaps"] if gap["id"] != "feature_completeness"]

    with pytest.raises(Tier1GapRegisterError, match="feature_completeness"):
        Tier1GapRegister.from_mapping(raw)

    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["gaps"][0]["status"] = "resolved"

    with pytest.raises(Tier1GapRegisterError, match="resolved gaps require"):
        Tier1GapRegister.from_mapping(raw)


def test_tier1_gap_register_rejects_certification_or_missing_blocked_claim():
    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["certification_effect"] = "production_certified"

    with pytest.raises(Tier1GapRegisterError, match="cannot certify"):
        Tier1GapRegister.from_mapping(raw)

    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["blocked_claims"] = [
        claim for claim in raw["blocked_claims"] if claim != "tier_one_superiority"
    ]

    with pytest.raises(Tier1GapRegisterError, match="tier_one_superiority"):
        Tier1GapRegister.from_mapping(raw)


def _register_to_mapping(register: Tier1GapRegister) -> dict[str, object]:
    return {
        "schema_version": TIER1_GAP_REGISTER_SCHEMA_VERSION,
        "checked_at": register.checked_at,
        "certification_effect": register.certification_effect,
        "purpose": register.purpose,
        "source_boundary": register.source_boundary,
        "superiority_claim_rule": register.superiority_claim_rule,
        "blocked_claims": list(register.blocked_claims),
        "gaps": [
            {
                "id": gap.id,
                "status": gap.status,
                "review_source": gap.review_source,
                "summary": gap.summary,
                "tier_one_references": list(gap.tier_one_references),
                "missing_capabilities": list(gap.missing_capabilities),
                "implemented_capabilities": list(gap.implemented_capabilities),
                "required_evidence_artifacts": list(gap.required_evidence_artifacts),
                "claim_limit": gap.claim_limit,
                "certification_effect": gap.certification_effect,
            }
            for gap in copy.deepcopy(register.gaps)
        ],
    }
