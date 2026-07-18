import copy
import json
from pathlib import Path

import pytest

from bias_nma_adv.r_reference_validation import (
    RReferenceValidationError,
    load_r_reference_output,
    validate_component_netmeta_cnma_output,
    validate_crossnma_sglt2_compatibility_output,
    validate_ctgov_binary_network_netmeta_output,
    validate_ctgov_binary_network_netsplit_output,
    validate_ctgov_hr_network_netmeta_output,
    validate_dose_response_metafor_polynomial_output,
    validate_dta_mada_reitsma_output,
    validate_dta_mada_source_table_output,
    validate_mbnmadose_semaglutide_polynomial_output,
    validate_metafor_tau2_crosscheck_output,
    validate_multinma_sglt2_binary_nma_output,
    validate_multiarm_netmeta_output,
    validate_pairwise_metafor_gosh_output,
    validate_pairwise_metafor_meta_output,
    validate_pairwise_metafor_sparse_binary_output,
    validate_publication_bias_metafor_regtest_output,
    validate_publication_bias_metafor_trimfill_output,
    validate_survival_hr_metafor_pairwise_output,
)


ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_OUTPUT = ROOT / "validation" / "reference_runs" / "pairwise_metafor_meta_output.json"
GOSH_OUTPUT = ROOT / "validation" / "reference_runs" / "sglt2_hf_metafor_gosh_output.json"
SPARSE_BINARY_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "psoriasis_sparse_binary_metafor_output.json"
)
MULTINMA_OUTPUT = ROOT / "validation" / "reference_runs" / "multinma_sglt2_binary_nma_output.json"
MULTIARM_OUTPUT = ROOT / "validation" / "reference_runs" / "multiarm_netmeta_output.json"
DTA_OUTPUT = ROOT / "validation" / "reference_runs" / "dta_mada_reitsma_output.json"
DTA_SOURCE_OUTPUT = ROOT / "validation" / "reference_runs" / "dta_mada_reitsma_midkine_source_output.json"
DOSE_RESPONSE_OUTPUT = ROOT / "validation" / "reference_runs" / "dose_response_metafor_polynomial_output.json"
MBNMADOSE_OUTPUT = ROOT / "validation" / "reference_runs" / "mbnmadose_semaglutide_polynomial_output.json"
SGLT2_SURVIVAL_OUTPUT = ROOT / "validation" / "reference_runs" / "sglt2_survival_hr_metafor_output.json"
PCSK9_SURVIVAL_OUTPUT = ROOT / "validation" / "reference_runs" / "pcsk9_survival_hr_metafor_output.json"
SGLT2_CKD_SURVIVAL_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "sglt2_ckd_survival_hr_metafor_output.json"
)
CTGOV_HR_NETWORK_OUTPUT = ROOT / "validation" / "reference_runs" / "t2d_ctgov_hr_network_netmeta_output.json"
PUBLICATION_BIAS_REGTEST_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "publication_bias_t2d_ctgov_regtest_output.json"
)
PUBLICATION_BIAS_TRIMFILL_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "publication_bias_glp1_metafor_trimfill_output.json"
)
TAU2_CROSSCHECK_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "metafor_tau2_crosscheck_survival_output.json"
)
COMPONENT_CNMA_OUTPUT = ROOT / "validation" / "reference_runs" / "component_netmeta_cnma_output.json"
CROSSNMA_COMPAT_OUTPUT = ROOT / "validation" / "reference_runs" / "crossnma_sglt2_compatibility_output.json"
CTGOV_BINARY_NETWORK_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "psoriasis_pasi90_ctgov_binary_network_netmeta_output.json"
)
CTGOV_BINARY_NETSPLIT_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "psoriasis_pasi90_ctgov_binary_network_netsplit_output.json"
)


def test_pairwise_metafor_output_matches_source_backed_python_artifact():
    summary = validate_pairwise_metafor_meta_output(PAIRWISE_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "pairwise_metafor_meta"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["max_abs_difference"] < 1e-12
    assert "hksj_floor_difference_documented" in summary["validated_components"]


def test_pairwise_metafor_gosh_output_matches_source_backed_subset_space():
    summary = validate_pairwise_metafor_gosh_output(GOSH_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "pairwise_metafor_gosh_sglt2"
    assert summary["benchmark_id"] == "sglt2_hf_primary_log_or"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "metafor::gosh fixed-effect subset diagnostic"
    assert summary["max_abs_difference"] < 1e-12
    assert "all_nonempty_subset_enumeration" in summary["validated_components"]
    assert "not an outlier-removal rule" in summary["source_policy_note"]


def test_pairwise_sparse_binary_metafor_output_matches_source_backed_counts():
    summary = validate_pairwise_metafor_sparse_binary_output(
        SPARSE_BINARY_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "pairwise_metafor_sparse_binary_psoriasis"
    assert summary["benchmark_id"] == "psoriasis_pasi90_ctgov_binary_network"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == (
        "metafor::rma.uni sparse binary count-derived log-OR"
    )
    assert summary["max_abs_difference"] < 1e-12
    assert "source_backed_low_control_event_count_rows" in summary["validated_components"]
    assert "zero-event parity" in summary["source_policy_note"]


def test_metafor_tau2_crosscheck_matches_source_backed_survival_benchmarks():
    summary = validate_metafor_tau2_crosscheck_output(TAU2_CROSSCHECK_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "pairwise_metafor_tau2_crosscheck_source"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "metafor::rma.uni FE/DL/PM/REML tau2 cross-check"
    assert set(summary["benchmark_ids"]) == {
        "sglt2_hf_reported_hr",
        "glp1_mace_reported_hr",
        "lipid_cv_outcomes_reported_hr",
        "hcc_os_reported_hr",
    }
    assert summary["max_abs_difference"] <= 0.05
    assert "tau2_estimator_cross_check" in summary["validated_components"]
    assert "not broad optimizer stress parity" in summary["source_policy_note"]


def test_multinma_sglt2_binary_nma_output_matches_source_backed_reference():
    summary = validate_multinma_sglt2_binary_nma_output(MULTINMA_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "bayesian_nma_multinma_cmdstan"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "multinma fixed-effect binomial NMA via rstan"
    assert summary["max_abs_difference"] <= 0.03
    assert "multinma_fixed_effect_log_or" in summary["validated_components"]
    assert "not broad Bayesian NMA" in summary["source_policy_note"]


def test_multiarm_netmeta_output_matches_python_multiarm_fixture_artifact():
    summary = validate_multiarm_netmeta_output(MULTIARM_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "multiarm_gls_netmeta_portfolio_fixture"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["max_abs_difference"] < 1e-12
    assert "multiarm_random_effect_estimates" in summary["validated_components"]


def test_dta_mada_output_matches_python_dta_algorithmic_fixture():
    summary = validate_dta_mada_reitsma_output(DTA_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "dta_bivariate_hsroc_reference"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "mada::reitsma"
    assert summary["max_abs_probability_difference"] <= 1e-3
    assert summary["max_abs_log_difference"] <= 2e-3
    assert summary["max_abs_variance_difference"] <= 3e-3
    assert summary["max_abs_rho_difference"] <= 6e-3
    assert "not source-backed clinical evidence" in summary["source_policy_note"]


def test_dta_mada_source_output_matches_source_backed_open_access_table():
    summary = validate_dta_mada_source_table_output(DTA_SOURCE_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "dta_source_table_mada_reitsma_smoke"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "mada::reitsma source-backed DTA table"
    assert summary["benchmark_id"] == "midkine_elisa_cancer_dta"
    assert summary["max_abs_difference"] <= 1e-5
    assert "source_backed_tp_fp_fn_tn_rows" in summary["validated_components"]
    assert "AUC is exported but not treated as local parity" in summary["auc_note"]


def test_dose_response_metafor_output_matches_source_backed_smoke_artifact():
    summary = validate_dose_response_metafor_polynomial_output(
        DOSE_RESPONSE_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "dose_response_metafor_polynomial_smoke"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "metafor fixed-effect polynomial meta-regression"
    assert summary["max_abs_difference"] < 1e-10
    assert "mbnmadose_limitation_preserved" in summary["validated_components"]
    assert "not MBNMAdose parity" in summary["source_policy_note"]


def test_mbnmadose_semaglutide_output_matches_source_backed_arm_level_smoke():
    summary = validate_mbnmadose_semaglutide_polynomial_output(
        MBNMADOSE_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "dose_response_mbnmadose"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == (
        "MBNMAdose common-effect linear polynomial dose-response smoke"
    )
    assert summary["benchmark_id"] == "semaglutide_obesity_dose_response"
    assert summary["max_abs_difference"] <= 0.25
    assert "mbnmadose_common_linear_beta" in summary["validated_components"]
    assert "not broad MBNMAdose parity" in summary["source_policy_note"]


@pytest.mark.parametrize(
    ("output_path", "benchmark_id"),
    [
        (SGLT2_SURVIVAL_OUTPUT, "sglt2_hf_reported_hr"),
        (PCSK9_SURVIVAL_OUTPUT, "pcsk9_mace_reported_hr"),
        (SGLT2_CKD_SURVIVAL_OUTPUT, "sglt2_ckd_reported_hr"),
    ],
)
def test_survival_hr_metafor_output_matches_source_backed_reported_hr_artifact(
    output_path,
    benchmark_id,
):
    summary = validate_survival_hr_metafor_pairwise_output(output_path, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "reported_hr_survival_metafor_pairwise"
    assert summary["benchmark_id"] == benchmark_id
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "metafor fixed-effect reported-HR meta-analysis"
    assert summary["max_abs_difference"] < 1e-10
    assert "survival_nma_limitation_preserved" in summary["validated_components"]
    assert "not KM reconstruction" in summary["source_policy_note"]


def test_ctgov_hr_network_netmeta_output_matches_source_backed_star_network():
    summary = validate_ctgov_hr_network_netmeta_output(CTGOV_HR_NETWORK_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "ctgov_hr_network_netmeta_star"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "netmeta fixed-effect CT.gov reported-HR star network"
    assert summary["max_abs_difference"] < 1e-10
    assert "star_network_limitation_preserved" in summary["validated_components"]
    assert "not closed-loop inconsistency" in summary["source_policy_note"]


def test_publication_bias_metafor_regtest_matches_source_backed_ctgov_hr_rows():
    summary = validate_publication_bias_metafor_regtest_output(
        PUBLICATION_BIAS_REGTEST_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "publication_bias_metafor_regtest_smoke"
    assert summary["benchmark_id"] == "t2d_mace_ctgov_hr_network"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "metafor::regtest small-study-effect diagnostic"
    assert summary["max_abs_difference"] < 1e-12
    assert "metafor_regtest_p_value" in summary["validated_components"]
    assert "not trim-and-fill" in summary["source_policy_note"]


def test_publication_bias_metafor_trimfill_matches_source_backed_glp1_hr_rows():
    summary = validate_publication_bias_metafor_trimfill_output(
        PUBLICATION_BIAS_TRIMFILL_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "publication_bias_metafor_trimfill_glp1"
    assert summary["benchmark_id"] == "glp1_mace_reported_hr"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "metafor::trimfill fixed-effect reported-HR sensitivity"
    assert summary["max_abs_difference"] < 1e-12
    assert "trimfill_adjusted_fixed_effect_summary" in summary["validated_components"]
    assert "not publication-bias proof" in summary["source_policy_note"]


def test_ctgov_binary_network_netmeta_output_matches_source_backed_closed_loop_network():
    summary = validate_ctgov_binary_network_netmeta_output(
        CTGOV_BINARY_NETWORK_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "ctgov_binary_network_netmeta_closed_loop"
    assert summary["benchmark_id"] == "psoriasis_pasi90_ctgov_binary_network"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "netmeta CT.gov arm-count closed-loop binary network"
    assert summary["max_abs_difference"] < 1e-12
    assert "closed_loop_source_backed_network" in summary["validated_components"]
    assert "not broad netmeta parity" in summary["source_policy_note"]


def test_ctgov_binary_network_netsplit_output_matches_source_backed_closed_loop_network():
    summary = validate_ctgov_binary_network_netsplit_output(
        CTGOV_BINARY_NETSPLIT_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "node_splitting_netmeta_netsplit_psoriasis"
    assert summary["benchmark_id"] == "psoriasis_pasi90_ctgov_binary_network"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "netmeta::netsplit back-calculation SIDE"
    assert summary["max_abs_difference"] < 1e-12
    assert "direct_indirect_difference_arithmetic" in summary["validated_components"]
    assert "not broad inconsistency parity" in summary["source_policy_note"]


def test_component_netmeta_cnma_output_matches_local_additive_component_core():
    summary = validate_component_netmeta_cnma_output(COMPONENT_CNMA_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "component_nma_netmeta_cnma"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["reference_method"] == "netmeta::discomb additive CNMA"
    assert summary["max_abs_difference"] < 1e-12
    assert "additive_component_effects" in summary["validated_components"]
    assert "not source-backed CNMA validation" in summary["source_policy_note"]


def test_crossnma_compatibility_preflight_blocks_incompatible_source_fixture():
    summary = validate_crossnma_sglt2_compatibility_output(
        CROSSNMA_COMPAT_OUTPUT,
        repo_root=ROOT,
    )

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "cross_design_crossnma"
    assert summary["status"] == "failed"
    assert summary["certification_effect"] == "none"
    assert summary["benchmark_id"] == "sglt2_rct_nrs_cross_design"
    assert summary["max_abs_difference"] <= 1e-12
    assert "log_hr_effect_scale_blocker_preserved" in summary["validated_components"]
    assert "no crossnma model was run" in summary["skip_reason"]


def test_pairwise_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(PAIRWISE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["metafor"]["fixed_effect"]["estimate"] += 0.01
    mutated = tmp_path / "pairwise_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="fixed_effect estimate"):
        validate_pairwise_metafor_meta_output(mutated, repo_root=ROOT)


def test_pairwise_metafor_gosh_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(GOSH_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["subsets"][-1]["estimate"] += 0.01
    mutated = tmp_path / "gosh_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="GOSH subset estimate"):
        validate_pairwise_metafor_gosh_output(mutated, repo_root=ROOT)


def test_multinma_reference_validation_rejects_diagnostic_drift(tmp_path):
    payload = load_r_reference_output(MULTINMA_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["diagnostics"]["divergent_transitions"] = 1
    mutated = tmp_path / "multinma_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="divergent transitions"):
        validate_multinma_sglt2_binary_nma_output(mutated, repo_root=ROOT)


def test_multiarm_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(MULTIARM_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["fixtures"][1]["random"]["B"]["se"] += 0.01
    mutated = tmp_path / "multiarm_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="heterogeneous random B se"):
        validate_multiarm_netmeta_output(mutated, repo_root=ROOT)


def test_dta_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(DTA_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["summary"]["pooled_sensitivity"] += 0.01
    mutated = tmp_path / "dta_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="pooled_sensitivity"):
        validate_dta_mada_reitsma_output(mutated, repo_root=ROOT)


def test_dta_source_reference_validation_rejects_count_drift(tmp_path):
    payload = load_r_reference_output(DTA_SOURCE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["study_effects"][0]["tp"] += 1
    mutated = tmp_path / "dta_source_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="tp mismatch"):
        validate_dta_mada_source_table_output(mutated, repo_root=ROOT)


def test_dose_response_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(DOSE_RESPONSE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["metafor"]["weighted_quadratic"]["coefficients"][1] += 0.01
    mutated = tmp_path / "dose_response_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="weighted_quadratic coefficient 1"):
        validate_dose_response_metafor_polynomial_output(mutated, repo_root=ROOT)


def test_mbnmadose_reference_validation_rejects_diagnostic_drift(tmp_path):
    payload = load_r_reference_output(MBNMADOSE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["mbnma"]["beta_1"]["rhat"] = 1.02
    mutated = tmp_path / "mbnmadose_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="R-hat exceeds"):
        validate_mbnmadose_semaglutide_polynomial_output(mutated, repo_root=ROOT)


def test_mbnmadose_reference_validation_rejects_source_arm_drift(tmp_path):
    payload = load_r_reference_output(MBNMADOSE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["study_arms"][0]["lsmean"] += 0.01
    mutated = tmp_path / "mbnmadose_arm_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="lsmean"):
        validate_mbnmadose_semaglutide_polynomial_output(mutated, repo_root=ROOT)


def test_survival_hr_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(SGLT2_SURVIVAL_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["metafor"]["fixed_effect"]["estimate"] += 0.01
    mutated = tmp_path / "survival_hr_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="fixed_effect estimate"):
        validate_survival_hr_metafor_pairwise_output(mutated, repo_root=ROOT)


def test_ctgov_hr_network_netmeta_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(CTGOV_HR_NETWORK_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["netmeta"]["common"]["GLP-1 RA"]["estimate"] += 0.01
    mutated = tmp_path / "ctgov_hr_network_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="GLP-1 RA netmeta common estimate"):
        validate_ctgov_hr_network_netmeta_output(mutated, repo_root=ROOT)


def test_publication_bias_regtest_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(PUBLICATION_BIAS_REGTEST_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["metafor"]["egger_lm_sei"]["slope"] += 0.01
    mutated = tmp_path / "publication_bias_regtest_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="regtest slope"):
        validate_publication_bias_metafor_regtest_output(mutated, repo_root=ROOT)


def test_publication_bias_trimfill_validation_rejects_imputed_row_drift(tmp_path):
    payload = load_r_reference_output(PUBLICATION_BIAS_TRIMFILL_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["filled_rows"][-1]["yi"] += 0.01
    mutated = tmp_path / "publication_bias_trimfill_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="adjusted estimate"):
        validate_publication_bias_metafor_trimfill_output(mutated, repo_root=ROOT)


def test_ctgov_binary_network_netmeta_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(CTGOV_BINARY_NETWORK_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["fixtures"][0]["common"]["etanercept"]["estimate"] += 0.01
    mutated = tmp_path / "ctgov_binary_network_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="etanercept CT.gov binary netmeta common estimate"):
        validate_ctgov_binary_network_netmeta_output(mutated, repo_root=ROOT)


def test_ctgov_binary_network_netsplit_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(CTGOV_BINARY_NETSPLIT_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["splits"][0]["difference"] += 0.01
    mutated = tmp_path / "ctgov_binary_netsplit_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="difference identity"):
        validate_ctgov_binary_network_netsplit_output(mutated, repo_root=ROOT)


def test_component_netmeta_cnma_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(COMPONENT_CNMA_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["component_effects"]["A"]["estimate"] += 0.01
    mutated = tmp_path / "component_cnma_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="component A estimate"):
        validate_component_netmeta_cnma_output(mutated, repo_root=ROOT)


def test_crossnma_compatibility_validation_rejects_softened_blocker(tmp_path):
    payload = load_r_reference_output(CROSSNMA_COMPAT_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["crossnma_api"]["crossnma_model_attempted"] = True
    mutated = tmp_path / "crossnma_compat_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="must not run a model"):
        validate_crossnma_sglt2_compatibility_output(mutated, repo_root=ROOT)
