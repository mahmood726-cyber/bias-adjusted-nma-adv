import copy
import json
from pathlib import Path

import pytest

from bias_nma_adv.r_reference_validation import (
    RReferenceValidationError,
    load_r_reference_output,
    validate_component_netmeta_cnma_output,
    validate_ctgov_binary_network_netmeta_output,
    validate_ctgov_hr_network_netmeta_output,
    validate_dose_response_metafor_polynomial_output,
    validate_dta_mada_reitsma_output,
    validate_dta_mada_source_table_output,
    validate_multinma_sglt2_binary_nma_output,
    validate_multiarm_netmeta_output,
    validate_pairwise_metafor_meta_output,
    validate_survival_hr_metafor_pairwise_output,
)


ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_OUTPUT = ROOT / "validation" / "reference_runs" / "pairwise_metafor_meta_output.json"
MULTINMA_OUTPUT = ROOT / "validation" / "reference_runs" / "multinma_sglt2_binary_nma_output.json"
MULTIARM_OUTPUT = ROOT / "validation" / "reference_runs" / "multiarm_netmeta_output.json"
DTA_OUTPUT = ROOT / "validation" / "reference_runs" / "dta_mada_reitsma_output.json"
DTA_SOURCE_OUTPUT = ROOT / "validation" / "reference_runs" / "dta_mada_reitsma_midkine_source_output.json"
DOSE_RESPONSE_OUTPUT = ROOT / "validation" / "reference_runs" / "dose_response_metafor_polynomial_output.json"
SGLT2_SURVIVAL_OUTPUT = ROOT / "validation" / "reference_runs" / "sglt2_survival_hr_metafor_output.json"
PCSK9_SURVIVAL_OUTPUT = ROOT / "validation" / "reference_runs" / "pcsk9_survival_hr_metafor_output.json"
SGLT2_CKD_SURVIVAL_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "sglt2_ckd_survival_hr_metafor_output.json"
)
CTGOV_HR_NETWORK_OUTPUT = ROOT / "validation" / "reference_runs" / "t2d_ctgov_hr_network_netmeta_output.json"
COMPONENT_CNMA_OUTPUT = ROOT / "validation" / "reference_runs" / "component_netmeta_cnma_output.json"
CTGOV_BINARY_NETWORK_OUTPUT = (
    ROOT / "validation" / "reference_runs" / "psoriasis_pasi90_ctgov_binary_network_netmeta_output.json"
)


def test_pairwise_metafor_output_matches_source_backed_python_artifact():
    summary = validate_pairwise_metafor_meta_output(PAIRWISE_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "pairwise_metafor_meta"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["max_abs_difference"] < 1e-12
    assert "hksj_floor_difference_documented" in summary["validated_components"]


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


def test_pairwise_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(PAIRWISE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["metafor"]["fixed_effect"]["estimate"] += 0.01
    mutated = tmp_path / "pairwise_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="fixed_effect estimate"):
        validate_pairwise_metafor_meta_output(mutated, repo_root=ROOT)


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


def test_ctgov_binary_network_netmeta_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(CTGOV_BINARY_NETWORK_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["fixtures"][0]["common"]["etanercept"]["estimate"] += 0.01
    mutated = tmp_path / "ctgov_binary_network_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="etanercept CT.gov binary netmeta common estimate"):
        validate_ctgov_binary_network_netmeta_output(mutated, repo_root=ROOT)


def test_component_netmeta_cnma_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(COMPONENT_CNMA_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["component_effects"]["A"]["estimate"] += 0.01
    mutated = tmp_path / "component_cnma_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="component A estimate"):
        validate_component_netmeta_cnma_output(mutated, repo_root=ROOT)
