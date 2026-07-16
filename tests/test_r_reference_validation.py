import copy
import json
from pathlib import Path

import pytest

from bias_nma_adv.r_reference_validation import (
    RReferenceValidationError,
    load_r_reference_output,
    validate_dta_mada_reitsma_output,
    validate_multiarm_netmeta_output,
    validate_pairwise_metafor_meta_output,
)


ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_OUTPUT = ROOT / "validation" / "reference_runs" / "pairwise_metafor_meta_output.json"
MULTIARM_OUTPUT = ROOT / "validation" / "reference_runs" / "multiarm_netmeta_output.json"
DTA_OUTPUT = ROOT / "validation" / "reference_runs" / "dta_mada_reitsma_output.json"


def test_pairwise_metafor_output_matches_source_backed_python_artifact():
    summary = validate_pairwise_metafor_meta_output(PAIRWISE_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "r_reference_validation/v1"
    assert summary["target_id"] == "pairwise_metafor_meta"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["max_abs_difference"] < 1e-12
    assert "hksj_floor_difference_documented" in summary["validated_components"]


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


def test_pairwise_reference_validation_rejects_numeric_drift(tmp_path):
    payload = load_r_reference_output(PAIRWISE_OUTPUT)
    payload = copy.deepcopy(payload)
    payload["metafor"]["fixed_effect"]["estimate"] += 0.01
    mutated = tmp_path / "pairwise_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RReferenceValidationError, match="fixed_effect estimate"):
        validate_pairwise_metafor_meta_output(mutated, repo_root=ROOT)


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
