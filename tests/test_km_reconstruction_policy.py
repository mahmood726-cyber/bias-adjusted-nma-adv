import copy
from pathlib import Path

import pytest

from bias_nma_adv.data import ValidationError
from bias_nma_adv.km_reconstruction import (
    KMPaperSource,
    KMReconstructionPolicy,
    load_km_reconstruction_policy,
    screen_km_reconstruction_result,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "validation" / "survival" / "km_reconstruction_policy.toml"


SOURCE = {
    "study_id": "OA-KM-TEST",
    "trial": "Open access KM screening fixture",
    "nct_id": "NCT12345678",
    "pmid": "12345678",
    "source_type": "open_access_paper",
    "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/",
    "access_statement": "Open access paper used only as a unit-test source locator fixture.",
    "figure_label": "Figure 2",
    "figure_page": "5",
    "outcome_label": "overall survival",
    "active_treatment": "active",
    "control_treatment": "control",
    "risk_table_status": "present_verified",
    "reuse_origin": "wasserstein_method_pattern_only",
}


RESULT = {
    "hr": 0.72,
    "hr_method": "log_rank_OminusE",
    "orientation_method": "legend_verified",
    "total_ipd_records": 200,
    "curve1_times": [float(i) for i in range(10)],
    "curve1_survivals": [1.0, 0.98, 0.96, 0.94, 0.91, 0.88, 0.84, 0.80, 0.77, 0.74],
    "curve2_times": [float(i) for i in range(10)],
    "curve2_survivals": [1.0, 0.97, 0.93, 0.90, 0.86, 0.81, 0.76, 0.70, 0.65, 0.60],
    "n_curves_found": 2,
    "verification_level": "figure_verified",
    "verification_hash": "a" * 64,
    "warnings": [],
}


def test_km_reconstruction_policy_loads_and_screens_valid_curve_result():
    policy = load_km_reconstruction_policy(POLICY_PATH)
    certificate = screen_km_reconstruction_result(SOURCE, RESULT, policy)

    assert certificate["schema_version"] == "km_reconstruction_screen/v1"
    assert certificate["status"] == "eligible_local_reconstruction"
    assert certificate["certification_effect"] == "none"
    assert certificate["source"]["reuse_origin"] == "wasserstein_method_pattern_only"
    assert certificate["extraction"]["curve1_points"] == 10
    assert certificate["extraction"]["curve2_points"] == 10


def test_km_reconstruction_blocks_wasserstein_text_synthetic_ipd_fallback():
    policy = load_km_reconstruction_policy(POLICY_PATH)
    result = copy.deepcopy(RESULT)
    result["hr_method"] = "text_synthetic_ipd"
    result["orientation_method"] = "text_synthetic_ipd"
    result["curve1_times"] = None
    result["curve1_survivals"] = None
    result["warnings"] = ["Synthetic IPD generated from text-reported HR (no reliable curve pair)."]

    with pytest.raises(ValidationError, match="blocked KM HR method"):
        screen_km_reconstruction_result(SOURCE, result, policy)


def test_km_reconstruction_rejects_text_only_or_unverified_curve_results():
    policy = load_km_reconstruction_policy(POLICY_PATH)
    result = copy.deepcopy(RESULT)
    result["curve2_survivals"] = result["curve2_survivals"][:5]

    with pytest.raises(ValidationError, match="different lengths"):
        screen_km_reconstruction_result(SOURCE, result, policy)

    result = copy.deepcopy(RESULT)
    result["verification_level"] = "unchecked"
    with pytest.raises(ValidationError, match="no verification level"):
        screen_km_reconstruction_result(SOURCE, result, policy)


def test_km_reconstruction_policy_rejects_scope_creep():
    policy = load_km_reconstruction_policy(POLICY_PATH)
    raw_policy = copy.deepcopy(policy.__dict__)
    raw_policy["schema_version"] = "km_reconstruction_policy/v1"
    raw_policy["synthetic_ipd_policy"] = "allowed"
    with pytest.raises(ValidationError, match="Synthetic IPD must be blocked"):
        KMReconstructionPolicy.from_mapping(raw_policy)

    source = copy.deepcopy(SOURCE)
    source["source_type"] = "pubmed_abstract"
    with pytest.raises(ValidationError, match="open_access_paper"):
        KMPaperSource.from_mapping(source)
