import json
from pathlib import Path
import subprocess
import sys

from bias_nma_adv.real_benchmark_atlas import (
    REAL_BENCHMARK_ATLAS_SCHEMA_VERSION,
    build_real_benchmark_atlas,
    summarize_real_benchmark_atlas,
)


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "validation" / "real_benchmark_atlas.json"
SCRIPT = ROOT / "scripts" / "write_real_benchmark_atlas.py"


def test_real_benchmark_atlas_summarizes_current_source_backed_coverage():
    atlas = build_real_benchmark_atlas(ROOT, checked_at="2026-07-15T00:00:00Z")

    assert atlas["schema_version"] == REAL_BENCHMARK_ATLAS_SCHEMA_VERSION
    assert atlas["status"] == "passed"
    assert atlas["certification_effect"] == "none"
    assert atlas["allowed_effect_evidence_sources"] == [
        "aact_clinicaltrials_gov",
        "clinicaltrials_gov",
        "ema_epar",
        "fda_review",
        "open_access_paper",
        "pactr_results",
        "pubmed_abstract",
        "who_ictrp_results",
    ]
    assert atlas["allowed_protocol_only_sources"] == [
        "other_trial_registry_protocol",
        "pactr_protocol",
        "who_ictrp_protocol",
    ]
    assert atlas["n_benchmarks"] == 24
    assert atlas["n_benchmark_study_effects"] == 144
    assert atlas["n_tau2_positive_benchmarks"] == 12
    assert atlas["n_unique_study_ids"] == 123
    assert atlas["n_unique_nct_ids"] == 88
    assert atlas["n_unique_pmids"] == 83
    assert atlas["domain_counts"] == {
        "arm_count_binary_closed_loop_network": 1,
        "binary_pairwise_meta": 1,
        "component_nma": 1,
        "cross_design_nma": 1,
        "diagnostic_test_accuracy": 1,
        "dose_response_pairwise": 1,
        "reported_hr_star_network": 1,
        "reported_survival_hr_pairwise": 17,
    }
    assert atlas["evidence_mode_counts"] == {
        "ctgov_arm_level_binary_counts": 1,
        "ctgov_component_lsmean": 1,
        "ctgov_dose_response_lsmean": 1,
        "open_access_jats_table_2x2": 1,
        "pubmed_abstract_event_counts": 1,
        "reported_hr_clinicaltrials_gov_results": 1,
        "reported_hr_pubmed_abstract": 17,
        "reported_hr_pubmed_abstract_cross_design": 1,
    }
    assert atlas["source_check_scope_counts"] == {
        "clinicaltrials_gov_arm_level_binary_counts": 2,
        "clinicaltrials_gov_component_lsmean": 1,
        "clinicaltrials_gov_dose_response_lsmean": 1,
        "clinicaltrials_gov_reported_hr_analysis": 10,
        "identity_and_reachability": 162,
        "open_access_jats_table_2x2": 11,
        "pubmed_abstract_binary_network_identity": 2,
        "pubmed_abstract_component_identity": 1,
        "pubmed_abstract_cross_design_reported_hr_tokens": 4,
        "pubmed_abstract_dose_response_identity": 1,
        "pubmed_abstract_event_count_tokens": 4,
        "pubmed_abstract_reported_hr_tokens": 77,
    }
    assert atlas["source_type_counts"] == {
        "clinicaltrials_gov": 95,
        "open_access_paper": 11,
        "pubmed_abstract": 170,
    }
    assert "tier-one parity" in atlas["does_not_prove"]
    assert (
        "additional source-backed closed-loop networks and external inconsistency references before performance claims"
        in atlas["required_next_gates"]
    )
    assert (
        "broader source-backed positive-tau benchmarks across domains before broad heterogeneity-stress claims"
        in atlas["required_next_gates"]
    )
    assert {item["id"] for item in atlas["benchmarks"]} == {
        "sglt2_hf_primary_log_or",
        "sglt2_hf_reported_hr",
        "pcsk9_mace_reported_hr",
        "sglt2_ckd_reported_hr",
        "glp1_mace_reported_hr",
        "parp_firstline_ovarian_pfs_reported_hr",
        "parp_recurrent_ovarian_pfs_reported_hr",
        "cdk46_breast_pfs_reported_hr",
        "rcc_firstline_pfs_reported_hr",
        "nsclc_firstline_pfs_reported_hr",
        "hfref_therapies_primary_reported_hr",
        "doac_af_primary_reported_hr",
        "tavi_savr_primary_reported_hr",
        "melanoma_pfs_reported_hr",
        "osimertinib_nsclc_pfs_reported_hr",
        "lipid_cv_outcomes_reported_hr",
        "her2_breast_pfs_reported_hr",
        "prostate_mhspc_os_reported_hr",
        "t2d_mace_ctgov_hr_network",
        "psoriasis_pasi90_ctgov_binary_network",
        "semaglutide_obesity_dose_response",
        "sitagliptin_pioglitazone_component",
        "sglt2_rct_nrs_cross_design",
        "midkine_elisa_cancer_dta",
    }
    assert {item["certification_effect"] for item in atlas["benchmarks"]} == {"none"}
    assert {item["has_positive_tau2"] for item in atlas["benchmarks"]} == {False, True}
    assert {
        item["id"] for item in atlas["benchmarks"] if item["has_positive_tau2"]
    } == {
        "sglt2_ckd_reported_hr",
        "glp1_mace_reported_hr",
        "parp_firstline_ovarian_pfs_reported_hr",
        "rcc_firstline_pfs_reported_hr",
        "nsclc_firstline_pfs_reported_hr",
        "hfref_therapies_primary_reported_hr",
        "doac_af_primary_reported_hr",
        "tavi_savr_primary_reported_hr",
        "melanoma_pfs_reported_hr",
        "osimertinib_nsclc_pfs_reported_hr",
        "lipid_cv_outcomes_reported_hr",
        "her2_breast_pfs_reported_hr",
    }


def test_real_benchmark_atlas_artifact_regenerates(tmp_path):
    output = tmp_path / "real_benchmark_atlas.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(ROOT),
            "--checked-at",
            "2026-07-17T22:40:00Z",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "real benchmark atlas written" in completed.stdout
    expected = json.loads(ATLAS.read_text(encoding="utf-8"))
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == expected


def test_real_benchmark_atlas_summary_is_validation_status_ready():
    atlas = json.loads(ATLAS.read_text(encoding="utf-8"))

    assert summarize_real_benchmark_atlas(atlas) == {
        "schema_version": REAL_BENCHMARK_ATLAS_SCHEMA_VERSION,
        "status": "passed",
        "n_benchmarks": 24,
        "n_benchmark_study_effects": 144,
        "n_tau2_positive_benchmarks": 12,
        "n_unique_nct_ids": 88,
        "n_unique_pmids": 83,
        "domain_counts": {
            "arm_count_binary_closed_loop_network": 1,
            "binary_pairwise_meta": 1,
            "component_nma": 1,
            "cross_design_nma": 1,
            "diagnostic_test_accuracy": 1,
            "dose_response_pairwise": 1,
            "reported_hr_star_network": 1,
            "reported_survival_hr_pairwise": 17,
        },
        "source_type_counts": {
            "clinicaltrials_gov": 95,
            "open_access_paper": 11,
            "pubmed_abstract": 170,
        },
        "certification_effect": "none",
    }
