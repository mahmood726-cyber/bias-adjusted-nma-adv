from pathlib import Path

import pytest

from bias_nma_adv.dta_coverage import (
    DTA_SOURCE_COVERAGE_SCHEMA_VERSION,
    DTACoverageError,
    DTASourceCoverage,
    load_dta_source_coverage,
    summarize_dta_source_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "validation" / "dta_source_coverage.toml"


def test_dta_source_coverage_records_current_source_backed_data_without_overclaiming():
    coverage = load_dta_source_coverage(COVERAGE)

    assert coverage.status == "active_source_backed_dta_data"
    assert coverage.model_status == "source_backed_not_reference_matched"
    assert coverage.certification_effect == "none"
    assert coverage.allowed_evidence_sources == (
        "aact_clinicaltrials_gov",
        "clinicaltrials_gov",
        "ema_epar",
        "fda_review",
        "open_access_paper",
        "pactr_results",
        "pubmed_abstract",
        "who_ictrp_results",
    )
    assert coverage.protocol_only_sources == (
        "other_trial_registry_protocol",
        "pactr_protocol",
        "who_ictrp_protocol",
    )
    assert coverage.registered_benchmark_ids == ("midkine_elisa_cancer_dta",)
    assert coverage.registered_source_counts == {
        "aact_clinicaltrials_gov": 0,
        "clinicaltrials_gov": 0,
        "ema_epar": 0,
        "fda_review": 0,
        "open_access_paper": 11,
        "pactr_results": 0,
        "pubmed_abstract": 0,
        "who_ictrp_results": 0,
    }
    assert {"tp", "fp", "fn", "tn"} <= set(coverage.required_source_fields)
    assert {"bivariate_random_effects_glmm", "hsroc"} <= set(
        coverage.required_model_families
    )


def test_dta_source_coverage_summary_is_validation_status_ready():
    summary = summarize_dta_source_coverage(load_dta_source_coverage(COVERAGE))

    assert summary == {
        "schema_version": DTA_SOURCE_COVERAGE_SCHEMA_VERSION,
        "checked_at": "2026-07-16",
        "status": "active_source_backed_dta_data",
        "model_status": "source_backed_not_reference_matched",
        "registered_benchmark_ids": ["midkine_elisa_cancer_dta"],
        "registered_source_counts": {
            "aact_clinicaltrials_gov": 0,
            "clinicaltrials_gov": 0,
            "ema_epar": 0,
            "fda_review": 0,
            "open_access_paper": 11,
            "pactr_results": 0,
            "pubmed_abstract": 0,
            "who_ictrp_results": 0,
        },
        "has_source_backed_dta_data": True,
        "required_model_families": [
            "bivariate_random_effects_glmm",
            "hsroc",
        ],
        "required_next_artifacts": [
            "dta_source_manifest_with_nct_pmid_or_open_access_identifiers",
            "source_verified_tp_fp_fn_tn_2x2_tables",
            "aact_ictrp_or_pactr_result_row_checks_when_used",
            "fda_or_ema_regulatory_review_row_checks_when_used",
            "threshold_and_reference_standard_metadata",
            "bivariate_glmm_or_hsroc_model_artifact",
            "mada_or_metafor_or_reference_software_parity_before_certification",
        ],
        "certification_effect": "none",
    }


def test_dta_source_coverage_rejects_fake_data_and_model_shortcuts():
    raw = _coverage_to_mapping(load_dta_source_coverage(COVERAGE))
    raw["status"] = "missing_source_backed_dta_data"
    raw["registered_benchmark_ids"] = ["fabricated_dta"]
    with pytest.raises(DTACoverageError, match="cannot list"):
        DTASourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_dta_source_coverage(COVERAGE))
    raw["required_model_families"] = ["univariate_sensitivity"]
    with pytest.raises(DTACoverageError, match="bivariate GLMM and HSROC"):
        DTASourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_dta_source_coverage(COVERAGE))
    raw["certification_effect"] = "production_certified"
    with pytest.raises(DTACoverageError, match="cannot certify"):
        DTASourceCoverage.from_mapping(raw)


def _coverage_to_mapping(coverage: DTASourceCoverage) -> dict[str, object]:
    return {
        "schema_version": DTA_SOURCE_COVERAGE_SCHEMA_VERSION,
        "checked_at": coverage.checked_at,
        "status": coverage.status,
        "model_status": coverage.model_status,
        "certification_effect": coverage.certification_effect,
        "purpose": coverage.purpose,
        "allowed_evidence_sources": list(coverage.allowed_evidence_sources),
        "protocol_only_sources": list(coverage.protocol_only_sources),
        "protocol_registry_rule": coverage.protocol_registry_rule,
        "registered_benchmark_ids": list(coverage.registered_benchmark_ids),
        "registered_source_counts": dict(coverage.registered_source_counts),
        "required_source_fields": list(coverage.required_source_fields),
        "required_model_families": list(coverage.required_model_families),
        "source_search_summary": list(coverage.source_search_summary),
        "required_next_artifacts": list(coverage.required_next_artifacts),
        "claim_limit": coverage.claim_limit,
    }
