from pathlib import Path

import pytest

from bias_nma_adv.dose_response_coverage import (
    DOSE_RESPONSE_COVERAGE_SCHEMA_VERSION,
    DoseResponseCoverageError,
    DoseResponseSourceCoverage,
    load_dose_response_source_coverage,
    summarize_dose_response_source_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "validation" / "dose_response_source_coverage.toml"


def test_dose_response_source_coverage_records_current_allowed_source_data():
    coverage = load_dose_response_source_coverage(COVERAGE)

    assert coverage.status == "active_source_backed_dose_response_data"
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
    assert coverage.registered_benchmark_ids == ("semaglutide_obesity_dose_response",)
    assert coverage.registered_source_counts == {
        "aact_clinicaltrials_gov": 0,
        "clinicaltrials_gov": 1,
        "ema_epar": 0,
        "fda_review": 0,
        "open_access_paper": 0,
        "pactr_results": 0,
        "pubmed_abstract": 1,
        "who_ictrp_results": 0,
    }
    assert any(
        "semaglutide_obesity_dose_response" in item for item in coverage.source_search_summary
    )
    assert (
        "additional_multi_trial_MBNMAdose_reference_runs_before_certification"
        in coverage.required_next_artifacts
    )
    assert (
        "shared_control_covariance_checks_before_certification"
        in coverage.required_next_artifacts
    )


def test_dose_response_source_coverage_summary_is_validation_status_ready():
    summary = summarize_dose_response_source_coverage(
        load_dose_response_source_coverage(COVERAGE)
    )

    assert summary == {
        "schema_version": DOSE_RESPONSE_COVERAGE_SCHEMA_VERSION,
        "checked_at": "2026-07-16",
        "status": "active_source_backed_dose_response_data",
        "registered_benchmark_ids": ["semaglutide_obesity_dose_response"],
        "registered_source_counts": {
            "aact_clinicaltrials_gov": 0,
            "clinicaltrials_gov": 1,
            "ema_epar": 0,
            "fda_review": 0,
            "open_access_paper": 0,
            "pactr_results": 0,
            "pubmed_abstract": 1,
            "who_ictrp_results": 0,
        },
        "has_source_backed_dose_response_data": True,
        "required_next_artifacts": [
            "dose_response_source_manifest_with_nct_pmid_and_open_access_identifiers",
            "dose_amount_unit_route_frequency_duration_or_equivalence_fields",
            "arm_level_or_contrast_effects_bound_to_source_snippets_or_tables",
            "source_identity_checks_for_clinicaltrials_gov_and_pubmed",
            "open_access_table_or_text_token_checks_for_dose_values_and_effects",
            "aact_or_registry_result_row_checks_when_used",
            "fda_or_ema_regulatory_review_row_checks_when_used",
            "dose_response_analysis_artifact",
            "additional_multi_trial_MBNMAdose_reference_runs_before_certification",
            "shared_control_covariance_checks_before_certification",
        ],
        "certification_effect": "none",
    }


def test_dose_response_source_coverage_rejects_softened_missing_status():
    raw = _coverage_to_mapping(load_dose_response_source_coverage(COVERAGE))
    raw["status"] = "missing_source_backed_dose_response_data"
    raw["registered_benchmark_ids"] = ["fabricated_dose_response"]

    with pytest.raises(DoseResponseCoverageError, match="cannot list"):
        DoseResponseSourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_dose_response_source_coverage(COVERAGE))
    raw["certification_effect"] = "production_certified"

    with pytest.raises(DoseResponseCoverageError, match="cannot certify"):
        DoseResponseSourceCoverage.from_mapping(raw)


def _coverage_to_mapping(coverage: DoseResponseSourceCoverage) -> dict[str, object]:
    return {
        "schema_version": DOSE_RESPONSE_COVERAGE_SCHEMA_VERSION,
        "checked_at": coverage.checked_at,
        "status": coverage.status,
        "certification_effect": coverage.certification_effect,
        "purpose": coverage.purpose,
        "allowed_evidence_sources": list(coverage.allowed_evidence_sources),
        "protocol_only_sources": list(coverage.protocol_only_sources),
        "protocol_registry_rule": coverage.protocol_registry_rule,
        "registered_benchmark_ids": list(coverage.registered_benchmark_ids),
        "registered_source_counts": dict(coverage.registered_source_counts),
        "source_search_summary": list(coverage.source_search_summary),
        "required_next_artifacts": list(coverage.required_next_artifacts),
        "claim_limit": coverage.claim_limit,
    }
