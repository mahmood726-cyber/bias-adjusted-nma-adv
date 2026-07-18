from pathlib import Path
import tomllib

import pytest

from bias_nma_adv.mlnmr_coverage import (
    MLNMR_SOURCE_COVERAGE_SCHEMA_VERSION,
    MLNMRCoverageError,
    MLNMRSourceCoverage,
    load_mlnmr_source_coverage,
    summarize_mlnmr_source_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "validation" / "mlnmr_source_coverage.toml"
SOURCE_SEARCH = ROOT / "validation" / "mlnmr_source_search_2026_07_17.toml"
SOURCE_SEARCH_FOLLOWUP = ROOT / "validation" / "mlnmr_source_search_2026_07_18.toml"
SOURCE_SEARCH_DIABETES = (
    ROOT / "validation" / "mlnmr_source_search_2026_07_18_diabetes.toml"
)


def test_mlnmr_source_coverage_records_current_real_data_blocker():
    coverage = load_mlnmr_source_coverage(COVERAGE)

    assert coverage.status == "missing_source_backed_mlnmr_data"
    assert coverage.model_status == "simulated_reference_only"
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
    assert coverage.registered_benchmark_ids == ()
    assert coverage.registered_source_counts == {
        "aact_clinicaltrials_gov": 0,
        "clinicaltrials_gov": 0,
        "ema_epar": 0,
        "fda_review": 0,
        "open_access_paper": 0,
        "pactr_results": 0,
        "pubmed_abstract": 0,
        "who_ictrp_results": 0,
    }
    assert {
        "public_trial_ipd_rows",
        "source_bound_aggregate_covariate_distribution",
        "covariate_overlap_diagnostics",
        "multinma_mlnmr_reference_run_before_certification",
    } <= set(coverage.required_source_components)
    exclusions = " ".join(coverage.excluded_source_patterns).lower()
    assert "simulated" in exclusions
    assert "synthetic" in exclusions
    assert "pseudo-ipd" in exclusions
    assert "survey" in exclusions
    assert coverage.formal_source_boundary_decision == (
        "real_mlnmr_domain_formally_out_of_scope_under_current_public_source_boundary"
    )
    assert coverage.large_scale_domain_exclusions() == {
        "mlnmr": coverage.formal_source_boundary_reason
    }
    assert "does not certify" in coverage.formal_source_boundary_reason
    assert "feature parity" in coverage.formal_source_boundary_reason
    assert "validation/mlnmr_source_search_2026_07_18_diabetes.toml" in (
        coverage.formal_decision_artifacts
    )


def test_mlnmr_source_coverage_summary_is_validation_status_ready():
    summary = summarize_mlnmr_source_coverage(load_mlnmr_source_coverage(COVERAGE))

    assert summary == {
        "schema_version": MLNMR_SOURCE_COVERAGE_SCHEMA_VERSION,
        "checked_at": "2026-07-18",
        "status": "missing_source_backed_mlnmr_data",
        "model_status": "simulated_reference_only",
        "registered_benchmark_ids": [],
        "registered_source_counts": {
            "aact_clinicaltrials_gov": 0,
            "clinicaltrials_gov": 0,
            "ema_epar": 0,
            "fda_review": 0,
            "open_access_paper": 0,
            "pactr_results": 0,
            "pubmed_abstract": 0,
            "who_ictrp_results": 0,
        },
        "has_source_backed_mlnmr_data": False,
        "formal_source_boundary_decision": "real_mlnmr_domain_formally_out_of_scope_under_current_public_source_boundary",
        "formal_source_boundary_reason": "Bounded searches found no public row-level randomized-trial IPD plus connected aggregate covariate network. Controlled-access Vivli or sponsor IPD, simulated package IPD, pseudo-IPD, and single-trial leads remain excluded; this does not certify ML-NMR, does not count as feature parity, and does not enable clinical population-adjustment claims.",
        "formal_decision_artifacts": [
            "validation/mlnmr_source_search_2026_07_17.toml",
            "validation/mlnmr_source_search_2026_07_18.toml",
            "validation/mlnmr_source_search_2026_07_18_diabetes.toml",
        ],
        "has_formal_source_boundary_exclusion": True,
        "required_source_components": [
            "public_trial_ipd_rows",
            "source_bound_aggregate_covariate_distribution",
            "shared_estimand_treatments_and_outcome",
            "source_verified_trial_identifiers_nct_pmid_or_doi",
            "covariate_overlap_diagnostics",
            "multinma_mlnmr_reference_run_before_certification",
        ],
        "excluded_source_patterns": [
            "simulated IPD from package vignettes or demos",
            "synthetic cohorts generated from GAN, VAE, KM reconstruction, or text-only reports",
            "pseudo-IPD reconstructed from Kaplan-Meier curves unless explicitly labelled non-real and excluded from real ML-NMR validation",
            "survey-only public IPD such as NHANES, MEPS, or NSDUH without randomized-trial treatment arms and shared aggregate evidence",
            "closed, proprietary, credentialed, or requester-only patient datasets",
        ],
        "required_next_artifacts": [
            "public_trial_ipd_manifest_with_nct_pmid_doi_or_oa_source",
            "ipd_row_hash_and_schema_validation",
            "aggregate_covariate_distribution_manifest_for_shared_network",
            "shared_estimand_and_treatment_mapping_between_ipd_and_aggregate_studies",
            "covariate_overlap_and_positivity_report",
            "multinma_mlnmr_reference_output_with_versions_hashes_and_tolerance",
        ],
        "certification_effect": "none",
    }


def test_mlnmr_source_coverage_rejects_pseudo_ipd_shortcuts_and_premature_certification():
    raw = _coverage_to_mapping(load_mlnmr_source_coverage(COVERAGE))
    raw["status"] = "active_source_backed_mlnmr_data"
    raw["registered_benchmark_ids"] = []
    with pytest.raises(MLNMRCoverageError, match="requires benchmark IDs"):
        MLNMRSourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_mlnmr_source_coverage(COVERAGE))
    raw["status"] = "missing_source_backed_mlnmr_data"
    raw["registered_benchmark_ids"] = ["simulated_psoriasis_mlnmr"]
    with pytest.raises(MLNMRCoverageError, match="cannot list"):
        MLNMRSourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_mlnmr_source_coverage(COVERAGE))
    raw["excluded_source_patterns"] = ["simulated example only"]
    with pytest.raises(MLNMRCoverageError, match="pseudo-IPD"):
        MLNMRSourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_mlnmr_source_coverage(COVERAGE))
    raw["certification_effect"] = "production_certified"
    with pytest.raises(MLNMRCoverageError, match="cannot certify"):
        MLNMRSourceCoverage.from_mapping(raw)

    raw = _coverage_to_mapping(load_mlnmr_source_coverage(COVERAGE))
    raw["formal_source_boundary_reason"] = "Excluded for convenience."
    with pytest.raises(MLNMRCoverageError, match="formal ML-NMR"):
        MLNMRSourceCoverage.from_mapping(raw)


def test_mlnmr_source_search_audit_records_excluded_candidates():
    with SOURCE_SEARCH.open("rb") as handle:
        audit = tomllib.load(handle)

    assert audit["schema_version"] == "mlnmr_source_search/v1"
    assert audit["status"] == "no_admissible_real_mlnmr_domain_found"
    assert "clinicaltrials_gov" in audit["allowed_evidence_sources"]
    assert "pubmed_abstract" in audit["allowed_evidence_sources"]
    assert "open_access_paper" in audit["allowed_evidence_sources"]

    candidates = {candidate["id"]: candidate for candidate in audit["candidates"]}
    assert {
        "portfolio_advanced_nma_pooling_example_mlnmr",
        "multinma_plaque_psoriasis_example",
        "multinma_ndmm_example",
        "jrssa_mlnmr_external_validation_ndmm",
    } <= set(candidates)
    assert all(candidate["eligibility"] == "excluded" for candidate in candidates.values())
    reasons = " ".join(candidate["exclusion_reason"].lower() for candidate in candidates.values())
    assert "synthetic" in reasons
    assert "simulated" in reasons
    assert "nct" in reasons
    assert "pmid" in reasons


def test_mlnmr_followup_source_search_excludes_single_trial_ipd_leads():
    with SOURCE_SEARCH_FOLLOWUP.open("rb") as handle:
        audit = tomllib.load(handle)

    assert audit["schema_version"] == "mlnmr_source_search/v1"
    assert audit["checked_at"] == "2026-07-18"
    assert audit["status"] == "no_admissible_real_mlnmr_domain_found"
    assert "clinicaltrials_gov" in audit["allowed_evidence_sources"]
    assert "pubmed_abstract" in audit["allowed_evidence_sources"]
    assert "open_access_paper" in audit["allowed_evidence_sources"]

    candidates = {candidate["id"]: candidate for candidate in audit["candidates"]}
    ippoms = candidates["nct00950248_ippoms_single_trial_lead"]
    assert ippoms["nct_id"] == "NCT00950248"
    assert ippoms["pmid"] == "32784117"
    assert ippoms["eligibility"] == "excluded"
    assert "single completed trial" in ippoms["exclusion_reason"]
    assert "connected IPD-plus-aggregate treatment network" in ippoms["exclusion_reason"]

    reasons = " ".join(candidate["exclusion_reason"].lower() for candidate in candidates.values())
    assert "pseudo-ipd" in reasons
    assert "simulated" in reasons
    assert "synthetic" in reasons
    assert "aggregate covariate" in reasons


def test_mlnmr_diabetes_source_search_excludes_controlled_access_ipd():
    with SOURCE_SEARCH_DIABETES.open("rb") as handle:
        audit = tomllib.load(handle)

    assert audit["schema_version"] == "mlnmr_source_search/v1"
    assert audit["checked_at"] == "2026-07-18"
    assert audit["status"] == "no_admissible_real_mlnmr_domain_found"

    candidates = {candidate["id"]: candidate for candidate in audit["candidates"]}
    diabetes = candidates["plos_medicine_t2d_frailty_ipd_nma_vivli"]
    assert diabetes["pmid"] == "40193407"
    assert diabetes["doi"] == "10.1371/journal.pmed.1004553"
    assert diabetes["eligibility"] == "excluded"
    reason = diabetes["exclusion_reason"].lower()
    assert "open-access ipd meta-analysis/nma" in reason
    assert "row-level trial ipd are not public" in reason
    assert "controlled-access" in reason


def _coverage_to_mapping(coverage: MLNMRSourceCoverage) -> dict[str, object]:
    return {
        "schema_version": MLNMR_SOURCE_COVERAGE_SCHEMA_VERSION,
        "checked_at": coverage.checked_at,
        "status": coverage.status,
        "model_status": coverage.model_status,
        "certification_effect": coverage.certification_effect,
        "purpose": coverage.purpose,
        "allowed_evidence_sources": list(coverage.allowed_evidence_sources),
        "protocol_only_sources": list(coverage.protocol_only_sources),
        "protocol_registry_rule": coverage.protocol_registry_rule,
        "formal_source_boundary_decision": coverage.formal_source_boundary_decision,
        "formal_source_boundary_reason": coverage.formal_source_boundary_reason,
        "formal_decision_artifacts": list(coverage.formal_decision_artifacts),
        "registered_benchmark_ids": list(coverage.registered_benchmark_ids),
        "registered_source_counts": dict(coverage.registered_source_counts),
        "required_source_components": list(coverage.required_source_components),
        "excluded_source_patterns": list(coverage.excluded_source_patterns),
        "source_search_summary": list(coverage.source_search_summary),
        "required_next_artifacts": list(coverage.required_next_artifacts),
        "claim_limit": coverage.claim_limit,
    }
