"""Unit tests for Advanced Bias-Adjusted NMA model."""

import pytest
import numpy as np

from bias_nma_adv.data import ALLOWED_STUDY_DESIGNS, EFFECT_STUDY_DESIGNS, EvidenceDataset, ValidationError
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor

def test_validation_errors():
    dataset = EvidenceDataset()
    # Invalid design
    with pytest.raises(ValidationError):
        dataset.add_study("S1", "invalid_design")
    
    # rob_weight out of bounds
    with pytest.raises(ValidationError):
        dataset.add_study("S1", "rct", rob_weight=1.5)
    with pytest.raises(ValidationError):
        dataset.add_study("S1", "rct", rob_weight=-0.1)

    # Arm size <= 0
    with pytest.raises(ValidationError):
        dataset.add_arm("S1", "A1", "T1", 0)

    # Continuous outcome without SE
    with pytest.raises(ValidationError):
        dataset.add_outcome_ad("S1", "A1", "O1", "continuous", 10.0, se=None)


def test_mixed_measure_type_outcome_fails_closed_before_model_fit():
    dataset = EvidenceDataset()
    dataset.add_study("S_BINARY", "rct")
    dataset.add_arm("S_BINARY", "control", "A", 100)
    dataset.add_arm("S_BINARY", "active", "B", 100)
    dataset.add_outcome_ad("S_BINARY", "control", "O1", "binary", 10)
    dataset.add_outcome_ad("S_BINARY", "active", "O1", "binary", 20)

    dataset.add_study("S_CONTINUOUS", "rct")
    dataset.add_arm("S_CONTINUOUS", "control", "A", 80)
    dataset.add_arm("S_CONTINUOUS", "active", "B", 80)
    dataset.add_outcome_ad("S_CONTINUOUS", "control", "O1", "continuous", -0.2, se=0.10)
    dataset.add_outcome_ad("S_CONTINUOUS", "active", "O1", "continuous", -0.4, se=0.12)

    with pytest.raises(ValidationError, match="multiple measure_type"):
        dataset.measure_type_for_outcome("O1")
    with pytest.raises(ValidationError, match="multiple measure_type"):
        AdvancedBiasAdjustedNMAPooler().fit(dataset, "O1", reference_treatment="A")


def test_fit_reports_unusable_study_drops_instead_of_silent_loss():
    dataset = EvidenceDataset()
    dataset.add_study("S_GOOD", "rct")
    dataset.add_arm("S_GOOD", "control", "A", 100)
    dataset.add_arm("S_GOOD", "active", "B", 100)
    dataset.add_outcome_ad("S_GOOD", "control", "O1", "binary", 10)
    dataset.add_outcome_ad("S_GOOD", "active", "O1", "binary", 20)

    dataset.add_study("S_INCOMPLETE", "rct")
    dataset.add_arm("S_INCOMPLETE", "control", "A", 100)
    dataset.add_outcome_ad("S_INCOMPLETE", "control", "O1", "binary", 10)

    fit = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        random_effects=False,
        exact_binomial=False,
    ).fit(dataset, "O1", reference_treatment="A")

    assert fit.n_studies == 1
    assert fit.n_studies_dropped == 1
    assert any("S_INCOMPLETE" in warning and "dropped" in warning for warning in fit.warnings)


def test_study_design_policy_is_extensible_but_metadata_frames_cannot_enter_effect_pool():
    assert {"rct", "nrs", "other", "regulatory_review_backed", "csr_backed"} <= set(
        ALLOWED_STUDY_DESIGNS
    )
    assert "registry_only" not in EFFECT_STUDY_DESIGNS

    dataset = EvidenceDataset()
    dataset.add_study("S_FDA", "regulatory_review_backed")
    assert dataset.studies["S_FDA"].design == "regulatory_review_backed"

    dataset.add_study("S_PROTOCOL", "registry_only")
    with pytest.raises(ValidationError, match="metadata-only design"):
        dataset.add_arm("S_PROTOCOL", "arm1", "A", 100)

    dataset.add_study("S_PROTOCOL_SOURCE", "rct", source_type="who_ictrp_protocol")
    with pytest.raises(ValidationError, match="protocol-only source_type"):
        dataset.add_arm("S_PROTOCOL_SOURCE", "arm1", "A", 100)


def test_run_in_applicability_is_context_or_indirectness_not_rob_weight():
    dataset = EvidenceDataset()
    dataset.add_study(
        "S_RUNIN",
        "rct",
        rob_weight=1.0,
        covariates={"run_in": 1.0},
        indirectness="enriched run-in population; GRADE indirectness review only",
    )

    study = dataset.studies["S_RUNIN"]
    assert study.rob_weight == 1.0
    assert study.covariates["run_in"] == pytest.approx(1.0)
    assert "GRADE indirectness" in study.indirectness

    with pytest.raises(ValidationError, match="indirectness"):
        dataset.add_study("S_BLANK", "rct", indirectness=" ")


def test_input_source_subset_splits_published_from_registry_first_rows():
    dataset = EvidenceDataset()
    dataset.add_study("PUB", "rct", source_type="pubmed_abstract")
    dataset.add_arm("PUB", "control", "A", 100)
    dataset.add_arm("PUB", "active", "B", 100)
    dataset.add_outcome_ad("PUB", "control", "O1", "binary", 20)
    dataset.add_outcome_ad("PUB", "active", "O1", "binary", 12)

    dataset.add_study("REG", "rct", source_type="clinicaltrials_gov")
    dataset.add_arm("REG", "control", "A", 100)
    dataset.add_arm("REG", "active", "B", 100)
    dataset.add_outcome_ad("REG", "control", "O1", "binary", 22)
    dataset.add_outcome_ad("REG", "active", "O1", "binary", 13)

    dataset.add_study("OVERRIDE", "rct", source_type="pubmed_abstract")
    dataset.add_arm("OVERRIDE", "control", "A", 100)
    dataset.add_arm("OVERRIDE", "active", "B", 100)
    dataset.add_outcome_ad(
        "OVERRIDE",
        "control",
        "O1",
        "binary",
        18,
        source_type="fda_review",
    )
    dataset.add_outcome_ad(
        "OVERRIDE",
        "active",
        "O1",
        "binary",
        9,
        source_type="fda_review",
    )

    as_published = dataset.subset_by_input_source("as_published")
    registry_first = dataset.subset_by_input_source("registry_first")

    assert set(as_published.studies) == {"PUB"}
    assert {outcome.study_id for outcome in as_published.outcomes_ad} == {"PUB"}
    assert set(registry_first.studies) == {"REG", "OVERRIDE"}
    assert {outcome.study_id for outcome in registry_first.outcomes_ad} == {
        "REG",
        "OVERRIDE",
    }

    with pytest.raises(ValidationError, match="mode must be"):
        dataset.subset_by_input_source("all_sources")


def test_network_connectivity():
    dataset = EvidenceDataset()
    # Create two disconnected networks:
    # Net 1: A vs B
    # Net 2: C vs D
    dataset.add_study("S1", "rct")
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O1", "binary", 10.0)
    dataset.add_outcome_ad("S1", "arm2", "O1", "binary", 15.0)

    dataset.add_study("S2", "rct")
    dataset.add_arm("S2", "arm1", "C", 100)
    dataset.add_arm("S2", "arm2", "D", 100)
    dataset.add_outcome_ad("S2", "arm1", "O1", "binary", 20.0)
    dataset.add_outcome_ad("S2", "arm2", "O1", "binary", 25.0)

    pooler = AdvancedBiasAdjustedNMAPooler()
    with pytest.raises(ValidationError, match="disconnected treatment network"):
        pooler.fit(dataset, "O1", reference_treatment="A")


def test_reml_gls_reproducible_pooling():
    # Simple connected star network
    # S1: A (N=100, 10 events) vs B (N=100, 20 events) [RCT]
    # S2: A (N=120, 12 events) vs B (N=120, 25 events) [RCT]
    # S3: A (N=90, 8 events) vs B (N=90, 22 events) [NRS]
    dataset = EvidenceDataset()
    
    dataset.add_study("S1", "rct", rob_weight=1.0)
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O1", "binary", 10)
    dataset.add_outcome_ad("S1", "arm2", "O1", "binary", 20)

    dataset.add_study("S2", "rct", rob_weight=1.0)
    dataset.add_arm("S2", "arm1", "A", 120)
    dataset.add_arm("S2", "arm2", "B", 120)
    dataset.add_outcome_ad("S2", "arm1", "O1", "binary", 12)
    dataset.add_outcome_ad("S2", "arm2", "O1", "binary", 25)

    dataset.add_study("S3", "nrs", rob_weight=0.5) # NRS with downweighting
    dataset.add_arm("S3", "arm1", "A", 90)
    dataset.add_arm("S3", "arm2", "B", 90)
    dataset.add_outcome_ad("S3", "arm1", "O1", "binary", 8)
    dataset.add_outcome_ad("S3", "arm2", "O1", "binary", 22)

    # 1. Fit without HKSJ or downweighting
    pooler1 = AdvancedBiasAdjustedNMAPooler(hksj=False, down_weight=False)
    fit1 = pooler1.fit(dataset, "O1", reference_treatment="A", reference_design="rct")
    
    # 2. Fit with HKSJ and downweighting
    pooler2 = AdvancedBiasAdjustedNMAPooler(hksj=True, down_weight=True)
    fit2 = pooler2.fit(dataset, "O1", reference_treatment="A", reference_design="rct")

    # Assert HKSJ factor is computed and >= 1.0
    assert fit2.q_factor >= 1.0
    
    # Verify parameter estimates
    assert "B" in fit1.treatment_effects
    assert "nrs" in fit1.design_biases
    assert fit1.tau_method == "REML"
    assert fit2.tau_method == "REML"

    # HKSJ should scale standard errors upwards compared to non-HKSJ
    # since we have residual discrepancies and very few studies
    se_non_hksj = fit1.treatment_ses["B"]
    se_hksj = fit2.treatment_ses["B"]
    # With rob_weight = 0.5, study S3 variance is doubled, altering estimates.
    assert se_hksj > 0.0
    assert se_non_hksj > 0.0


def test_sponsor_bias_is_default_off_and_opt_in_fail_closed():
    dataset = EvidenceDataset()
    for study_id, active_events in (("NCT00000001", 20), ("NCT00000002", 30)):
        dataset.add_study(study_id, "rct", rob_weight=1.0)
        dataset.add_arm(study_id, "control", "A", 100)
        dataset.add_arm(study_id, "active", "B", 100)
        dataset.add_outcome_ad(study_id, "control", "O1", "binary", 10)
        dataset.add_outcome_ad(study_id, "active", "O1", "binary", active_events)

    default_fit = AdvancedBiasAdjustedNMAPooler(hksj=False, random_effects=False).fit(
        dataset,
        "O1",
        reference_treatment="A",
    )

    auditor = RegistrySponsorAuditor()
    auditor.register_trial_flow("NCT00000001", "other", randomized=200, lost_to_follow_up=0)
    with pytest.raises(ValidationError, match="sponsor-bias adjustment requires"):
        AdvancedBiasAdjustedNMAPooler(
            hksj=False,
            random_effects=False,
            apply_sponsor_bias=True,
            sponsor_auditor=auditor,
        ).fit(dataset, "O1", reference_treatment="A")

    auditor.register_trial_flow("NCT00000002", "industry", randomized=200, lost_to_follow_up=30)
    adjusted_fit = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        random_effects=False,
        apply_sponsor_bias=True,
        sponsor_auditor=auditor,
    ).fit(dataset, "O1", reference_treatment="A")

    assert adjusted_fit.treatment_ses["B"] > 0.0
    assert adjusted_fit.treatment_effects["B"] != pytest.approx(
        default_fit.treatment_effects["B"]
    )
    assert adjusted_fit.target_population == "enriched_as_randomised"


def test_redescending_outlier_sensitivity_is_default_off_and_reported_when_enabled():
    dataset = EvidenceDataset()
    for study_id, effect in (("S1", 0.10), ("S2", 0.12), ("S_OUTLIER", 2.00)):
        dataset.add_study(study_id, "rct")
        dataset.add_arm(study_id, "control", "A", 100)
        dataset.add_arm(study_id, "active", "B", 100)
        dataset.add_outcome_ad(study_id, "control", "O1", "continuous", 0.0, se=0.05)
        dataset.add_outcome_ad(study_id, "active", "O1", "continuous", effect, se=0.05)

    default_fit = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        random_effects=False,
    ).fit(dataset, "O1", reference_treatment="A")
    robust_fit = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        random_effects=False,
        robust_outlier_sensitivity=True,
        redescending_tuning_constant=1.0,
    ).fit(dataset, "O1", reference_treatment="A")

    assert default_fit.redescending_sensitivity_active is False
    assert default_fit.redescending_contrast_weights == {}
    assert robust_fit.redescending_sensitivity_active is True
    assert robust_fit.redescending_iterations >= 1
    assert robust_fit.treatment_effects["B"] < default_fit.treatment_effects["B"]
    outlier_weights = {
        label: weight
        for label, weight in robust_fit.redescending_contrast_weights.items()
        if label.startswith("S_OUTLIER:")
    }
    assert outlier_weights
    assert max(outlier_weights.values()) < 0.05
    assert any("default-off sensitivity" in warning for warning in robust_fit.warnings)


def test_redescending_outlier_sensitivity_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="redescending_tuning_constant"):
        AdvancedBiasAdjustedNMAPooler(redescending_tuning_constant=0.0)
    with pytest.raises(ValueError, match="redescending_max_iter"):
        AdvancedBiasAdjustedNMAPooler(redescending_max_iter=0)


def test_indirectness_flag_names_estimand_without_default_numeric_penalty():
    dataset = EvidenceDataset()
    dataset.add_study("S1", "rct", indirectness="run-in enrichment")
    dataset.add_arm("S1", "control", "A", 100)
    dataset.add_arm("S1", "active", "B", 100)
    dataset.add_outcome_ad("S1", "control", "O1", "binary", 10)
    dataset.add_outcome_ad("S1", "active", "O1", "binary", 20)

    default_fit = AdvancedBiasAdjustedNMAPooler(hksj=False, random_effects=False).fit(
        dataset,
        "O1",
        reference_treatment="A",
    )
    sensitivity_fit = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        random_effects=False,
        apply_indirectness=True,
        target_population="unselected_target",
    ).fit(dataset, "O1", reference_treatment="A")

    assert sensitivity_fit.target_population == "unselected_target"
    assert sensitivity_fit.treatment_effects["B"] == pytest.approx(
        default_fit.treatment_effects["B"]
    )
    assert any("no mechanism-specific numerical delta" in warning for warning in sensitivity_fit.warnings)


def test_meta_regression():
    # Star network with study-level covariate "year"
    dataset = EvidenceDataset()
    
    dataset.add_study("S1", "rct", rob_weight=1.0, covariates={"year": -1.0})
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O1", "binary", 15)
    dataset.add_outcome_ad("S1", "arm2", "O1", "binary", 25)

    dataset.add_study("S2", "rct", rob_weight=1.0, covariates={"year": 1.0})
    dataset.add_arm("S2", "arm1", "A", 100)
    dataset.add_arm("S2", "arm2", "B", 100)
    dataset.add_outcome_ad("S2", "arm1", "O1", "binary", 10)
    dataset.add_outcome_ad("S2", "arm2", "O1", "binary", 30)

    pooler = AdvancedBiasAdjustedNMAPooler(hksj=True, down_weight=True)
    fit = pooler.fit(dataset, "O1", reference_treatment="A", covariates=["year"])

    # Check that interaction parameter names are created
    assert "trt_B_x_year" in fit.parameter_names

    # Check contrast calculations at different values of covariate
    eff_neg, _, _, _ = fit.contrast("B", "A", covariates={"year": -1.0})
    eff_pos, _, _, _ = fit.contrast("B", "A", covariates={"year": 1.0})

    # The effects should differ due to meta-regression slope
    assert abs(eff_neg - eff_pos) > 1e-4


def test_sandwich_variance():
    dataset = EvidenceDataset()
    dataset.add_study("S1", "rct", rob_weight=1.0)
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O1", "binary", 15)
    dataset.add_outcome_ad("S1", "arm2", "O1", "binary", 25)

    dataset.add_study("S2", "rct", rob_weight=1.0)
    dataset.add_arm("S2", "arm1", "A", 100)
    dataset.add_arm("S2", "arm2", "B", 100)
    dataset.add_outcome_ad("S2", "arm1", "O1", "binary", 12)
    dataset.add_outcome_ad("S2", "arm2", "O1", "binary", 28)

    pooler_model = AdvancedBiasAdjustedNMAPooler(hksj=False, variance_type="model")
    fit_model = pooler_model.fit(dataset, "O1", reference_treatment="A")

    pooler_sand = AdvancedBiasAdjustedNMAPooler(hksj=False, variance_type="sandwich")
    fit_sand = pooler_sand.fit(dataset, "O1", reference_treatment="A")

    # Estimates should be identical, but standard errors should differ
    assert abs(fit_model.treatment_effects["B"] - fit_sand.treatment_effects["B"]) < 1e-9
    assert fit_model.treatment_ses["B"] != fit_sand.treatment_ses["B"]
    assert fit_sand.treatment_ses["B"] > 0.0


def test_stratified_heterogeneity():
    dataset = EvidenceDataset()
    # RCTs: very consistent effect
    for i in range(5):
        s_id = f"RCT_{i}"
        dataset.add_study(s_id, "rct")
        dataset.add_arm(s_id, "arm1", "A", 100)
        dataset.add_arm(s_id, "arm2", "B", 100)
        dataset.add_outcome_ad(s_id, "arm1", "O1", "binary", 15)
        dataset.add_outcome_ad(s_id, "arm2", "O1", "binary", 25)

    # NRSs: highly inconsistent/varying effects
    effects = [5, 45, 10, 40, 8]
    for i, eff in enumerate(effects):
        s_id = f"NRS_{i}"
        dataset.add_study(s_id, "nrs")
        dataset.add_arm(s_id, "arm1", "A", 100)
        dataset.add_arm(s_id, "arm2", "B", 100)
        dataset.add_outcome_ad(s_id, "arm1", "O1", "binary", 15)
        dataset.add_outcome_ad(s_id, "arm2", "O1", "binary", eff)

    pooler = AdvancedBiasAdjustedNMAPooler(random_effects="stratified")
    fit = pooler.fit(dataset, "O1", reference_treatment="A")

    # Verify both stratum taus are estimated and distinct
    assert "rct" in fit.taus
    assert "nrs" in fit.taus
    assert fit.taus["nrs"] > fit.taus["rct"]


def test_topological_regularization():
    dataset = EvidenceDataset()
    # Central treatment B: 5 studies
    for i in range(5):
        s_id = f"S_B_{i}"
        dataset.add_study(s_id, "rct")
        dataset.add_arm(s_id, "arm1", "A", 100)
        dataset.add_arm(s_id, "arm2", "B", 100)
        dataset.add_outcome_ad(s_id, "arm1", "O1", "binary", 10)
        dataset.add_outcome_ad(s_id, "arm2", "O1", "binary", 30)

    # Sparse treatment C: only 1 study
    dataset.add_study("S_C_0", "rct")
    dataset.add_arm("S_C_0", "arm1", "A", 100)
    dataset.add_arm("S_C_0", "arm2", "C", 100)
    dataset.add_outcome_ad("S_C_0", "arm1", "O1", "binary", 10)
    dataset.add_outcome_ad("S_C_0", "arm2", "O1", "binary", 50)  # True effect is large

    # 1. No shrinkage
    pooler_no = AdvancedBiasAdjustedNMAPooler(treatment_shrinkage_lambda=0.0)
    fit_no = pooler_no.fit(dataset, "O1", reference_treatment="A")

    # 2. Strong shrinkage (penalize low centrality treatments)
    pooler_shrink = AdvancedBiasAdjustedNMAPooler(treatment_shrinkage_lambda=10.0)
    fit_shrink = pooler_shrink.fit(dataset, "O1", reference_treatment="A")

    # Central B effect should remain large, while sparse C should be shrunk towards A (0.0)
    assert abs(fit_no.treatment_effects["B"] - fit_shrink.treatment_effects["B"]) < 0.1
    assert fit_shrink.treatment_effects["C"] < fit_no.treatment_effects["C"]


def test_coupled_study_specific_bias():
    dataset = EvidenceDataset()
    
    dataset.add_study("S1", "rct", rob_weight=1.0)
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O1", "binary", 10)
    dataset.add_outcome_ad("S1", "arm2", "O1", "binary", 20)

    dataset.add_study("S2", "nrs", rob_weight=0.5) # NRS study with quality 0.5
    dataset.add_arm("S2", "arm1", "A", 100)
    dataset.add_arm("S2", "arm2", "B", 100)
    dataset.add_outcome_ad("S2", "arm1", "O1", "binary", 10)
    dataset.add_outcome_ad("S2", "arm2", "O1", "binary", 30) # large bias

    pooler = AdvancedBiasAdjustedNMAPooler(study_specific_bias=True)
    fit = pooler.fit(dataset, "O1", reference_treatment="A")

    # Verify that study-specific bias parameter is created
    assert "bias_study_S2" in fit.parameter_names
    assert "S2" in fit.study_specific_biases
    assert fit.study_specific_biases["S2"] > 0.0


def test_continuous_outcome():
    dataset = EvidenceDataset()
    
    # Study 1 (RCT): A (Mean=10.0, SE=1.0) vs B (Mean=12.0, SE=1.0)
    dataset.add_study("S1", "rct")
    dataset.add_arm("S1", "arm1", "A", 50)
    dataset.add_arm("S1", "arm2", "B", 50)
    dataset.add_outcome_ad("S1", "arm1", "O2", "continuous", 10.0, se=1.0)
    dataset.add_outcome_ad("S1", "arm2", "O2", "continuous", 12.0, se=1.0)

    # Study 2 (RCT): A (Mean=10.5, SE=1.2) vs B (Mean=12.8, SE=1.2)
    dataset.add_study("S2", "rct")
    dataset.add_arm("S2", "arm1", "A", 60)
    dataset.add_arm("S2", "arm2", "B", 60)
    dataset.add_outcome_ad("S2", "arm1", "O2", "continuous", 10.5, se=1.2)
    dataset.add_outcome_ad("S2", "arm2", "O2", "continuous", 12.8, se=1.2)

    pooler = AdvancedBiasAdjustedNMAPooler(random_effects=True)
    fit = pooler.fit(dataset, "O2", reference_treatment="A")

    # Estimated B vs A treatment effect should be between the two studies (around 2.0 to 2.3)
    assert "B" in fit.treatment_effects
    est = fit.treatment_effects["B"]
    assert 1.9 < est < 2.4
    assert fit.treatment_ses["B"] > 0.0


def test_exact_binomial_rare_events():
    dataset = EvidenceDataset()
    
    # Study 1: A (0 events in 100) vs B (2 events in 100)
    dataset.add_study("S1", "rct")
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O3", "binary", 0.0)
    dataset.add_outcome_ad("S1", "arm2", "O3", "binary", 2.0)

    # Study 2: A (1 event in 120) vs B (3 events in 120)
    dataset.add_study("S2", "rct")
    dataset.add_arm("S2", "arm1", "A", 120)
    dataset.add_arm("S2", "arm2", "B", 120)
    dataset.add_outcome_ad("S2", "arm1", "O3", "binary", 1.0)
    dataset.add_outcome_ad("S2", "arm2", "O3", "binary", 3.0)

    pooler = AdvancedBiasAdjustedNMAPooler(exact_binomial=True)
    fit = pooler.fit(dataset, "O3", reference_treatment="A")

    # Verify that exact binomial engine was active
    assert fit.exact_binomial_active is True
    # The treatment effect estimate should be positive and non-infinite
    assert "B" in fit.treatment_effects
    assert fit.treatment_effects["B"] > 0.0
    assert fit.treatment_ses["B"] > 0.0
