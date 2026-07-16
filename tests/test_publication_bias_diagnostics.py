import numpy as np
import pytest

from bias_nma_adv.publication_bias import (
    RegistryPublicationBiasAuditor,
    egger_regression_diagnostic,
    selection_weight_sensitivity,
)


def test_egger_regression_diagnostic_recovers_small_study_intercept():
    standard_errors = np.array(
        [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]
    )
    precision = 1.0 / standard_errors
    noise = np.array(
        [-0.03, 0.02, 0.01, -0.02, 0.03, -0.01, 0.00, 0.02, -0.02, 0.01, -0.01, 0.00]
    )
    standard_normal_deviate = 0.25 + 0.6 * precision + noise
    effects = standard_normal_deviate * standard_errors

    diagnostic = egger_regression_diagnostic(effects, standard_errors)

    assert diagnostic.k == 12
    assert diagnostic.residual_df == 10
    assert diagnostic.intercept == pytest.approx(0.2564922076)
    assert diagnostic.intercept_se == pytest.approx(0.0156165167)
    assert diagnostic.t_value == pytest.approx(16.4244186218)
    assert diagnostic.p_value == pytest.approx(1.4579126872e-08)
    assert diagnostic.slope == pytest.approx(0.5987990326)
    assert diagnostic.warnings == ()


def test_egger_regression_diagnostic_warns_when_underpowered():
    standard_errors = np.array([0.10, 0.12, 0.14, 0.16, 0.18])
    effects = np.array([0.622, 0.6324, 0.6364, 0.6368, 0.6504])

    diagnostic = egger_regression_diagnostic(effects, standard_errors)

    assert diagnostic.k == 5
    assert diagnostic.warnings == (
        "Egger regression is underpowered with fewer than 10 studies.",
    )


def test_egger_regression_diagnostic_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least three studies"):
        egger_regression_diagnostic(np.array([0.1, 0.2]), np.array([0.1, 0.1]))

    with pytest.raises(ValueError, match="finite and positive"):
        egger_regression_diagnostic(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.0, 0.1]),
        )


def test_selection_weight_sensitivity_uses_prespecified_probabilities():
    effects = np.array([0.1, 0.2, 0.8])
    standard_errors = np.array([0.1, 0.1, 0.1])
    selection_probabilities = np.array([1.0, 1.0, 0.25])

    sensitivity = selection_weight_sensitivity(
        effects,
        standard_errors,
        selection_probabilities,
    )

    assert sensitivity.k == 3
    assert sensitivity.observed_estimate == pytest.approx(0.3666666667)
    assert sensitivity.observed_se == pytest.approx(0.0577350269)
    assert sensitivity.adjusted_estimate == pytest.approx(0.5833333333)
    assert sensitivity.adjusted_se == pytest.approx(0.0707106781)
    assert sensitivity.selection_probabilities == (1.0, 1.0, 0.25)
    assert sensitivity.inverse_selection_weights == (1.0, 1.0, 4.0)
    assert any("not a publication-bias proof" in warning for warning in sensitivity.warnings)
    assert any("below 0.5" in warning for warning in sensitivity.warnings)


def test_selection_weight_sensitivity_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same length"):
        selection_weight_sensitivity(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.1]),
            np.array([1.0]),
        )

    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        selection_weight_sensitivity(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.1]),
            np.array([1.0, 0.0]),
        )


def test_registry_unpublished_ratio_requires_drug_specific_intervention_metadata():
    auditor = RegistryPublicationBiasAuditor()
    auditor.register_trial_protocol("NCT00000001", "mace", "mace", "completed")
    auditor.register_trial_protocol("NCT00000002", "mace", "mace", "completed")

    with pytest.raises(ValueError, match="requires intervention metadata"):
        auditor.calculate_unpublished_ratio("DrugX", ["NCT00000001"])


def test_registry_unpublished_ratio_filters_by_drug_intervention():
    auditor = RegistryPublicationBiasAuditor()
    auditor.register_trial_protocol("NCT00000001", "mace", "mace", "completed", ["DrugX"])
    auditor.register_trial_protocol("NCT00000002", "mace", "mace", "completed", ["DrugX"])
    auditor.register_trial_protocol("NCT00000003", "mace", "mace", "completed", ["OtherDrug"])
    auditor.register_trial_protocol("NCT00000004", "mace", "mace", "withdrawn", ["DrugX"])

    assert auditor.calculate_unpublished_ratio("DrugX", ["NCT00000001"]) == pytest.approx(0.5)
