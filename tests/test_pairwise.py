import math

import numpy as np
import pytest

from bias_nma_adv.pairwise import (
    PairwiseMetaError,
    fit_pairwise_meta,
    leave_one_out_outlier_diagnostic,
)


def test_fixed_effect_matches_inverse_variance_reference():
    y = np.array([0.2, 0.4, -0.1])
    v = np.array([0.04, 0.09, 0.16])
    weights = 1.0 / v
    expected = float(np.sum(weights * y) / np.sum(weights))
    expected_se = math.sqrt(float(1.0 / np.sum(weights)))

    result = fit_pairwise_meta(y, v, method="FE")

    assert result.method == "FE"
    assert result.estimate == pytest.approx(expected, abs=1e-12)
    assert result.se == pytest.approx(expected_se, abs=1e-12)
    assert result.tau2 == 0.0
    assert result.ci_low < result.estimate < result.ci_high


def test_equal_variance_random_effect_estimators_match_closed_form():
    y = np.array([0.0, 1.0, 2.0])
    v = np.array([0.25, 0.25, 0.25])
    expected_tau2 = 0.75

    for method in ("DL", "PM", "REML"):
        result = fit_pairwise_meta(y, v, method=method)
        assert result.tau2 == pytest.approx(expected_tau2, abs=1e-6)
        assert result.estimate == pytest.approx(1.0, abs=1e-12)


def test_paule_mandel_tau2_solves_q_equals_degrees_of_freedom():
    y = np.array([-0.7, -0.1, 0.2, 0.8])
    v = np.array([0.04, 0.05, 0.06, 0.05])
    result = fit_pairwise_meta(y, v, method="PM")

    assert result.tau2 > 0.0
    assert result.q == pytest.approx(result.df, abs=1e-7)


def test_hksj_floor_prevents_zero_standard_error_when_residual_q_is_zero():
    y = np.array([0.2, 0.2, 0.2])
    v = np.array([0.04, 0.04, 0.04])

    floored = fit_pairwise_meta(y, v, method="FE", hksj=True, hksj_floor=True)
    raw = fit_pairwise_meta(y, v, method="FE", hksj=True, hksj_floor=False)
    ordinary = fit_pairwise_meta(y, v, method="FE", hksj=False)

    assert floored.hksj_q_factor == 1.0
    assert floored.se == pytest.approx(ordinary.se, abs=1e-12)
    assert raw.hksj_q_factor == 0.0
    assert raw.se == 0.0


def test_prediction_interval_is_wider_than_confidence_interval_when_tau_positive():
    y = np.array([0.0, 1.0, 2.0])
    v = np.array([0.25, 0.25, 0.25])
    result = fit_pairwise_meta(y, v, method="REML", prediction_interval=True)

    assert result.prediction_interval is not None
    ci_width = result.ci_high - result.ci_low
    pi_width = result.prediction_interval[1] - result.prediction_interval[0]
    assert result.tau2 > 0.0
    assert pi_width > ci_width


def test_leave_one_out_outlier_diagnostic_flags_discordant_study():
    y = np.array([0.0, 0.05, -0.02, 1.0])
    v = np.array([0.04, 0.04, 0.04, 0.04])

    diagnostic = leave_one_out_outlier_diagnostic(y, v, method="FE")

    assert diagnostic.method == "FE"
    assert diagnostic.full_estimate == pytest.approx(0.2575, abs=1e-12)
    assert len(diagnostic.diagnostics) == 4
    assert all(row.absolute_delta_estimate >= 0.0 for row in diagnostic.diagnostics)
    assert all(math.isfinite(row.standardized_delta) for row in diagnostic.diagnostics)
    assert any("do not replace full GOSH" in warning for warning in diagnostic.warnings)

    most_influential = max(
        diagnostic.diagnostics,
        key=lambda row: row.absolute_delta_estimate,
    )
    assert most_influential.omitted_index == 3
    assert most_influential.omitted_effect == pytest.approx(1.0)
    assert most_influential.estimate == pytest.approx(0.01, abs=1e-12)


def test_leave_one_out_outlier_diagnostic_requires_at_least_three_studies():
    with pytest.raises(PairwiseMetaError, match="at least three studies"):
        leave_one_out_outlier_diagnostic(
            np.array([0.0, 1.0]),
            np.array([0.04, 0.04]),
        )


def test_single_study_fails_closed_for_heterogeneity_but_returns_fixed_effect():
    result = fit_pairwise_meta(
        np.array([0.3]),
        np.array([0.04]),
        method="REML",
        hksj=True,
        prediction_interval=True,
    )

    assert result.estimate == pytest.approx(0.3)
    assert result.se == pytest.approx(0.2)
    assert result.tau2 == 0.0
    assert result.prediction_interval is None
    assert any("Only one study" in warning for warning in result.warnings)
    assert any("HKSJ" in warning for warning in result.warnings)
    assert any("Prediction interval" in warning for warning in result.warnings)


def test_invalid_pairwise_inputs_fail_closed():
    with pytest.raises(PairwiseMetaError, match="same length"):
        fit_pairwise_meta(np.array([0.1, 0.2]), np.array([0.1]))

    with pytest.raises(PairwiseMetaError, match="positive"):
        fit_pairwise_meta(np.array([0.1]), np.array([0.0]))

    with pytest.raises(PairwiseMetaError, match="method"):
        fit_pairwise_meta(np.array([0.1]), np.array([0.1]), method="magic")
