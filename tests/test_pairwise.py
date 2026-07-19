import math

import numpy as np
import pytest

from bias_nma_adv.pairwise import (
    PairwiseMetaError,
    fit_pairwise_meta,
    gosh_outlier_space_diagnostic,
    leave_one_out_outlier_diagnostic,
    numerical_stress_report,
    pairwise_numerical_stress_matrix,
    reml_local_minimum_diagnostic,
    tau2_cross_check_report,
    trim_and_fill_sensitivity,
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


def test_gosh_outlier_space_diagnostic_enumerates_bounded_subset_space():
    y = np.array([0.0, 0.05, -0.02, 1.0])
    v = np.array([0.04, 0.04, 0.04, 0.04])

    diagnostic = gosh_outlier_space_diagnostic(
        y,
        v,
        method="FE",
        min_subset_size=3,
    )

    assert diagnostic.method == "FE"
    assert diagnostic.full_estimate == pytest.approx(0.2575, abs=1e-12)
    assert diagnostic.full_tau2 == 0.0
    assert diagnostic.n_subsets == 5
    assert len(diagnostic.diagnostics) == 5
    assert diagnostic.max_abs_delta_estimate > 0.0
    assert any("exploratory screens" in warning for warning in diagnostic.warnings)

    most_extreme = max(
        diagnostic.diagnostics,
        key=lambda row: row.absolute_delta_estimate,
    )
    assert most_extreme.subset_indices == (0, 1, 2)
    assert most_extreme.k_subset == 3
    assert most_extreme.estimate == pytest.approx(0.01, abs=1e-12)
    assert most_extreme.delta_estimate == pytest.approx(-0.2475, abs=1e-12)


def test_gosh_outlier_space_diagnostic_fails_closed_when_subset_space_is_too_large():
    with pytest.raises(PairwiseMetaError, match="exceeding max_subsets"):
        gosh_outlier_space_diagnostic(
            np.zeros(6),
            np.full(6, 0.04),
            min_subset_size=2,
            max_subsets=3,
        )


def test_tau2_cross_check_report_compares_estimators_without_certifying_choice():
    y = np.array([-0.7, -0.1, 0.2, 0.8])
    v = np.array([0.04, 0.05, 0.06, 0.05])

    report = tau2_cross_check_report(y, v)

    by_method = {row.method: row for row in report.diagnostics}
    # FE is a common-effect fit, not a tau2 estimator, and must not appear here.
    assert set(by_method) == {"DL", "PM", "REML"}
    assert all(row.status == "passed" for row in report.diagnostics)
    assert report.primary_method == "REML"
    # Previously 0.0 -- FE hard-sets tau2=0, which pinned tau2_min to the boundary.
    assert report.tau2_min is not None and report.tau2_min > 0.0
    assert report.tau2_max is not None
    assert report.tau2_max > 0.0
    assert report.max_abs_estimate_delta is not None
    assert report.max_abs_se_delta is not None
    # FE is fully quarantined: it appears in NO field of the report body.
    assert report.estimate_signs == {"DL": 1, "PM": 1, "REML": 1}
    assert set(report.methods_crossing_null) == {"DL", "PM", "REML"}
    assert any("Alternative tau2" in warning for warning in report.warnings)
    # The old sign-change warning here was a pure FE artifact: FE estimated
    # -0.002, i.e. noise the other side of zero from three tau2 estimators that
    # all agree on +1.
    assert not any("change sign" in warning for warning in report.warnings)
    assert not any("Null-crossing" in warning for warning in report.warnings)

    # FE is still reported, just quarantined from every aggregate.
    assert report.common_effect_reference is not None
    assert report.common_effect_reference.method == "FE"
    assert report.common_effect_reference.tau2 == pytest.approx(0.0)


def test_tau2_cross_check_rejects_common_effect_method_as_tau2_estimator():
    """Seeded-defect guard: a caller must not be able to reintroduce FE."""
    y = np.array([-0.7, -0.1, 0.2, 0.8])
    v = np.array([0.04, 0.05, 0.06, 0.05])

    with pytest.raises(PairwiseMetaError, match="common-effect method"):
        tau2_cross_check_report(y, v, methods=("FE", "DL", "PM", "REML"))
    with pytest.raises(PairwiseMetaError, match="common-effect method"):
        tau2_cross_check_report(y, v, methods=("DL", "REML"), primary_method="FE")
    with pytest.raises(PairwiseMetaError, match="common-effect method"):
        numerical_stress_report(y, v, methods=("FE", "DL", "PM", "REML"))


def test_tau2_cross_check_still_warns_on_genuine_estimator_sign_change():
    """The sign alarm must survive removing FE -- proven on a real DL/PM/REML split.

    DL gives -0.0079 (tau2=0.215) while PM/REML give +0.006 (tau2~0.48). No
    common-effect fit is involved, so this is genuine estimator disagreement.
    """
    y = np.array([-0.812, 0.025, 0.887, 0.576, -0.565])
    v = np.array([0.012085, 0.000616, 0.075931, 0.001391, 0.018443])

    report = tau2_cross_check_report(y, v)

    assert report.estimate_signs["DL"] == -1
    assert report.estimate_signs["PM"] == 1
    assert report.estimate_signs["REML"] == 1
    assert any("change sign" in warning for warning in report.warnings)


def test_numerical_stress_report_flags_dominant_study_and_boundary_tau2():
    y = np.array([0.0, 0.1, 0.2])
    v = np.array([0.0001, 1.0, 1.0])

    report = numerical_stress_report(y, v, dominant_weight_threshold=0.80)

    assert report.k == 3
    assert report.status == "warning"
    assert report.max_weight_fraction > 0.99
    assert report.min_variance == pytest.approx(0.0001)
    assert report.max_variance == pytest.approx(1.0)
    assert any("Dominant-study" in warning for warning in report.warnings)


def test_reml_local_minimum_diagnostic_profiles_tau2_objective():
    y = np.array([0.0, 1.0, 2.0])
    v = np.array([0.25, 0.25, 0.25])

    diagnostic = reml_local_minimum_diagnostic(y, v, n_grid=41, tau2_upper=2.0)

    assert diagnostic.k == 3
    assert diagnostic.optimizer_tau2 == pytest.approx(0.75, abs=1e-6)
    assert diagnostic.best_grid_tau2 == pytest.approx(0.75, abs=0.05)
    assert diagnostic.objective_gap <= 0.0
    assert diagnostic.local_minima
    assert diagnostic.boundary_minimum is False
    assert len(diagnostic.profile) == 41
    assert any("numerical screens" in warning for warning in diagnostic.warnings)


def test_reml_local_minimum_diagnostic_flags_boundary_minimum():
    y = np.array([0.2, 0.2, 0.2])
    v = np.array([0.04, 0.04, 0.04])

    diagnostic = reml_local_minimum_diagnostic(y, v, n_grid=11, tau2_upper=1.0)

    assert diagnostic.optimizer_tau2 == pytest.approx(0.0)
    assert diagnostic.best_grid_tau2 == pytest.approx(0.0)
    assert diagnostic.boundary_minimum is True
    assert any("boundary" in warning for warning in diagnostic.warnings)


def test_pairwise_numerical_stress_matrix_reports_scenario_statuses():
    matrix = pairwise_numerical_stress_matrix(
        {
            "stable": (
                np.array([0.0, 1.0, 2.0]),
                np.array([0.25, 0.25, 0.25]),
            ),
            "dominant": (
                np.array([0.0, 0.1, 0.2]),
                np.array([0.0001, 1.0, 1.0]),
            ),
            "invalid": (
                np.array([0.0, 0.1]),
                np.array([0.04, 0.0]),
            ),
        },
        n_grid=17,
    )

    assert matrix.n_scenarios == 3
    assert matrix.status_counts["failed"] == 1
    assert matrix.status_counts["warning"] == 2
    by_id = {scenario.scenario_id: scenario for scenario in matrix.scenarios}
    assert by_id["stable"].stress_report is not None
    assert by_id["stable"].reml_profile is not None
    assert by_id["dominant"].status == "warning"
    assert by_id["invalid"].status == "failed"
    assert by_id["invalid"].error is not None
    assert any("reference-package" in warning for warning in matrix.warnings)


def test_trim_and_fill_sensitivity_mirrors_overrepresented_side():
    y = np.array([0.0, 0.8, 0.9, 1.2])
    se = np.array([0.2, 0.2, 0.2, 0.2])

    sensitivity = trim_and_fill_sensitivity(y, se, method="FE", side="left")

    assert sensitivity.k_observed == 4
    assert sensitivity.k_filled == 2
    assert sensitivity.fill_side == "left"
    assert sensitivity.observed_estimate == pytest.approx(0.725, abs=1e-12)
    assert sensitivity.adjusted_estimate < sensitivity.observed_estimate
    assert len(sensitivity.filled_effects) == 2
    assert len(sensitivity.filled_variances) == 2
    assert any("not reference-matched" in warning for warning in sensitivity.warnings)


def test_trim_and_fill_sensitivity_fails_closed_for_invalid_inputs():
    with pytest.raises(PairwiseMetaError, match="at least three"):
        trim_and_fill_sensitivity(np.array([0.0, 1.0]), np.array([0.2, 0.2]))

    with pytest.raises(PairwiseMetaError, match="side"):
        trim_and_fill_sensitivity(
            np.array([0.0, 0.1, 0.2]),
            np.array([0.2, 0.2, 0.2]),
            side="magic",
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


def test_null_crossing_guard_enforces_log_scale_and_positive_se():
    """The counting site hard-codes the log-scale null; the invariant is now guarded."""
    from bias_nma_adv.pairwise import PairwiseMethodDiagnostic, crosses_log_scale_null

    def diag(**kw):
        base = dict(
            method="DL", status="passed", estimate=0.1, se=0.2,
            ci_low=-0.3, ci_high=0.5, tau2=0.0, q=1.0, df=3,
            warnings=(), error=None,
        )
        base.update(kw)
        return PairwiseMethodDiagnostic(**base)

    # Normal log-scale behaviour.
    assert crosses_log_scale_null(diag()) is True
    assert crosses_log_scale_null(diag(ci_low=0.2, ci_high=0.5)) is False
    # Missing interval is simply not counted.
    assert crosses_log_scale_null(diag(ci_low=None)) is False

    # se <= 0 makes the count meaningless -- must raise, not silently count.
    with pytest.raises(PairwiseMetaError, match="non-positive standard error"):
        crosses_log_scale_null(diag(se=0.0))
    with pytest.raises(PairwiseMetaError, match="non-positive standard error"):
        crosses_log_scale_null(diag(se=-0.1))

    # An inverted interval is incoherent and must raise.
    with pytest.raises(PairwiseMetaError, match="inverted interval"):
        crosses_log_scale_null(diag(ci_low=0.9, ci_high=0.1))

    # A ratio-scale interval (null = 1.0) is NOT counted as crossing, which is
    # the correct log-scale answer -- the guard cannot rescue a caller who
    # passes the wrong scale, but it no longer inverts silently on bad SEs.
    assert crosses_log_scale_null(diag(estimate=1.05, ci_low=0.85, ci_high=1.30)) is False
