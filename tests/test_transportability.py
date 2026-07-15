import math

import numpy as np
import pytest

from bias_nma_adv.transportability import (
    TransportabilityError,
    bottleneck_distance,
    certify_support,
    compute_persistence,
    count_subpopulations,
    fit_meta_regression,
    is_in_convex_hull,
    supported_transport,
    transport_effect,
)


def test_transport_meta_regression_recovers_linear_effect_at_target():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = 0.10 + 0.05 * x
    v = np.full_like(y, 0.01)

    fit = fit_meta_regression(y, v, x, effect_scale="RD")
    transported = transport_effect(fit, np.array([4.0]))

    assert fit.tau2 == pytest.approx(0.0, abs=1e-8)
    assert fit.beta[0] == pytest.approx(0.10, abs=1e-10)
    assert fit.beta[1] == pytest.approx(0.05, abs=1e-10)
    assert transported.estimate == pytest.approx(0.30, abs=1e-10)
    assert transported.ci_low < transported.estimate < transported.ci_high


def test_target_covariance_increases_transport_standard_error():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = 0.10 + 0.05 * x
    v = np.full_like(y, 0.01)
    fit = fit_meta_regression(y, v, x, effect_scale="RD")

    no_target_uncertainty = transport_effect(fit, np.array([1.5]))
    with_target_uncertainty = transport_effect(
        fit,
        np.array([1.5]),
        target_x_cov=np.array([[4.0]]),
    )

    assert with_target_uncertainty.se > no_target_uncertainty.se


def test_transport_refuses_noncollapsible_odds_ratio_scale():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = np.array([0.2, 0.1, 0.0, -0.1])
    v = np.full_like(y, 0.03)
    fit = fit_meta_regression(y, v, x, effect_scale="logOR")

    with pytest.raises(TransportabilityError, match="non-collapsible"):
        transport_effect(fit, np.array([0.0]))


def test_ratio_scale_with_variable_baseline_risk_warns_for_rd_sensitivity():
    baseline_risk = np.array([0.05, 0.12, 0.35, 0.55, 0.62])
    y = np.array([-0.3, -0.2, -0.1, 0.0, 0.1])
    v = np.full_like(y, 0.02)
    fit = fit_meta_regression(y, v, baseline_risk, effect_scale="logRR")

    result = transport_effect(fit, np.array([0.3]), baseline_risk_idx=0)

    assert any("RD scale" in warning for warning in result.warnings)


def test_unit_square_h1_persistence_is_exact():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    diagram = compute_persistence(square)

    assert diagram.h1.shape == (1, 2)
    assert diagram.h1[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert diagram.h1[0, 1] == pytest.approx(math.sqrt(2.0), abs=1e-9)
    assert bottleneck_distance(diagram.finite(1), diagram.finite(1)) == pytest.approx(0.0)


def test_donut_centre_is_topological_gap_inside_convex_hull():
    modifiers, y, v, truth = _donut_fixture(n=44, seed=1)
    centre = truth["hole_center"]

    assert is_in_convex_hull(modifiers, centre)
    certificate = certify_support(modifiers, centre, n_boot=60, seed=1)

    assert certificate.grade == "GAP"
    assert certificate.topological_gap
    assert any(hole.significant for hole in certificate.holes)


def test_supported_transport_returns_estimate_even_when_support_is_none():
    modifiers, y, v, _ = _donut_fixture(n=44, seed=3)

    estimate, certificate, fit = supported_transport(
        y,
        v,
        modifiers,
        np.array([99.0, 99.0]),
        effect_scale="RD",
        n_boot=30,
        seed=3,
    )

    assert np.isfinite(estimate.estimate)
    assert np.isfinite(fit.beta).all()
    assert certificate.grade == "NONE"
    assert not certificate.supported


def test_two_clusters_are_counted_as_two_subpopulations():
    cluster_a = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    cluster_b = cluster_a + np.array([20.0, 0.0])
    cloud = np.vstack([cluster_a, cluster_b])

    assert count_subpopulations(cloud) == 2
    assert compute_persistence(cloud).n_components_at(1.0) == 2


def _donut_fixture(n=44, seed=1):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = 4.0 + rng.normal(0.0, 0.12, size=n)
    modifiers = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    y = 0.2 + 0.03 * modifiers[:, 0] - 0.02 * modifiers[:, 1]
    variances = np.full(n, 0.03)
    truth = {
        "hole_center": np.array([0.0, 0.0]),
        "supported_point": modifiers[0],
    }
    return modifiers, y, variances, truth
