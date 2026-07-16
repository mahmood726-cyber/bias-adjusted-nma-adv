import pytest

from bias_nma_adv.redescending import (
    RedescendingRobustError,
    tukey_biweight_pairwise_sensitivity,
)


def test_tukey_biweight_sensitivity_downweights_extreme_residual():
    effects = {
        "S1": 0.10,
        "S2": 0.12,
        "S3": 0.11,
        "OUTLIER": 2.00,
    }
    standard_errors = {study_id: 0.10 for study_id in effects}

    robust = tukey_biweight_pairwise_sensitivity(effects, standard_errors)
    naive = sum(effects.values()) / len(effects)

    assert robust.estimate < naive
    assert robust.estimate == pytest.approx(0.11, abs=0.03)
    assert robust.study_weights["OUTLIER"] < 0.01
    assert sum(robust.study_weights.values()) == pytest.approx(1.0)
    assert robust.warning is None


def test_tukey_biweight_sensitivity_preserves_clean_symmetric_signal():
    effects = {
        "S1": 0.10,
        "S2": 0.12,
        "S3": 0.11,
    }
    standard_errors = {study_id: 0.10 for study_id in effects}

    robust = tukey_biweight_pairwise_sensitivity(effects, standard_errors)

    assert robust.estimate == pytest.approx(0.11, abs=0.005)
    assert all(weight > 0.0 for weight in robust.study_weights.values())


@pytest.mark.parametrize(
    ("effects", "standard_errors", "message"),
    [
        ({"S1": 1.0}, {"S1": 0.1}, "at least two"),
        ({"S1": 1.0, "S2": 2.0}, {"S1": 0.1}, "identical study ids"),
        ({"S1": 1.0, "S2": 2.0}, {"S1": 0.1, "S2": 0.0}, "finite and > 0"),
    ],
)
def test_tukey_biweight_sensitivity_rejects_bad_inputs(
    effects,
    standard_errors,
    message,
):
    with pytest.raises(RedescendingRobustError, match=message):
        tukey_biweight_pairwise_sensitivity(effects, standard_errors)
