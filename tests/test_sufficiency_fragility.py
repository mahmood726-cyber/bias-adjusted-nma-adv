import pytest

from bias_nma_adv.sufficiency_fragility import (
    SufficiencyFragilityError,
    binary_event_fragility_index,
    e_value_for_ci_limit,
    e_value_for_risk_ratio,
)


def test_e_value_for_risk_ratio_is_symmetric_around_null():
    harmful = e_value_for_risk_ratio(2.0)
    protective = e_value_for_risk_ratio(0.5)

    assert harmful == pytest.approx(3.414213562373095)
    assert protective == pytest.approx(harmful)
    assert e_value_for_risk_ratio(1.0) == 1.0


def test_e_value_for_ci_limit_uses_limit_closest_to_null():
    assert e_value_for_ci_limit(2.0, 1.2, 3.4) == pytest.approx(
        e_value_for_risk_ratio(1.2)
    )
    assert e_value_for_ci_limit(0.5, 0.2, 0.8) == pytest.approx(
        e_value_for_risk_ratio(0.8)
    )
    assert e_value_for_ci_limit(0.8, 0.5, 1.2) == 1.0


def test_binary_event_fragility_index_modifies_one_arm_toward_null():
    result = binary_event_fragility_index(5, 100, 20, 100)

    assert result.initial_p_value < 0.05
    assert result.final_p_value >= 0.05
    assert result.fragility_index > 0
    assert result.modified_arm == "treatment"
    assert result.control_events_final == 20
    assert result.treatment_events_final == 5 + result.fragility_index
    assert result.warning is None


def test_binary_event_fragility_index_reports_already_non_significant():
    result = binary_event_fragility_index(10, 100, 12, 100)

    assert result.fragility_index == 0
    assert result.modified_arm == "none"
    assert result.warning == "already_non_significant"


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 0.0, 2.0),
        (1.0, 2.0, 1.0),
        (2.0, 1.0, 1.5),
    ],
)
def test_e_value_for_ci_limit_rejects_invalid_inputs(args):
    with pytest.raises(SufficiencyFragilityError):
        e_value_for_ci_limit(*args)


def test_binary_event_fragility_index_rejects_invalid_counts():
    with pytest.raises(SufficiencyFragilityError, match="cannot exceed"):
        binary_event_fragility_index(101, 100, 20, 100)
    with pytest.raises(SufficiencyFragilityError, match="non-negative integer"):
        binary_event_fragility_index(10.0, 100, 20, 100)
