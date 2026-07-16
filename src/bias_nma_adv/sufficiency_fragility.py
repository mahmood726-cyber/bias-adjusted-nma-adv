"""Sufficiency and fragility sensitivity summaries.

These helpers do not create clinical evidence. They transform source-backed
ratio estimates or event counts into transparent sensitivity quantities that
can be reported beside the primary estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.stats import fisher_exact


class SufficiencyFragilityError(ValueError):
    """Raised when sensitivity inputs are impossible or malformed."""


@dataclass(frozen=True)
class FragilityIndexResult:
    """One-arm binary event fragility-index result."""

    fragility_index: int
    initial_p_value: float
    final_p_value: float
    modified_arm: str
    treatment_events_final: int
    control_events_final: int
    warning: str | None = None


def e_value_for_risk_ratio(ratio: float) -> float:
    """Compute VanderWeele-Ding E-value for a risk-ratio-like effect estimate."""

    ratio = _positive_finite("ratio", ratio)
    harmful_ratio = ratio if ratio >= 1.0 else 1.0 / ratio
    if harmful_ratio == 1.0:
        return 1.0
    return harmful_ratio + math.sqrt(harmful_ratio * (harmful_ratio - 1.0))


def e_value_for_ci_limit(ratio: float, ci_lower: float, ci_upper: float) -> float:
    """Compute the E-value for the confidence limit closest to the null."""

    ratio = _positive_finite("ratio", ratio)
    ci_lower = _positive_finite("ci_lower", ci_lower)
    ci_upper = _positive_finite("ci_upper", ci_upper)
    if ci_lower > ci_upper:
        raise SufficiencyFragilityError("ci_lower must be <= ci_upper.")
    if not ci_lower <= ratio <= ci_upper:
        raise SufficiencyFragilityError("ratio must lie within the confidence interval.")

    if ratio == 1.0:
        return 1.0
    if ratio > 1.0:
        if ci_lower <= 1.0:
            return 1.0
        return e_value_for_risk_ratio(ci_lower)
    if ci_upper >= 1.0:
        return 1.0
    return e_value_for_risk_ratio(ci_upper)


def binary_event_fragility_index(
    treatment_events: int,
    treatment_n: int,
    control_events: int,
    control_n: int,
    *,
    alpha: float = 0.05,
) -> FragilityIndexResult:
    """Flip one arm toward the null until Fisher's exact test is non-significant."""

    treatment_events = _event_count("treatment_events", treatment_events)
    treatment_n = _event_count("treatment_n", treatment_n)
    control_events = _event_count("control_events", control_events)
    control_n = _event_count("control_n", control_n)
    if not 0.0 < alpha < 1.0:
        raise SufficiencyFragilityError("alpha must be in (0, 1).")
    if treatment_events > treatment_n:
        raise SufficiencyFragilityError("treatment_events cannot exceed treatment_n.")
    if control_events > control_n:
        raise SufficiencyFragilityError("control_events cannot exceed control_n.")

    current_treatment_events = treatment_events
    current_control_events = control_events
    initial_p = _two_sided_fisher_p(
        current_treatment_events,
        treatment_n,
        current_control_events,
        control_n,
    )
    if initial_p >= alpha:
        return FragilityIndexResult(
            fragility_index=0,
            initial_p_value=initial_p,
            final_p_value=initial_p,
            modified_arm="none",
            treatment_events_final=current_treatment_events,
            control_events_final=current_control_events,
            warning="already_non_significant",
        )

    treatment_rate = treatment_events / treatment_n
    control_rate = control_events / control_n
    if treatment_rate < control_rate:
        modified_arm = "treatment"
        max_flips = treatment_n - treatment_events
    elif treatment_rate > control_rate:
        modified_arm = "control"
        max_flips = control_n - control_events
    else:
        return FragilityIndexResult(
            fragility_index=0,
            initial_p_value=initial_p,
            final_p_value=initial_p,
            modified_arm="none",
            treatment_events_final=current_treatment_events,
            control_events_final=current_control_events,
            warning="equal_event_rates",
        )

    final_p = initial_p
    for flips in range(1, max_flips + 1):
        if modified_arm == "treatment":
            current_treatment_events = treatment_events + flips
        else:
            current_control_events = control_events + flips
        final_p = _two_sided_fisher_p(
            current_treatment_events,
            treatment_n,
            current_control_events,
            control_n,
        )
        if final_p >= alpha:
            return FragilityIndexResult(
                fragility_index=flips,
                initial_p_value=initial_p,
                final_p_value=final_p,
                modified_arm=modified_arm,
                treatment_events_final=current_treatment_events,
                control_events_final=current_control_events,
            )

    return FragilityIndexResult(
        fragility_index=max_flips,
        initial_p_value=initial_p,
        final_p_value=final_p,
        modified_arm=modified_arm,
        treatment_events_final=current_treatment_events,
        control_events_final=current_control_events,
        warning="exhausted_available_non_events_before_alpha",
    )


def _two_sided_fisher_p(
    treatment_events: int,
    treatment_n: int,
    control_events: int,
    control_n: int,
) -> float:
    table = [
        [treatment_events, treatment_n - treatment_events],
        [control_events, control_n - control_events],
    ]
    result = fisher_exact(table, alternative="two-sided")
    return float(result.pvalue if hasattr(result, "pvalue") else result[1])


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SufficiencyFragilityError(f"{name} must be finite and > 0.")
    return value


def _event_count(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise SufficiencyFragilityError(f"{name} must be a non-negative integer.")
    if not isinstance(value, int):
        raise SufficiencyFragilityError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise SufficiencyFragilityError(f"{name} must be a non-negative integer.")
    return value
