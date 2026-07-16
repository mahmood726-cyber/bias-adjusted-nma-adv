"""Redescending robust sensitivity screens for pairwise meta-analysis.

This module is deliberately a sensitivity screen, not a replacement estimator
for production NMA. It downweights large standardized residuals with Tukey's
biweight function and reports the induced study weights transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


class RedescendingRobustError(ValueError):
    """Raised when redescending robust inputs are malformed."""


@dataclass(frozen=True)
class RedescendingRobustResult:
    """Result from a Tukey-biweight inverse-variance sensitivity fit."""

    estimate: float
    standard_error: float
    iterations: int
    tuning_constant: float
    study_weights: dict[str, float]
    warning: str | None = None


def tukey_biweight_pairwise_sensitivity(
    effects: Mapping[str, float],
    standard_errors: Mapping[str, float],
    *,
    tuning_constant: float = 4.685,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> RedescendingRobustResult:
    """Fit a deterministic redescending sensitivity estimate for one contrast."""

    study_ids, y, se = _validated_arrays(effects, standard_errors)
    if tuning_constant <= 0.0 or not math.isfinite(tuning_constant):
        raise RedescendingRobustError("tuning_constant must be finite and > 0.")
    if max_iter < 1:
        raise RedescendingRobustError("max_iter must be >= 1.")
    if tol <= 0.0 or not math.isfinite(tol):
        raise RedescendingRobustError("tol must be finite and > 0.")

    inverse_variance = 1.0 / np.square(se)
    estimate = float(np.sum(inverse_variance * y) / np.sum(inverse_variance))
    final_weights = inverse_variance.copy()
    warning: str | None = None

    iterations = 0
    for iterations in range(1, max_iter + 1):
        standardized = (y - estimate) / se
        robust_weights = _tukey_weights(standardized, tuning_constant)
        final_weights = inverse_variance * robust_weights
        if float(np.sum(final_weights)) <= 0.0:
            warning = "all_studies_redescended_to_zero_weight"
            final_weights = inverse_variance
            break
        updated = float(np.sum(final_weights * y) / np.sum(final_weights))
        if abs(updated - estimate) <= tol:
            estimate = updated
            break
        estimate = updated
    else:
        warning = "maximum_iterations_reached"

    total_weight = float(np.sum(final_weights))
    standard_error = math.sqrt(1.0 / total_weight)
    normalized_weights = {
        study_id: float(weight / total_weight)
        for study_id, weight in zip(study_ids, final_weights, strict=True)
    }
    return RedescendingRobustResult(
        estimate=estimate,
        standard_error=standard_error,
        iterations=iterations,
        tuning_constant=tuning_constant,
        study_weights=normalized_weights,
        warning=warning,
    )


def _tukey_weights(standardized_residuals: np.ndarray, tuning_constant: float) -> np.ndarray:
    scaled = standardized_residuals / tuning_constant
    weights = np.square(1.0 - np.square(scaled))
    weights[np.abs(scaled) >= 1.0] = 0.0
    return weights


def _validated_arrays(
    effects: Mapping[str, float],
    standard_errors: Mapping[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if set(effects) != set(standard_errors):
        raise RedescendingRobustError("effects and standard_errors must have identical study ids.")
    if len(effects) < 2:
        raise RedescendingRobustError("at least two studies are required.")

    study_ids = list(effects)
    y = np.array([float(effects[study_id]) for study_id in study_ids], dtype=float)
    se = np.array([float(standard_errors[study_id]) for study_id in study_ids], dtype=float)
    if not np.all(np.isfinite(y)):
        raise RedescendingRobustError("effects must be finite.")
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        raise RedescendingRobustError("standard_errors must be finite and > 0.")
    return study_ids, y, se
