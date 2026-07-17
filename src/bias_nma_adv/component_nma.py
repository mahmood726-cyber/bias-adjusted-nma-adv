"""Additive component network meta-analysis utilities.

This module intentionally implements the narrow estimability core needed for
component-NMA reference fixtures. It is not a full replacement for netmeta's
CNMA workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


COMPONENT_NMA_SCHEMA_VERSION = "component_nma_fit/v1"


class ComponentNMAError(ValueError):
    """Raised when a component-NMA fixture is malformed or not estimable."""


@dataclass(frozen=True)
class ComponentContrast:
    """One observed contrast for an additive component model."""

    study_id: str
    treat1: str
    treat2: str
    estimate: float
    se: float

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ComponentContrast":
        required = {"study_id", "treat1", "treat2", "estimate", "se"}
        missing = sorted(required - set(raw))
        if missing:
            raise ComponentNMAError(f"component contrast missing required keys: {missing}")
        contrast = cls(
            study_id=str(raw["study_id"]),
            treat1=str(raw["treat1"]),
            treat2=str(raw["treat2"]),
            estimate=float(raw["estimate"]),
            se=float(raw["se"]),
        )
        contrast.validate()
        return contrast

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ComponentNMAError("study_id must not be empty.")
        if not self.treat1.strip() or not self.treat2.strip():
            raise ComponentNMAError(f"{self.study_id}: treatments must not be empty.")
        if self.treat1 == self.treat2:
            raise ComponentNMAError(f"{self.study_id}: contrast compares a treatment with itself.")
        if not math.isfinite(self.estimate):
            raise ComponentNMAError(f"{self.study_id}: estimate must be finite.")
        if not math.isfinite(self.se) or self.se <= 0.0:
            raise ComponentNMAError(f"{self.study_id}: se must be positive and finite.")


@dataclass(frozen=True)
class ComponentEffect:
    """One fitted component or treatment-combination effect."""

    name: str
    estimate: float
    se: float
    estimable: bool


@dataclass(frozen=True)
class ComponentNMAFit:
    """Fitted additive component model."""

    schema_version: str
    inactive_treatment: str
    components: tuple[str, ...]
    rank: int
    df: int
    q: float
    component_effects: tuple[ComponentEffect, ...]
    treatment_effects: tuple[ComponentEffect, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/TOML-friendly representation."""

        return {
            "schema_version": self.schema_version,
            "inactive_treatment": self.inactive_treatment,
            "components": list(self.components),
            "rank": self.rank,
            "df": self.df,
            "q": self.q,
            "component_effects": [
                {
                    "name": item.name,
                    "estimate": item.estimate,
                    "se": item.se,
                    "estimable": item.estimable,
                }
                for item in self.component_effects
            ],
            "treatment_effects": [
                {
                    "name": item.name,
                    "estimate": item.estimate,
                    "se": item.se,
                    "estimable": item.estimable,
                }
                for item in self.treatment_effects
            ],
            "warnings": list(self.warnings),
        }


def fit_additive_component_nma(
    contrasts: list[ComponentContrast | dict[str, Any]],
    *,
    inactive_treatment: str = "Placebo",
    component_separator: str = "+",
    tolerance: float = 1e-10,
) -> ComponentNMAFit:
    """Fit a fixed-effect additive component model by weighted least squares.

    Each row is interpreted as ``estimate = effect(treat1) - effect(treat2)``.
    Treatments are decomposed into components by splitting on
    ``component_separator`` except for the inactive treatment, which has no
    active components.
    """

    parsed = [
        item if isinstance(item, ComponentContrast) else ComponentContrast.from_mapping(item)
        for item in contrasts
    ]
    if not parsed:
        raise ComponentNMAError("at least one component contrast is required.")
    inactive = inactive_treatment.strip()
    if not inactive:
        raise ComponentNMAError("inactive_treatment must not be empty.")
    if not component_separator.strip():
        raise ComponentNMAError("component_separator must not be empty.")
    if tolerance <= 0.0:
        raise ComponentNMAError("tolerance must be positive.")

    components = _component_order(parsed, inactive, component_separator)
    if not components:
        raise ComponentNMAError("at least one active component is required.")

    design = np.vstack(
        [
            _component_row(
                contrast.treat1,
                contrast.treat2,
                components,
                inactive,
                component_separator,
            )
            for contrast in parsed
        ]
    )
    y = np.array([contrast.estimate for contrast in parsed], dtype=float)
    variance = np.square(np.array([contrast.se for contrast in parsed], dtype=float))
    weights = 1.0 / variance
    sqrt_w = np.sqrt(weights)
    xw = design * sqrt_w[:, None]
    yw = y * sqrt_w
    xtwx = xw.T @ xw
    xtwy = xw.T @ yw
    rank = int(np.linalg.matrix_rank(xw, tol=tolerance))
    if rank == design.shape[1]:
        covariance = np.linalg.inv(xtwx)
        beta = covariance @ xtwy
    else:
        covariance = np.linalg.pinv(xtwx, rcond=tolerance)
        beta = covariance @ xtwy

    fitted = design @ beta
    residuals = y - fitted
    q = float(np.sum(weights * np.square(residuals)))
    df = max(0, len(parsed) - rank)
    projection = np.linalg.pinv(design, rcond=tolerance) @ design
    component_effects = tuple(
        ComponentEffect(
            name=component,
            estimate=float(beta[index]),
            se=float(math.sqrt(max(covariance[index, index], 0.0))),
            estimable=bool(abs(projection[index, index] - 1.0) <= 1e-7),
        )
        for index, component in enumerate(components)
    )
    treatment_effects = tuple(
        _treatment_effect(
            treatment,
            beta=beta,
            covariance=covariance,
            estimability_projection=projection,
            components=components,
            inactive_treatment=inactive,
            component_separator=component_separator,
        )
        for treatment in _treatment_order(parsed)
    )

    warnings: list[str] = []
    if rank < len(components):
        warnings.append(
            "Component design matrix is rank deficient; non-estimable effects use a generalized inverse."
        )
    if df == 0:
        warnings.append("No residual degrees of freedom remain for additive-model fit diagnostics.")

    return ComponentNMAFit(
        schema_version=COMPONENT_NMA_SCHEMA_VERSION,
        inactive_treatment=inactive,
        components=components,
        rank=rank,
        df=df,
        q=q,
        component_effects=component_effects,
        treatment_effects=treatment_effects,
        warnings=tuple(warnings),
    )


def _component_order(
    contrasts: list[ComponentContrast],
    inactive_treatment: str,
    component_separator: str,
) -> tuple[str, ...]:
    seen: list[str] = []
    for contrast in contrasts:
        for treatment in (contrast.treat1, contrast.treat2):
            for component in _components_for_treatment(
                treatment,
                inactive_treatment,
                component_separator,
            ):
                if component not in seen:
                    seen.append(component)
    return tuple(seen)


def _treatment_order(contrasts: list[ComponentContrast]) -> tuple[str, ...]:
    seen: list[str] = []
    for contrast in contrasts:
        for treatment in (contrast.treat1, contrast.treat2):
            if treatment not in seen:
                seen.append(treatment)
    return tuple(seen)


def _component_row(
    treat1: str,
    treat2: str,
    components: tuple[str, ...],
    inactive_treatment: str,
    component_separator: str,
) -> np.ndarray:
    left = set(_components_for_treatment(treat1, inactive_treatment, component_separator))
    right = set(_components_for_treatment(treat2, inactive_treatment, component_separator))
    return np.array(
        [float(component in left) - float(component in right) for component in components],
        dtype=float,
    )


def _components_for_treatment(
    treatment: str,
    inactive_treatment: str,
    component_separator: str,
) -> tuple[str, ...]:
    if treatment == inactive_treatment:
        return ()
    parts = tuple(part.strip() for part in treatment.split(component_separator))
    if not parts or any(not part for part in parts):
        raise ComponentNMAError(f"invalid component treatment label {treatment!r}.")
    return parts


def _treatment_effect(
    treatment: str,
    *,
    beta: np.ndarray,
    covariance: np.ndarray,
    estimability_projection: np.ndarray,
    components: tuple[str, ...],
    inactive_treatment: str,
    component_separator: str,
) -> ComponentEffect:
    row = _component_row(
        treatment,
        inactive_treatment,
        components,
        inactive_treatment,
        component_separator,
    )
    variance = float(row @ covariance @ row)
    return ComponentEffect(
        name=treatment,
        estimate=float(row @ beta),
        se=float(math.sqrt(max(variance, 0.0))),
        estimable=bool(np.allclose(row, row @ estimability_projection, atol=1e-7)),
    )
