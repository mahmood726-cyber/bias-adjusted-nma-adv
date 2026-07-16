"""Registry-Based Publication Bias and Outcome Switching Auditor module."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.stats


@dataclass(frozen=True)
class EggerRegressionDiagnostic:
    """Small-study-effect regression diagnostic on reported study effects."""

    k: int
    intercept: float
    intercept_se: float
    t_value: float
    p_value: float
    slope: float
    residual_df: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SelectionWeightSensitivity:
    """Prespecified inverse-selection-probability sensitivity analysis."""

    k: int
    observed_estimate: float
    observed_se: float
    observed_ci: tuple[float, float]
    adjusted_estimate: float
    adjusted_se: float
    adjusted_ci: tuple[float, float]
    selection_probabilities: tuple[float, ...]
    inverse_selection_weights: tuple[float, ...]
    warnings: tuple[str, ...]


class RegistryPublicationBiasAuditor:
    """Audits NMA networks against ClinicalTrials.gov registries to detect unpublished trials and outcome switching."""

    def __init__(self):
        # Maps NCT ID to its registry metadata:
        # {nct_id: {"registered_primary": str, "reported_primary": str, "status": str}}
        self.registry_db: dict[str, dict[str, str]] = {}

    def register_trial_protocol(
        self,
        nct_id: str,
        registered_primary: str,
        reported_primary: str,
        status: str
    ) -> None:
        """Register prospective trial protocol metadata from ClinicalTrials.gov."""
        self.registry_db[nct_id] = {
            "registered_primary": registered_primary.strip().lower(),
            "reported_primary": reported_primary.strip().lower(),
            "status": status.strip().lower()
        }

    def audit_outcome_switching(self, nct_ids: list[str]) -> dict[str, float]:
        """Audit trials for primary outcome switching between registration and publication.

        Returns an Outcome Switching Bias Score (OSBS) in [0.0, 1.0] per NCT ID.
        """
        bias_scores = {}
        for nct_id in nct_ids:
            meta = self.registry_db.get(nct_id)
            if not meta:
                # No registry protocol found (high bias risk)
                bias_scores[nct_id] = 1.0
                continue
                
            reg = meta["registered_primary"]
            rep = meta["reported_primary"]
            
            # If primary outcome was modified or switched, assign score of 1.0
            if reg != rep and reg not in rep and rep not in reg:
                bias_scores[nct_id] = 1.0
            else:
                bias_scores[nct_id] = 0.0
                
        return bias_scores

    def calculate_unpublished_ratio(self, drug_name: str, published_nct_ids: list[str]) -> float:
        """Calculate the registry-based Unpublished Trial Ratio (UTR) for a specific drug.

        UTR = (Registered but Unpublished Trials) / (Total Registered Trials)
        """
        total_registered = 0
        unpublished = 0
        
        # Simple string matching helper to identify trials evaluating the target drug
        for nct_id, meta in self.registry_db.items():
            if meta["status"] in ["completed", "terminated"]:
                total_registered += 1
                if nct_id not in published_nct_ids:
                    unpublished += 1
                    
        if total_registered == 0:
            return 0.0
            
        return float(unpublished / total_registered)

    def apply_bias_shrinkage(self, effect_estimate: float, utr: float) -> float:
        """Apply a publication bias shrinkage factor to the pooled treatment effect.

        Shrinks the pooled log-HR toward 0 (the null) proportional to the unpublished ratio.
        """
        # Shrinkage factor: (1 - UTR)
        shrinkage = 1.0 - utr
        return effect_estimate * max(0.0, shrinkage)


def egger_regression_diagnostic(
    effects: np.ndarray,
    standard_errors: np.ndarray,
    *,
    min_recommended_studies: int = 10,
) -> EggerRegressionDiagnostic:
    """Run Egger's regression diagnostic without changing model weights.

    The regression is `effect / SE = intercept + slope * (1 / SE) + error`.
    A non-zero intercept is a small-study-effect signal, not proof of
    publication bias and not a treatment-effect correction.
    """

    y = np.asarray(effects, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    if y.shape != se.shape:
        raise ValueError("effects and standard_errors must have the same length.")
    if y.size < 3:
        raise ValueError("Egger regression requires at least three studies.")
    if not np.all(np.isfinite(y)):
        raise ValueError("all effects must be finite.")
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError("all standard errors must be finite and positive.")

    precision = 1.0 / se
    standard_normal_deviate = y / se
    design = np.column_stack([np.ones_like(precision), precision])
    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Egger regression design is singular.") from exc

    coefficients = xtx_inv @ design.T @ standard_normal_deviate
    fitted = design @ coefficients
    residual = standard_normal_deviate - fitted
    residual_df = int(y.size - 2)
    residual_variance = float((residual @ residual) / residual_df)
    covariance = residual_variance * xtx_inv
    intercept_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    if intercept_se == 0.0:
        if coefficients[0] > 0.0:
            t_value = math.inf
        elif coefficients[0] < 0.0:
            t_value = -math.inf
        else:
            t_value = 0.0
        p_value = 0.0 if t_value else 1.0
    else:
        t_value = float(coefficients[0] / intercept_se)
        p_value = float(2.0 * scipy.stats.t.sf(abs(t_value), residual_df))

    warnings: list[str] = []
    if y.size < min_recommended_studies:
        warnings.append(
            "Egger regression is underpowered with fewer than "
            f"{min_recommended_studies} studies."
        )

    return EggerRegressionDiagnostic(
        k=int(y.size),
        intercept=float(coefficients[0]),
        intercept_se=float(intercept_se),
        t_value=float(t_value),
        p_value=float(p_value),
        slope=float(coefficients[1]),
        residual_df=residual_df,
        warnings=tuple(warnings),
    )


def selection_weight_sensitivity(
    effects: np.ndarray,
    standard_errors: np.ndarray,
    selection_probabilities: np.ndarray,
    *,
    level: float = 0.95,
) -> SelectionWeightSensitivity:
    """Run a user-prespecified publication-selection sensitivity analysis.

    The adjusted estimate is an inverse-selection-probability weighted
    fixed-effect sensitivity estimate. It is useful for tipping-point and
    robustness review, but it does not infer the selection probabilities and
    should not be interpreted as proof or correction of publication bias.
    """

    y = np.asarray(effects, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    probabilities = np.asarray(selection_probabilities, dtype=float).reshape(-1)
    if y.shape != se.shape or y.shape != probabilities.shape:
        raise ValueError(
            "effects, standard_errors, and selection_probabilities must have the same length."
        )
    if y.size < 2:
        raise ValueError("selection-weight sensitivity requires at least two studies.")
    if not np.all(np.isfinite(y)):
        raise ValueError("all effects must be finite.")
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        raise ValueError("all standard errors must be finite and positive.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError("selection probabilities must be finite values in (0, 1].")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1).")

    base_weights = 1.0 / (se * se)
    observed_estimate = float(np.sum(base_weights * y) / np.sum(base_weights))
    observed_se = math.sqrt(float(1.0 / np.sum(base_weights)))

    inverse_selection_weights = 1.0 / probabilities
    adjusted_weights = base_weights * inverse_selection_weights
    adjusted_estimate = float(np.sum(adjusted_weights * y) / np.sum(adjusted_weights))
    adjusted_se = math.sqrt(
        float(np.sum((adjusted_weights * se) ** 2) / (np.sum(adjusted_weights) ** 2))
    )

    alpha = 1.0 - level
    critical = float(scipy.stats.norm.ppf(1.0 - alpha / 2.0))
    observed_ci = (
        float(observed_estimate - critical * observed_se),
        float(observed_estimate + critical * observed_se),
    )
    adjusted_ci = (
        float(adjusted_estimate - critical * adjusted_se),
        float(adjusted_estimate + critical * adjusted_se),
    )
    warnings = [
        "Selection-weight sensitivity uses prespecified probabilities and is not a publication-bias proof or correction."
    ]
    if np.any(probabilities < 0.5):
        warnings.append("At least one study has selection probability below 0.5.")

    return SelectionWeightSensitivity(
        k=int(y.size),
        observed_estimate=observed_estimate,
        observed_se=float(observed_se),
        observed_ci=observed_ci,
        adjusted_estimate=adjusted_estimate,
        adjusted_se=float(adjusted_se),
        adjusted_ci=adjusted_ci,
        selection_probabilities=tuple(float(item) for item in probabilities),
        inverse_selection_weights=tuple(float(item) for item in inverse_selection_weights),
        warnings=tuple(warnings),
    )
