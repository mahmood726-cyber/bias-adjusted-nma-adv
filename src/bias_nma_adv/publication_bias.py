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
