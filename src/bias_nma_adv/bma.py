"""Bayesian Model Averaging (BMA) solver for NMA network consistency."""

from __future__ import annotations

import numpy as np

class BayesianModelAverager:
    """Averages treatment effects and variances over multiple NMA structural sub-graphs using BIC weights."""

    def __init__(self):
        pass

    def calculate_posterior_probabilities(self, bics: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities of models from their BICs.

        P(M_k | D) = exp(-0.5 * BIC_k) / sum(exp(-0.5 * BIC_j))
        """
        # Subtract min(BIC) to prevent underflow in exponentiation
        shifted_bics = bics - np.min(bics)
        unnormalized = np.exp(-0.5 * shifted_bics)
        return unnormalized / np.sum(unnormalized)

    def average_effects(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        bics: np.ndarray
    ) -> tuple[float, float]:
        """Averages treatment effects and standard errors over models using posterior weights.

        Returns:
         - BMA pooled effect (mean).
         - BMA pooled variance (incorporates model-specification uncertainty).
        """
        post_probs = self.calculate_posterior_probabilities(bics)
        
        # 1. Averaged treatment effect
        bma_effect = float(np.sum(post_probs * effects))
        
        # 2. Averaged variance (law of total variance: E[Var] + Var(E))
        within_model_variance = np.sum(post_probs * variances)
        between_model_variance = np.sum(post_probs * np.square(effects - bma_effect))
        bma_variance = float(within_model_variance + between_model_variance)
        
        return bma_effect, bma_variance
