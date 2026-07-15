"""Copula-Based Joint Likelihood Estimator for Efficacy and Safety outcomes."""

from __future__ import annotations

import numpy as np

class ClaytonCopulaJointEstimator:
    """Clayton Copula estimator linking survival survival probability (u) and safety non-event probability (v)."""

    def __init__(self, theta: float = 1.0):
        self.theta = max(theta, 1e-5) # Clayton parameter must be > 0

    def joint_cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the joint survival probability using Clayton Copula."""
        u_clamped = np.clip(u, 1e-15, 1.0)
        v_clamped = np.clip(v, 1e-15, 1.0)
        
        term = np.power(u_clamped, -self.theta) + np.power(v_clamped, -self.theta) - 1.0
        return np.power(np.maximum(term, 1e-15), -1.0 / self.theta)

    def joint_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute the Clayton copula density."""
        u_clamped = np.clip(u, 1e-15, 1.0)
        v_clamped = np.clip(v, 1e-15, 1.0)

        term1 = 1.0 + self.theta
        term2 = np.power(u_clamped * v_clamped, -(1.0 + self.theta))
        term3 = np.power(u_clamped, -self.theta) + np.power(v_clamped, -self.theta) - 1.0
        term4 = np.power(np.maximum(term3, 1e-15), -(2.0 + 1.0 / self.theta))
        
        return term1 * term2 * term4

    def log_likelihood(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute the log-likelihood of the joint distribution observed across trials."""
        density = self.joint_density(u, v)
        return float(np.sum(np.log(np.maximum(density, 1e-15))))

    def fit(self, u: np.ndarray, v: np.ndarray) -> float:
        """Estimate the copula parameter theta using grid search optimization."""
        best_theta = self.theta
        best_ll = -np.inf
        
        # Grid search over theta
        for candidate in np.linspace(0.1, 5.0, 50):
            self.theta = candidate
            ll = self.log_likelihood(u, v)
            if ll > best_ll:
                best_ll = ll
                best_theta = candidate
                
        self.theta = best_theta
        return best_theta
