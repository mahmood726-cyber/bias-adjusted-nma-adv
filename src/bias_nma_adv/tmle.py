"""Targeted Minimum Loss-Based Estimation (TMLE) double-robust causal inference module."""

from __future__ import annotations

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def logit(p: np.ndarray) -> np.ndarray:
    """Stable logit function."""
    p_clamped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.log(p_clamped / (1.0 - p_clamped))

class LogisticRegressionSolver:
    """Simple NumPy-based logistic regression solver using gradient descent."""

    def __init__(self, lr: float = 0.1, n_iter: int = 100):
        self.lr = lr
        self.n_iter = n_iter
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> None:
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        
        offset_val = np.zeros(n_samples) if offset is None else offset

        for _ in range(self.n_iter):
            # Calculate predictions on logit scale
            logits = x @ self.weights + offset_val
            preds = sigmoid(logits)
            
            # Gradient descent step
            gradient = x.T @ (preds - y) / n_samples
            self.weights -= self.lr * gradient

    def predict_proba(self, x: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        offset_val = np.zeros(len(x)) if offset is None else offset
        if self.weights is None:
            return sigmoid(offset_val)
        return sigmoid(x @ self.weights + offset_val)


class DoubleRobustTMLE:
    """Targeted Minimum Loss-Based Estimation (TMLE) for Marginal Risk Differences."""

    def __init__(self, max_iter: int = 200, lr: float = 0.1):
        self.max_iter = max_iter
        self.lr = lr

    def estimate_risk_difference(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> float:
        """Calculate the double-robust TMLE Marginal Risk Difference E(Y|A=1) - E(Y|A=0)."""
        n_samples = len(treatment)
        
        # 1. Fit Propensity Score Model: g(W) = P(A=1 | W)
        prop_model = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
        prop_model.fit(covariates, treatment)
        g_w = np.clip(prop_model.predict_proba(covariates), 0.025, 0.975) # Truncate propensity scores

        # 2. Fit Initial Outcome Model: Q(A, W) = E(Y | A, W)
        # Outcome features: columns of covariates + treatment indicator A
        outcome_features = np.column_stack([covariates, treatment])
        outcome_model = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
        outcome_model.fit(outcome_features, outcome)

        # 3. Predict counterfactual outcomes under A=1 and A=0
        features_1 = np.column_stack([covariates, np.ones(n_samples)])
        features_0 = np.column_stack([covariates, np.zeros(n_samples)])
        
        q_1w = outcome_model.predict_proba(features_1)
        q_0w = outcome_model.predict_proba(features_0)
        q_aw = np.where(treatment == 1, q_1w, q_0w)

        # 4. Compute Clever Covariate H(A, W)
        h_aw = treatment / g_w - (1.0 - treatment) / (1.0 - g_w)
        h_1w = 1.0 / g_w
        h_0w = -1.0 / (1.0 - g_w)

        # 5. TMLE Fluctuation Step (Update step)
        # Fit logistic regression on Y using logit(Q(A,W)) as offset and H(A,W) as covariate without intercept
        fluct_model = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
        # x is clever covariate (n_samples x 1), offset is logit(q_aw)
        x_fluct = h_aw.reshape(-1, 1)
        offset_fluct = logit(q_aw)
        fluct_model.fit(x_fluct, outcome, offset=offset_fluct)
        epsilon = fluct_model.weights[0] if fluct_model.weights is not None else 0.0

        # 6. Update counterfactual predictions
        q_1w_star = sigmoid(logit(q_1w) + epsilon * h_1w)
        q_0w_star = sigmoid(logit(q_0w) + epsilon * h_0w)

        # 7. Marginal Risk Difference
        risk_diff = np.mean(q_1w_star - q_0w_star)
        return float(risk_diff)
