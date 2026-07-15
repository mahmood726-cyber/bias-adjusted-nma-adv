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
            logits = x @ self.weights + offset_val
            preds = sigmoid(logits)
            gradient = x.T @ (preds - y) / n_samples
            self.weights -= self.lr * gradient

    def predict_proba(self, x: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        offset_val = np.zeros(len(x)) if offset is None else offset
        if self.weights is None:
            return sigmoid(offset_val)
        return sigmoid(x @ self.weights + offset_val)


class CollaborativeTMLE:
    """C-TMLE solver selecting propensity covariates collaboratively based on outcome loss reduction."""

    def __init__(self, max_iter: int = 150, lr: float = 0.05):
        self.max_iter = max_iter
        self.lr = lr

    def estimate_risk_difference(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> float:
        n_samples, n_covariates = covariates.shape
        
        # 1. Fit initial outcome model: Q(A, W)
        outcome_features = np.column_stack([covariates, treatment])
        outcome_model = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
        outcome_model.fit(outcome_features, outcome)
        
        # Predict under observed treatment
        features_obs = np.column_stack([covariates, treatment])
        q_aw = outcome_model.predict_proba(features_obs)
        
        # 2. Collaborative search for propensity score covariates
        best_loss = float("inf")
        best_g_w = np.zeros(n_samples)
        
        # Forward selection step: evaluate adding each covariate to the propensity score model
        for j in range(n_covariates):
            # Candidate features: column j of covariates (plus intercept if needed, already in covariates)
            cand_features = covariates[:, :j+1]
            
            # Fit candidate propensity model
            prop_model = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
            prop_model.fit(cand_features, treatment)
            g_w_cand = np.clip(prop_model.predict_proba(cand_features), 0.025, 0.975)
            
            # Evaluate outcome loss after fluctuation step
            h_aw = treatment / g_w_cand - (1.0 - treatment) / (1.0 - g_w_cand)
            
            fluct_model = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
            fluct_model.fit(h_aw.reshape(-1, 1), outcome, offset=logit(q_aw))
            eps = fluct_model.weights[0] if fluct_model.weights is not None else 0.0
            
            # Predict fluctuated outcome
            q_aw_star = sigmoid(logit(q_aw) + eps * h_aw)
            loss = float(np.mean(np.square(q_aw_star - outcome)))
            
            if loss < best_loss:
                best_loss = loss
                best_g_w = g_w_cand
                
        # 3. Final TMLE update step using the best collaborative propensity score best_g_w
        h_1w = 1.0 / best_g_w
        h_0w = -1.0 / (1.0 - best_g_w)
        h_aw_best = treatment / best_g_w - (1.0 - treatment) / (1.0 - best_g_w)
        
        final_fluct = LogisticRegressionSolver(lr=self.lr, n_iter=self.max_iter)
        final_fluct.fit(h_aw_best.reshape(-1, 1), outcome, offset=logit(q_aw))
        final_eps = final_fluct.weights[0] if final_fluct.weights is not None else 0.0
        
        # Predict counterfactual outcomes
        features_1 = np.column_stack([covariates, np.ones(n_samples)])
        features_0 = np.column_stack([covariates, np.zeros(n_samples)])
        q_1w = outcome_model.predict_proba(features_1)
        q_0w = outcome_model.predict_proba(features_0)
        
        q_1w_star = sigmoid(logit(q_1w) + final_eps * h_1w)
        q_0w_star = sigmoid(logit(q_0w) + final_eps * h_0w)
        
        return float(np.mean(q_1w_star - q_0w_star))
