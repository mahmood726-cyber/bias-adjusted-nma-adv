"""Multinomial Logistic GLMM solver for competing-risk network meta-analysis."""

from __future__ import annotations

import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax function over the last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    unnormalized = np.exp(shifted)
    return unnormalized / np.sum(unnormalized, axis=-1, keepdims=True)

class MultinomialGLMMSolver:
    """Fits multinomial logistic regression for multi-category clinical endpoints."""

    def __init__(self, lr: float = 0.05, n_iter: int = 150):
        self.lr = lr
        self.n_iter = n_iter
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit multinomial logistic weights.

        x: shape (n_samples, n_features)
        y: shape (n_samples, n_classes) - one-hot encoded targets
        """
        n_samples, n_features = x.shape
        n_classes = y.shape[1]
        
        # Initialize weights: shape (n_features, n_classes)
        self.weights = np.zeros((n_features, n_classes))

        for _ in range(self.n_iter):
            # Compute probabilities
            logits = x @ self.weights
            probs = softmax(logits)
            
            # Gradient: X^T * (probs - y) / n_samples
            gradient = x.T @ (probs - y) / n_samples
            self.weights -= self.lr * gradient

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return softmax(x @ self.weights)
