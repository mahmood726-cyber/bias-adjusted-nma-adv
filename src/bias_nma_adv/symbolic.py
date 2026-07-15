"""Symbolic Regression module for discovering optimal non-proportional hazard functions."""

from __future__ import annotations

import numpy as np

class SymbolicHazardRegressor:
    """Discovers closed-form mathematical functions for time-varying hazards using operator-space search."""

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        # Standard candidate basis functions: f(t)
        self.basis_functions = [
            lambda t: np.ones_like(t),
            lambda t: t,
            lambda t: np.sqrt(np.maximum(t, 1e-6)),
            lambda t: np.log(np.maximum(t, 1e-6)),
            lambda t: np.exp(-np.clip(t, 0.0, 10.0))
        ]
        self.basis_names = ["1", "t", "sqrt(t)", "ln(t)", "exp(-t)"]

    def fit_best_formula(self, times: np.ndarray, hazard_rates: np.ndarray) -> tuple[str, np.ndarray, float]:
        """Search the basis function combinations to find the formula minimizing MSE.

        Returns:
         - The formula string representation.
         - The fitted coefficient vector.
         - The minimum Mean Squared Error (MSE).
         """
        best_mse = float("inf")
        best_formula = "None"
        best_coefs = np.zeros(1)
        
        n_basis = len(self.basis_functions)
        
        # Search all pairs of basis functions: f(t) = c1 * b_i(t) + c2 * b_j(t)
        for i in range(n_basis):
            for j in range(i, n_basis):
                # Design matrix
                col1 = self.basis_functions[i](times)
                col2 = self.basis_functions[j](times)
                x = np.column_stack([col1, col2])
                
                # Solve ordinary least squares: (X^T X)^{-1} X^T y
                try:
                    coefs = np.linalg.pinv(x.T @ x) @ x.T @ hazard_rates
                    preds = x @ coefs
                    mse = float(np.mean(np.square(preds - hazard_rates)))
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_coefs = coefs
                        best_formula = f"{coefs[0]:.4f} * {self.basis_names[i]} + {coefs[1]:.4f} * {self.basis_names[j]}"
                except np.linalg.LinAlgError:
                    continue
                    
        return best_formula, best_coefs, best_mse
