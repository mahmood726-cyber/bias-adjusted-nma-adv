"""Hamiltonian Monte Carlo (HMC) Sampler using pure NumPy and Leapfrog Integration."""

from __future__ import annotations

import numpy as np

class HamiltonianMonteCarloSampler:
    """HMC sampler utilizing gradient information for efficient parameter space exploration."""

    def __init__(self, step_size: float = 0.05, n_steps: int = 10, seed: int = 42):
        self.step_size = step_size
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        initial_position: np.ndarray,
        log_posterior_fn: callable,
        grad_log_posterior_fn: callable,
        n_samples: int = 100
    ) -> np.ndarray:
        """Draw samples using Hamiltonian dynamics.

        Parameters:
         - log_posterior_fn: Returns ln(P(theta | D))
         - grad_log_posterior_fn: Returns the gradient vector of ln(P(theta | D))
        """
        position = np.asarray(initial_position, dtype=float).copy()
        samples = []

        for _ in range(n_samples):
            # 1. Sample momentum from standard normal
            momentum = self.rng.normal(0.0, 1.0, size=position.shape)
            
            # 2. Compute current Hamiltonian energy
            current_u = -log_posterior_fn(position)
            current_k = 0.5 * np.sum(np.square(momentum))
            
            # 3. Leapfrog integration steps
            new_position = position.copy()
            new_momentum = momentum.copy()
            
            # First half-step for momentum
            grad = -grad_log_posterior_fn(new_position)
            new_momentum -= 0.5 * self.step_size * grad
            
            # Alternate full steps
            for i in range(self.n_steps):
                new_position += self.step_size * new_momentum
                if i < self.n_steps - 1:
                    grad = -grad_log_posterior_fn(new_position)
                    new_momentum -= self.step_size * grad
            
            # Final half-step for momentum
            grad = -grad_log_posterior_fn(new_position)
            new_momentum -= 0.5 * self.step_size * grad
            
            # Negate momentum to make proposal symmetric
            new_momentum = -new_momentum
            
            # 4. Compute proposed Hamiltonian energy
            proposed_u = -log_posterior_fn(new_position)
            proposed_k = 0.5 * np.sum(np.square(new_momentum))
            
            # 5. Metropolis-Hastings acceptance step
            if self.rng.random() < np.exp(current_u + current_k - proposed_u - proposed_k):
                position = new_position
                
            samples.append(position.copy())
            
        return np.array(samples)
