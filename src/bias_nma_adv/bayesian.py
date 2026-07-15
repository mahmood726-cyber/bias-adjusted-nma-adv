"""Bayesian NMA MCMC Sampler utilizing Metropolis-Hastings for joint parameter estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

@dataclass
class BayesianMCMCFitResult:
    parameter_names: tuple[str, ...]
    chains: np.ndarray  # Shape: (n_samples, n_params + n_designs)
    posterior_means: dict[str, float]
    posterior_sds: dict[str, float]
    credible_intervals: dict[str, tuple[float, float]]
    acceptance_rate: float


class BayesianNMAMCMCSampler:
    """Self-contained Bayesian MCMC sampler for advanced bias-adjusted NMA."""

    def __init__(
        self,
        n_samples: int = 5000,
        burn_in: int = 1000,
        thinning: int = 1,
        proposal_sd_beta: float = 0.05,
        proposal_sd_tau: float = 0.02
    ):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.proposal_sd_beta = proposal_sd_beta
        self.proposal_sd_tau = proposal_sd_tau

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        param_names: tuple[str, ...],
        blocks: list,
        unique_designs: list[str],
        design_to_idx: dict[str, int],
        bias_prior_sd: float = 1.0,
        treatment_shrinkage_lambda: float = 0.0,
        treatment_centralities: dict[str, float] | None = None,
        bias_prior_mean: float = 0.0,
        seed: int = 42
    ) -> BayesianMCMCFitResult:
        rng = np.random.default_rng(seed)
        n_params = x.shape[1]
        n_designs = len(unique_designs)
        n_vars = n_params + n_designs

        # Precompute prior precision matrix P and prior mean vector
        p_matrix = np.zeros((n_params, n_params), dtype=float)
        mu_vector = np.zeros(n_params, dtype=float)
        study_rob = {b.study_id: b.rob_weight for b in blocks}
        centralities = treatment_centralities or {}

        for idx, name in enumerate(param_names):
            if name.startswith("bias_study_"):
                s_id = name[11:]
                rob = study_rob.get(s_id, 1.0)
                p_matrix[idx, idx] = 1.0 / (((bias_prior_sd * bias_prior_sd) * (1.0 - rob)) + 1e-6)
                mu_vector[idx] = bias_prior_mean
            elif name.startswith("bias_") or "_x_" in name:
                p_matrix[idx, idx] = 1.0 / (bias_prior_sd * bias_prior_sd)
                if name.startswith("bias_"):
                    mu_vector[idx] = bias_prior_mean
            elif name.startswith("trt_") and "_x_" not in name:
                trt = name[4:]
                p_matrix[idx, idx] = treatment_shrinkage_lambda * (1.0 - centralities.get(trt, 1.0))

        # Initial values
        beta = np.zeros(n_params, dtype=float)
        taus = np.full(n_designs, 0.1, dtype=float)
        
        current_state = np.concatenate([beta, taus])
        current_log_post = self._log_posterior(
            beta, taus, y, x, v, blocks, design_to_idx, p_matrix, mu_vector
        )

        samples = []
        accepted = 0

        total_iterations = self.burn_in + self.n_samples * self.thinning

        for i in range(total_iterations):
            # Propose new state
            proposal_noise = rng.normal(0.0, 1.0, size=n_vars)
            proposal_noise[:n_params] *= self.proposal_sd_beta
            proposal_noise[n_params:] *= self.proposal_sd_tau

            proposed_state = current_state + proposal_noise
            proposed_beta = proposed_state[:n_params]
            proposed_taus = proposed_state[n_params:]

            # Reject negative heterogeneities immediately
            if np.any(proposed_taus < 0.0):
                log_alpha = -float("inf")
            else:
                proposed_log_post = self._log_posterior(
                    proposed_beta, proposed_taus, y, x, v, blocks, design_to_idx, p_matrix, mu_vector
                )
                log_alpha = proposed_log_post - current_log_post

            # Accept/Reject step
            if math.log(rng.uniform(0.0, 1.0)) < log_alpha:
                current_state = proposed_state
                current_log_post = proposed_log_post
                if i >= self.burn_in:
                    accepted += 1

            if i >= self.burn_in and (i - self.burn_in) % self.thinning == 0:
                samples.append(current_state.copy())

        samples = np.array(samples)
        acceptance_rate = accepted / max(total_iterations - self.burn_in, 1)

        # Compute summary statistics
        posterior_means = {}
        posterior_sds = {}
        credible_intervals = {}

        all_names = list(param_names) + [f"tau_{d}" for d in unique_designs]

        for idx, name in enumerate(all_names):
            chain = samples[:, idx]
            posterior_means[name] = float(np.mean(chain))
            posterior_sds[name] = float(np.std(chain))
            credible_intervals[name] = (
                float(np.percentile(chain, 2.5)),
                float(np.percentile(chain, 97.5))
            )

        return BayesianMCMCFitResult(
            parameter_names=tuple(all_names),
            chains=samples,
            posterior_means=posterior_means,
            posterior_sds=posterior_sds,
            credible_intervals=credible_intervals,
            acceptance_rate=acceptance_rate
        )

    def _log_posterior(
        self,
        beta: np.ndarray,
        taus: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        blocks: list,
        design_to_idx: dict[str, int],
        p_matrix: np.ndarray,
        mu_vector: np.ndarray
    ) -> float:
        # Assemble covariance matrix M
        m = v.copy()
        cursor = 0
        for block in blocks:
            size = block.y.shape[0]
            d_idx = design_to_idx.get(block.design, 0)
            tau = taus[d_idx]
            m[cursor : cursor + size, cursor : cursor + size] += np.eye(size, dtype=float) * (tau * tau)
            cursor += size

        try:
            m_inv = np.linalg.inv(m)
        except np.linalg.LinAlgError:
            return -float("inf")

        sign, logdet_m = np.linalg.slogdet(m)
        if sign <= 0:
            return -float("inf")

        # Likelihood
        resid = y - x @ beta
        log_lik = -0.5 * logdet_m - 0.5 * resid.T @ m_inv @ resid

        # Prior log probability on Beta
        diff = beta - mu_vector
        log_prior_beta = -0.5 * diff.T @ p_matrix @ diff

        # Prior log probability on Taus (Half-Normal, sd = 1.0)
        log_prior_tau = -0.5 * np.sum(taus * taus)

        return log_lik + log_prior_beta + log_prior_tau
