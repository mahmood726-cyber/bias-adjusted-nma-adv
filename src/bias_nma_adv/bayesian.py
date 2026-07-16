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
    diagnostics: dict[str, "MCMCDiagnostic"] = field(default_factory=dict)
    diagnostic_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCMCDiagnostic:
    """Convergence and Monte Carlo error diagnostic for one parameter."""

    parameter: str
    n_chains: int
    n_draws: int
    r_hat: float | None
    ess_bulk: float
    ess_tail: float
    mcse_mean: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BayesianPredictiveCheck:
    """Prior or posterior predictive summary for aggregate model checks."""

    check_type: str
    n_draws: int
    n_observations: int
    observed_mean: float | None
    predictive_mean: float
    predictive_interval: tuple[float, float]
    bayesian_p_value: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PosteriorTreatmentRanking:
    """Draw-preserving posterior treatment ranking summary."""

    treatments: tuple[str, ...]
    beneficial_direction: str
    n_draws: int
    rank_probabilities: dict[str, tuple[float, ...]]
    mean_ranks: dict[str, float]
    warnings: tuple[str, ...]


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

        diagnostics = compute_mcmc_diagnostics(samples, tuple(all_names))
        diagnostic_warnings: list[str] = []
        if acceptance_rate < 0.15 or acceptance_rate > 0.95:
            diagnostic_warnings.append(
                "Metropolis-Hastings acceptance rate is outside the broad diagnostic range [0.15, 0.95]."
            )
        for diagnostic in diagnostics.values():
            diagnostic_warnings.extend(
                f"{diagnostic.parameter}: {warning}" for warning in diagnostic.warnings
            )

        return BayesianMCMCFitResult(
            parameter_names=tuple(all_names),
            chains=samples,
            posterior_means=posterior_means,
            posterior_sds=posterior_sds,
            credible_intervals=credible_intervals,
            acceptance_rate=acceptance_rate,
            diagnostics=diagnostics,
            diagnostic_warnings=tuple(diagnostic_warnings),
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


def compute_mcmc_diagnostics(
    draws: np.ndarray,
    parameter_names: tuple[str, ...],
) -> dict[str, MCMCDiagnostic]:
    """Compute deterministic MCMC diagnostics for 2D or 3D posterior draws.

    ``draws`` may be shaped as ``(draw, parameter)`` for a single chain or as
    ``(chain, draw, parameter)`` for multiple chains. R-hat is intentionally
    unavailable for single-chain output.
    """

    array = np.asarray(draws, dtype=float)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError("draws must have shape (draw, parameter) or (chain, draw, parameter).")
    n_chains, n_draws, n_parameters = array.shape
    if n_parameters != len(parameter_names):
        raise ValueError("parameter_names length must match the draws parameter dimension.")
    if n_draws < 4:
        raise ValueError("at least four draws are required for MCMC diagnostics.")
    if not np.all(np.isfinite(array)):
        raise ValueError("MCMC draws must be finite.")

    diagnostics: dict[str, MCMCDiagnostic] = {}
    for idx, name in enumerate(parameter_names):
        parameter_draws = array[:, :, idx]
        flattened = parameter_draws.reshape(-1)
        ess = _effective_sample_size(flattened)
        tail_ess = min(
            _effective_sample_size(flattened[flattened <= np.percentile(flattened, 25.0)]),
            _effective_sample_size(flattened[flattened >= np.percentile(flattened, 75.0)]),
        )
        sd = float(np.std(flattened, ddof=1)) if flattened.size > 1 else 0.0
        mcse = float(sd / math.sqrt(max(ess, 1.0)))
        r_hat = _split_r_hat(parameter_draws) if n_chains >= 2 else None
        warnings: list[str] = []
        if r_hat is None:
            warnings.append("R-hat unavailable: at least two chains are required.")
        elif r_hat > 1.01:
            warnings.append(f"R-hat exceeds 1.01 ({r_hat:.4f}).")
        if ess < 400.0:
            warnings.append(f"Bulk ESS below 400 ({ess:.1f}).")
        if tail_ess < 400.0:
            warnings.append(f"Tail ESS below 400 ({tail_ess:.1f}).")
        diagnostics[name] = MCMCDiagnostic(
            parameter=name,
            n_chains=int(n_chains),
            n_draws=int(n_draws),
            r_hat=r_hat,
            ess_bulk=float(ess),
            ess_tail=float(tail_ess),
            mcse_mean=mcse,
            warnings=tuple(warnings),
        )
    return diagnostics


def prior_predictive_check(
    x: np.ndarray,
    v: np.ndarray,
    *,
    n_draws: int = 1000,
    beta_sd: float = 1.0,
    seed: int = 123,
) -> BayesianPredictiveCheck:
    """Simulate a simple prior predictive distribution for the linear predictor."""

    design = np.asarray(x, dtype=float)
    variances = np.asarray(v, dtype=float)
    if design.ndim != 2:
        raise ValueError("x must be a 2D design matrix.")
    if variances.shape != (design.shape[0], design.shape[0]):
        raise ValueError("v must be a square covariance matrix matching x rows.")
    if n_draws < 10:
        raise ValueError("n_draws must be at least 10.")
    if beta_sd <= 0.0:
        raise ValueError("beta_sd must be positive.")
    if not np.all(np.isfinite(design)):
        raise ValueError("x must be finite.")
    if not np.all(np.isfinite(variances)):
        raise ValueError("v must be finite.")

    rng = np.random.default_rng(seed)
    beta_draws = rng.normal(0.0, beta_sd, size=(int(n_draws), design.shape[1]))
    means = beta_draws @ design.T
    predictive = _draw_multivariate_normal_rows(rng, means, variances)
    predictive_means = np.mean(predictive, axis=1)
    warnings = (
        "Prior predictive check is a local prototype simulation and is not Stan/multinma parity.",
    )
    return BayesianPredictiveCheck(
        check_type="prior_predictive",
        n_draws=int(n_draws),
        n_observations=int(design.shape[0]),
        observed_mean=None,
        predictive_mean=float(np.mean(predictive_means)),
        predictive_interval=(
            float(np.percentile(predictive_means, 2.5)),
            float(np.percentile(predictive_means, 97.5)),
        ),
        bayesian_p_value=None,
        warnings=warnings,
    )


def posterior_predictive_check(
    draws: np.ndarray,
    parameter_names: tuple[str, ...],
    y: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    *,
    seed: int = 123,
) -> BayesianPredictiveCheck:
    """Simulate posterior predictive means from stored beta draws."""

    array = _coerce_draw_array(draws, parameter_names)
    observed = np.asarray(y, dtype=float).reshape(-1)
    design = np.asarray(x, dtype=float)
    variances = np.asarray(v, dtype=float)
    if design.ndim != 2:
        raise ValueError("x must be a 2D design matrix.")
    if observed.shape != (design.shape[0],):
        raise ValueError("y length must match x rows.")
    if variances.shape != (design.shape[0], design.shape[0]):
        raise ValueError("v must be a square covariance matrix matching x rows.")
    if design.shape[1] > array.shape[1]:
        raise ValueError("draws must contain at least x.shape[1] beta parameters.")
    if not np.all(np.isfinite(observed)):
        raise ValueError("y must be finite.")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(variances)):
        raise ValueError("x and v must be finite.")

    rng = np.random.default_rng(seed)
    beta_draws = array[:, : design.shape[1]]
    means = beta_draws @ design.T
    predictive = _draw_multivariate_normal_rows(rng, means, variances)
    predictive_means = np.mean(predictive, axis=1)
    observed_mean = float(np.mean(observed))
    p_value = float(np.mean(predictive_means >= observed_mean))
    warnings = (
        "Posterior predictive check uses stored prototype MCMC draws and is not Stan/multinma parity.",
    )
    return BayesianPredictiveCheck(
        check_type="posterior_predictive",
        n_draws=int(array.shape[0]),
        n_observations=int(observed.size),
        observed_mean=observed_mean,
        predictive_mean=float(np.mean(predictive_means)),
        predictive_interval=(
            float(np.percentile(predictive_means, 2.5)),
            float(np.percentile(predictive_means, 97.5)),
        ),
        bayesian_p_value=p_value,
        warnings=warnings,
    )


def posterior_treatment_ranking(
    draws: np.ndarray,
    parameter_names: tuple[str, ...],
    *,
    reference_treatment: str,
    beneficial_direction: str = "lower",
) -> PosteriorTreatmentRanking:
    """Calculate rank probabilities from joint posterior treatment-effect draws."""

    array = _coerce_draw_array(draws, parameter_names)
    if beneficial_direction not in {"lower", "higher"}:
        raise ValueError("beneficial_direction must be 'lower' or 'higher'.")
    if not reference_treatment.strip():
        raise ValueError("reference_treatment must not be empty.")

    treatment_columns: list[tuple[str, int]] = []
    for idx, name in enumerate(parameter_names):
        if name.startswith("trt_") and "_x_" not in name:
            treatment_columns.append((name[4:], idx))
    if not treatment_columns:
        raise ValueError("parameter_names must include at least one trt_ treatment effect.")

    treatments = (reference_treatment, *tuple(treatment for treatment, _ in treatment_columns))
    effect_draws = np.zeros((array.shape[0], len(treatments)), dtype=float)
    for col_idx, (_, draw_idx) in enumerate(treatment_columns, start=1):
        effect_draws[:, col_idx] = array[:, draw_idx]

    if beneficial_direction == "lower":
        order = np.argsort(effect_draws, axis=1)
    else:
        order = np.argsort(-effect_draws, axis=1)
    ranks = np.empty_like(order)
    draw_indices = np.arange(array.shape[0])[:, np.newaxis]
    ranks[draw_indices, order] = np.arange(1, len(treatments) + 1)

    rank_probabilities: dict[str, tuple[float, ...]] = {}
    mean_ranks: dict[str, float] = {}
    for idx, treatment in enumerate(treatments):
        treatment_ranks = ranks[:, idx]
        rank_probabilities[treatment] = tuple(
            float(np.mean(treatment_ranks == rank))
            for rank in range(1, len(treatments) + 1)
        )
        mean_ranks[treatment] = float(np.mean(treatment_ranks))

    warnings = (
        "Ranking preserves joint prototype MCMC draws but is not SUCRA/multinma reference parity.",
    )
    return PosteriorTreatmentRanking(
        treatments=tuple(treatments),
        beneficial_direction=beneficial_direction,
        n_draws=int(array.shape[0]),
        rank_probabilities=rank_probabilities,
        mean_ranks=mean_ranks,
        warnings=warnings,
    )


def _coerce_draw_array(draws: np.ndarray, parameter_names: tuple[str, ...]) -> np.ndarray:
    array = np.asarray(draws, dtype=float)
    if array.ndim == 3:
        array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    if array.ndim != 2:
        raise ValueError("draws must have shape (draw, parameter) or (chain, draw, parameter).")
    if array.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length must match the draws parameter dimension.")
    if array.shape[0] < 4:
        raise ValueError("at least four draws are required.")
    if not np.all(np.isfinite(array)):
        raise ValueError("draws must be finite.")
    return array


def _draw_multivariate_normal_rows(
    rng: np.random.Generator,
    means: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    sign, _ = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("v must be positive definite.")
    noise = rng.multivariate_normal(
        np.zeros(covariance.shape[0], dtype=float),
        covariance,
        size=means.shape[0],
    )
    return means + noise


def _effective_sample_size(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=float).reshape(-1)
    n = series.size
    if n <= 1:
        return float(n)
    centered = series - float(np.mean(series))
    variance = float(np.dot(centered, centered) / n)
    if variance <= 0.0:
        return float(n)
    autocorrelation_sum = 0.0
    for lag in range(1, n):
        covariance = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
        rho = covariance / variance
        if rho <= 0.0:
            break
        autocorrelation_sum += 2.0 * rho
    return float(max(1.0, n / (1.0 + autocorrelation_sum)))


def _split_r_hat(chains: np.ndarray) -> float:
    n_chains, n_draws = chains.shape
    split_draws = n_draws // 2
    if split_draws < 2:
        return float("inf")
    split = chains[:, : split_draws * 2].reshape(n_chains * 2, split_draws)
    chain_means = np.mean(split, axis=1)
    chain_variances = np.var(split, axis=1, ddof=1)
    within = float(np.mean(chain_variances))
    if within <= 0.0:
        return 1.0
    between = float(split_draws * np.var(chain_means, ddof=1))
    var_hat = ((split_draws - 1.0) / split_draws) * within + between / split_draws
    return float(math.sqrt(max(var_hat / within, 0.0)))
