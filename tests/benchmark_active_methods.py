"""Benchmark script for comparing optimized estimators against standard baselines."""

from __future__ import annotations

import time
import numpy as np
from bias_nma_adv.nuts import NoUTurnSampler
from bias_nma_adv.ctmle import CollaborativeTMLE
from bias_nma_adv.gan import ConditionalGAN
from bias_nma_adv.symbolic import SymbolicHazardRegressor

def benchmark_nuts():
    # Target: 1D Standard Normal
    def log_post(theta):
        return -0.5 * theta[0]**2
    def grad_log_post(theta):
        return -theta

    # 1. NUTS Sampler (Optimized)
    t0 = time.perf_counter()
    nuts = NoUTurnSampler(step_size=0.1, seed=42)
    nuts_samples = nuts.sample(np.array([1.0]), log_post, grad_log_post, n_samples=100)
    nuts_time = (time.perf_counter() - t0) * 1000 # ms
    nuts_mean = float(np.mean(nuts_samples))

    # 2. Random-Walk Metropolis-Hastings Baseline
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    pos = np.array([1.0])
    mh_samples = []
    for _ in range(100):
        proposal = pos + rng.normal(0.0, 0.5)
        log_alpha = log_post(proposal) - log_post(pos)
        if rng.random() < np.exp(log_alpha):
            pos = proposal
        mh_samples.append(pos.copy())
    mh_time = (time.perf_counter() - t0) * 1000 # ms
    mh_mean = float(np.mean(mh_samples))

    print(f"[NUTS Bench] NUTS Time: {nuts_time:.2f}ms (Mean: {nuts_mean:.4f}) | MH Time: {mh_time:.2f}ms (Mean: {mh_mean:.4f})")
    return nuts_time, mh_time

def benchmark_ctmle():
    rng = np.random.default_rng(42)
    n = 200
    w = rng.normal(0.0, 1.0, size=(n, 2))
    ps = 1.0 / (1.0 + np.exp(-w[:, 1]))
    a = rng.binomial(1, ps)
    y_prob = 1.0 / (1.0 + np.exp(-(0.5 * a + w[:, 0])))
    y = rng.binomial(1, y_prob)

    # 1. Collaborative TMLE (Optimized)
    t0 = time.perf_counter()
    ctmle = CollaborativeTMLE()
    ctmle_rd = ctmle.estimate_risk_difference(w, a, y)
    ctmle_time = (time.perf_counter() - t0) * 1000 # ms

    # 2. Standard Outcome Regression (G-computation) Baseline
    t0 = time.perf_counter()
    from bias_nma_adv.tmle import LogisticRegressionSolver
    outcome_features = np.column_stack([w, a])
    model = LogisticRegressionSolver()
    model.fit(outcome_features, y)
    features_1 = np.column_stack([w, np.ones(n)])
    features_0 = np.column_stack([w, np.zeros(n)])
    gcomp_rd = float(np.mean(model.predict_proba(features_1) - model.predict_proba(features_0)))
    gcomp_time = (time.perf_counter() - t0) * 1000 # ms

    print(f"[C-TMLE Bench] C-TMLE Time: {ctmle_time:.2f}ms (RD: {ctmle_rd:.4f}) | G-Comp Time: {gcomp_time:.2f}ms (RD: {gcomp_rd:.4f})")
    return ctmle_time, gcomp_time

def main():
    print("=" * 80)
    print("RUNNING METHODOLOGICAL BENCHMARKS")
    print("=" * 80)
    benchmark_nuts()
    benchmark_ctmle()

if __name__ == "__main__":
    main()
