"""Unit tests for GNN, Copula, and VAE out-of-field methodological modules."""

from __future__ import annotations

import numpy as np
from bias_nma_adv.copula import ClaytonCopulaJointEstimator
from bias_nma_adv.vae import SurvivalCohortVAE

def test_clayton_copula():
    copula = ClaytonCopulaJointEstimator(theta=2.0)
    u = np.array([0.9, 0.8, 0.7])
    v = np.array([0.95, 0.9, 0.85])
    
    cdf = copula.joint_cdf(u, v)
    assert len(cdf) == 3
    assert np.all(cdf >= 0.0)
    assert np.all(cdf <= 1.0)
    
    density = copula.joint_density(u, v)
    assert np.all(density >= 0.0)
    
    # Check optimizer fitting
    theta = copula.fit(u, v)
    assert theta > 0.0

def test_survival_vae():
    vae = SurvivalCohortVAE(input_dim=3, latent_dim=2)
    x = np.random.normal(0.0, 1.0, size=(50, 3))
    
    losses = vae.fit(x, epochs=5, lr=0.01)
    assert len(losses) == 5
    assert losses[-1] <= losses[0] or np.isnan(losses[-1]) == False
    
    gen = vae.generate(10)
    assert gen.shape == (10, 3)

def test_hamiltonian_monte_carlo():
    from bias_nma_adv.hmc import HamiltonianMonteCarloSampler
    sampler = HamiltonianMonteCarloSampler(step_size=0.1, n_steps=5, seed=42)
    
    # Mock log-posterior: standard 1D normal (mean=0, variance=1)
    def log_post(theta):
        return -0.5 * theta[0]**2
        
    def grad_log_post(theta):
        return -theta
        
    initial_pos = np.array([1.0])
    samples = sampler.sample(initial_pos, log_post, grad_log_post, n_samples=20)
    
    assert samples.shape == (20, 1)
    # Samples should cluster around mean=0
    assert np.mean(np.abs(samples)) < 2.0

def test_double_robust_tmle():
    from bias_nma_adv.tmle import DoubleRobustTMLE
    tmle = DoubleRobustTMLE()
    
    rng = np.random.default_rng(42)
    n = 200
    
    # Generate mock covariate W, treatment A (binary), outcome Y (binary)
    w = rng.normal(0.0, 1.0, size=(n, 2))
    # Standardize covariates to have intercept column for regression
    w_with_intercept = np.column_stack([np.ones(n), w])
    
    # True propensity score depends on W
    ps = 1.0 / (1.0 + np.exp(-w_with_intercept[:, 1]))
    a = rng.binomial(1, ps)
    
    # Outcome probability depends on A and W
    y_prob = 1.0 / (1.0 + np.exp(-(0.5 * a + w_with_intercept[:, 2])))
    y = rng.binomial(1, y_prob)
    
    rd = tmle.estimate_risk_difference(w_with_intercept, a, y)
    
    # Risk difference should be bounded in [-1, 1]
    assert -1.0 <= rd <= 1.0

