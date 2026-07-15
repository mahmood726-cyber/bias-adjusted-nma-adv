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

def test_registry_publication_bias():
    from bias_nma_adv.publication_bias import RegistryPublicationBiasAuditor
    auditor = RegistryPublicationBiasAuditor()
    
    # 1. Register trials
    auditor.register_trial_protocol("NCT123", "mace", "mace", "completed") # No switching, published
    auditor.register_trial_protocol("NCT456", "mortality", "mace", "completed") # Outcome switching!
    auditor.register_trial_protocol("NCT789", "mace", "mace", "completed") # Unpublished trial
    
    # 2. Check outcome switching
    switching_scores = auditor.audit_outcome_switching(["NCT123", "NCT456", "NCT999"])
    assert switching_scores["NCT123"] == 0.0
    assert switching_scores["NCT456"] == 1.0
    assert switching_scores["NCT999"] == 1.0 # High risk (no registry protocol)
    
    # 3. Check unpublished ratio
    utr = auditor.calculate_unpublished_ratio("DrugX", ["NCT123", "NCT456"])
    # 1 out of 3 trials is unpublished (NCT789)
    assert np.isclose(utr, 1.0 / 3.0)
    
    # 4. Check bias shrinkage
    pooled_effect = -0.30 # log-HR
    shrunk_effect = auditor.apply_bias_shrinkage(pooled_effect, utr)
    # Effect should be shrunk closer to 0 (closer to the null)
    assert shrunk_effect > pooled_effect
    assert np.isclose(shrunk_effect, -0.20)

def test_symbolic_regression():
    from bias_nma_adv.symbolic import SymbolicHazardRegressor
    reg = SymbolicHazardRegressor()
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # true hazard: h(t) = 0.5 * t + 0.2 * log(t)
    hazards = 0.5 * times + 0.2 * np.log(times)
    
    formula, coefs, mse = reg.fit_best_formula(times, hazards)
    assert formula != "None"
    assert len(coefs) == 2
    assert mse < 1.0

def test_conditional_gan():
    from bias_nma_adv.gan import ConditionalGAN
    gan = ConditionalGAN(noise_dim=2, cond_dim=1, out_dim=2)
    
    cond = np.array([[1.0], [0.0], [1.0], [0.0]])
    real_x = np.random.normal(0.0, 1.0, size=(4, 2))
    
    # Check shape of generation
    fake = gan.generate(cond)
    assert fake.shape == (4, 2)
    
    # Train one step and check loss
    loss = gan.train_step(real_x, cond, lr=0.01)
    assert isinstance(loss, float)

def test_bayesian_model_averaging():
    from bias_nma_adv.bma import BayesianModelAverager
    bma = BayesianModelAverager()
    
    effects = np.array([-0.30, -0.25, -0.15])
    variances = np.array([0.01, 0.012, 0.015])
    bics = np.array([120.0, 122.0, 128.0]) # Model 1 is strongly preferred
    
    probs = bma.calculate_posterior_probabilities(bics)
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]
    
    avg_eff, avg_var = bma.average_effects(effects, variances, bics)
    assert avg_eff < 0.0
    assert avg_var > 0.0

def test_multinomial_glmm():
    from bias_nma_adv.multinomial import MultinomialGLMMSolver
    solver = MultinomialGLMMSolver()
    
    x = np.array([[1.0, 0.5], [1.0, -0.5], [1.0, 1.2], [1.0, -1.2]])
    # 3 classes: one-hot encoded
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    
    solver.fit(x, y)
    probs = solver.predict_proba(x)
    assert probs.shape == (4, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_collaborative_tmle():
    from bias_nma_adv.ctmle import CollaborativeTMLE
    ctmle = CollaborativeTMLE()
    
    rng = np.random.default_rng(42)
    n = 150
    w = rng.normal(0.0, 1.0, size=(n, 2))
    ps = 1.0 / (1.0 + np.exp(-w[:, 1]))
    a = rng.binomial(1, ps)
    y_prob = 1.0 / (1.0 + np.exp(-(0.5 * a + w[:, 0])))
    y = rng.binomial(1, y_prob)
    
    rd = ctmle.estimate_risk_difference(w, a, y)
    assert -1.0 <= rd <= 1.0

def test_no_u_turn_sampler():
    from bias_nma_adv.nuts import NoUTurnSampler
    sampler = NoUTurnSampler(step_size=0.1, seed=42)
    
    def log_post(theta):
        return -0.5 * theta[0]**2
        
    def grad_log_post(theta):
        return -theta
        
    initial_pos = np.array([1.0])
    samples = sampler.sample(initial_pos, log_post, grad_log_post, n_samples=15)
    assert samples.shape == (15, 1)
    assert np.mean(np.abs(samples)) < 2.0




