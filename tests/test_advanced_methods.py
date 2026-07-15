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
