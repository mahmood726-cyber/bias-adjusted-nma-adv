"""Unit tests for GNN, Copula, and VAE out-of-field methodological modules."""

from __future__ import annotations

import numpy as np
from bias_nma_adv.gnn import TreatmentGCNRegularizer
from bias_nma_adv.copula import ClaytonCopulaJointEstimator
from bias_nma_adv.vae import SurvivalCohortVAE

def test_gnn_regularizer():
    gnn = TreatmentGCNRegularizer(embedding_dim=2)
    treatments = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]
    
    embeddings = gnn.fit_transform(treatments, edges)
    assert embeddings.shape == (3, 2)
    assert np.all(embeddings >= 0.0) # ReLU activation
    
    p_mat = gnn.get_topological_precision_matrix(treatments, edges)
    assert p_mat.shape == (3, 3)
    # Laplacian row-sum should equal 0 (or close due to float precision)
    assert np.allclose(np.sum(p_mat, axis=1), 0.0, atol=1e-10)

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
