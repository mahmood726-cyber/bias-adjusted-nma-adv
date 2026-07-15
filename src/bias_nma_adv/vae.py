"""NumPy-based Variational Autoencoder (VAE) for generating synthetic patient-level cohorts."""

from __future__ import annotations

import numpy as np

class SurvivalCohortVAE:
    """Simple Variational Autoencoder (VAE) to generate synthetic patient-level cohorts (survival, covariates)."""

    def __init__(self, input_dim: int, latent_dim: int = 2, seed: int = 42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)
        
        # Encoder weights (Layer 1: Input -> Latent Mean, Layer 2: Input -> Latent LogVar)
        self.w_enc_mu = self.rng.normal(0.0, 0.1, size=(input_dim, latent_dim))
        self.b_enc_mu = np.zeros(latent_dim)
        self.w_enc_logvar = self.rng.normal(0.0, 0.1, size=(input_dim, latent_dim))
        self.b_enc_logvar = np.zeros(latent_dim)
        
        # Decoder weights (Layer 1: Latent -> Reconstructed Input)
        self.w_dec = self.rng.normal(0.0, 0.1, size=(latent_dim, input_dim))
        self.b_dec = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode input covariates into mean and log-variance parameters."""
        mu = x @ self.w_enc_mu + self.b_enc_mu
        logvar = x @ self.w_enc_logvar + self.b_enc_logvar
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Sample from the latent space using the reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = self.rng.normal(size=mu.shape)
        return mu + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent samples back to reconstructed input spaces."""
        return z @ self.w_dec + self.b_dec

    def loss(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> float:
        """Compute the VAE loss (MSE reconstruction loss + KL Divergence)."""
        recon_loss = np.mean(np.square(x - x_recon))
        kl_loss = -0.5 * np.mean(np.sum(1.0 + logvar - np.square(mu) - np.exp(logvar), axis=1))
        return float(recon_loss + 0.1 * kl_loss)

    def fit(self, x: np.ndarray, epochs: int = 10, lr: float = 0.01) -> list[float]:
        """Train the VAE weights using basic SGD."""
        losses = []
        n_samples = len(x)
        
        for _ in range(epochs):
            # Forward Pass
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            
            # Loss
            loss_val = self.loss(x, x_recon, mu, logvar)
            losses.append(loss_val)
            
            # Gradients (Reconstruction gradient)
            d_recon = 2.0 * (x_recon - x) / n_samples
            
            # Decoder Backpropagation
            dw_dec = z.T @ d_recon
            db_dec = np.sum(d_recon, axis=0)
            
            # Encoder Backpropagation (chain rule through z)
            dz = d_recon @ self.w_dec.T
            dmu = dz
            dlogvar = dz * 0.5 * np.exp(0.5 * logvar) * self.rng.normal(size=mu.shape)
            
            # Encoder weight updates
            dw_enc_mu = x.T @ dmu
            db_enc_mu = np.sum(dmu, axis=0)
            dw_enc_logvar = x.T @ dlogvar
            db_enc_logvar = np.sum(dlogvar, axis=0)
            
            # Apply SGD updates
            self.w_dec -= lr * dw_dec
            self.b_dec -= lr * db_dec
            self.w_enc_mu -= lr * dw_enc_mu
            self.b_enc_mu -= lr * db_enc_mu
            self.w_enc_logvar -= lr * dw_enc_logvar
            self.b_enc_logvar -= lr * db_enc_logvar
            
        return losses

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic patient cohorts by sampling the standard normal prior."""
        z = self.rng.normal(size=(n_samples, self.latent_dim))
        return self.decode(z)
