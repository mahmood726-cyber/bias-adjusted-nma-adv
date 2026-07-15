"""Conditional Generative Adversarial Network (cGAN) for physiological patient covariate reconstruction."""

from __future__ import annotations

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

class ConditionalGAN:
    """cGAN simulator generating synthetic patient covariates conditional on clinical cohort markers."""

    def __init__(self, noise_dim: int = 2, cond_dim: int = 1, out_dim: int = 3, seed: int = 42):
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.rng = np.random.default_rng(seed)
        
        # Initialize generator weights: maps (noise + cond) -> output
        self.w_g = self.rng.normal(0.0, 0.1, size=(out_dim, noise_dim + cond_dim))
        self.b_g = np.zeros(out_dim)
        
        # Initialize discriminator weights: maps (output + cond) -> 1 (real/fake probability)
        self.w_d = self.rng.normal(0.0, 0.1, size=(1, out_dim + cond_dim))
        self.b_d = np.zeros(1)

    def generate(self, cond: np.ndarray) -> np.ndarray:
        """Generate synthetic patient covariates given condition variables.

        cond: shape (n_samples, cond_dim)
        """
        n_samples = len(cond)
        noise = self.rng.normal(0.0, 1.0, size=(n_samples, self.noise_dim))
        
        # Concatenate noise and condition: shape (n_samples, noise_dim + cond_dim)
        inputs = np.column_stack([noise, cond])
        
        # Linear projection
        outputs = inputs @ self.w_g.T + self.b_g
        return outputs

    def train_step(self, real_x: np.ndarray, cond: np.ndarray, lr: float = 0.01) -> float:
        """Perform a single discriminator and generator gradient descent step.

        Returns:
         - Discriminator binary cross-entropy loss.
        """
        n_samples = len(real_x)
        
        # 1. Generate fake samples
        fake_x = self.generate(cond)
        
        # 2. Discriminate real and fake
        real_inputs = np.column_stack([real_x, cond])
        fake_inputs = np.column_stack([fake_x, cond])
        
        real_preds = sigmoid(real_inputs @ self.w_d.T + self.b_d)
        fake_preds = sigmoid(fake_inputs @ self.w_d.T + self.b_d)
        
        # 3. Discriminator loss (binary cross-entropy)
        loss = -float(np.mean(np.log(np.maximum(real_preds, 1e-15)) + np.log(np.maximum(1.0 - fake_preds, 1e-15))))
        
        # 4. Discriminator gradients
        d_loss_d_real = (real_preds - 1.0) / n_samples
        d_loss_d_fake = fake_preds / n_samples
        
        grad_w_d = d_loss_d_real.T @ real_inputs + d_loss_d_fake.T @ fake_inputs
        grad_b_d = np.sum(d_loss_d_real + d_loss_d_fake)
        
        # Update Discriminator parameters
        self.w_d -= lr * grad_w_d
        self.b_d -= lr * grad_b_d
        
        # 5. Generator updates (maximize fake predictions)
        # Gradient of generator loss w.r.t fake predictions: -1 / fake_preds
        d_gen_d_fake = -1.0 / np.maximum(fake_preds, 1e-15) / n_samples
        # Backprop through discriminator sigmoid
        d_gen_d_logits = d_gen_d_fake * fake_preds * (1.0 - fake_preds)
        # Backprop to fake_x: shape (n_samples, out_dim)
        d_gen_d_x = d_gen_d_logits @ self.w_d[:, :self.out_dim]
        
        # Generator gradients w.r.t w_g
        noise = self.rng.normal(0.0, 1.0, size=(n_samples, self.noise_dim))
        gen_inputs = np.column_stack([noise, cond])
        
        grad_w_g = d_gen_d_x.T @ gen_inputs
        grad_b_g = np.sum(d_gen_d_x, axis=0)
        
        # Update Generator parameters
        self.w_g -= lr * grad_w_g
        self.b_g -= lr * grad_b_g
        
        return loss
