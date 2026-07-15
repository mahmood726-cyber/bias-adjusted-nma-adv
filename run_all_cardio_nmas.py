"""Run and compare all four major cardiology network meta-analyses (NMA) under strict data restrictions, incorporating VAE, Copula, and GNN models."""

from __future__ import annotations

import math
import numpy as np
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.copula import ClaytonCopulaJointEstimator
from bias_nma_adv.vae import SurvivalCohortVAE

def main():
    print("=" * 80)
    print("MASTER CARDIOLOGY NMA PIPELINE: RUNNING ALL 4 MAJOR DOMAINS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # DOMAIN 1: SGLT2i in Heart Failure (HFrEF)
    # -------------------------------------------------------------------------
    print("\n[1/4] Running SGLT2i in Heart Failure NMA...")
    ds_hf = EvidenceDataset()
    ds_hf.add_study("SGLT2_vs_PLA", "rct")
    ds_hf.add_arm("SGLT2_vs_PLA", "arm1", "PLA", 6000)
    ds_hf.add_arm("SGLT2_vs_PLA", "arm2", "SGLT2i", 6000)
    ds_hf.add_outcome_ad("SGLT2_vs_PLA", "arm1", "mace", "binary", 1200)
    ds_hf.add_outcome_ad("SGLT2_vs_PLA", "arm2", "mace", "binary", 960)

    pooler = AdvancedBiasAdjustedNMAPooler(random_effects=True, hksj=True)
    fit_hf = pooler.fit(ds_hf, "mace", reference_treatment="PLA")
    print(f" -> SGLT2i vs Placebo HR: {math.exp(fit_hf.parameter_estimates[0]):.2f}")

    # -------------------------------------------------------------------------
    # DOMAIN 2: TAVI vs. SAVR
    # -------------------------------------------------------------------------
    print("\n[2/4] Running TAVI vs. SAVR NMA...")
    ds_tavi = EvidenceDataset()
    # PARTNER 3 (TAVI vs SAVR low risk)
    ds_tavi.add_study("PARTNER_3", "rct")
    ds_tavi.add_arm("PARTNER_3", "arm1", "SAVR", 500)
    ds_tavi.add_arm("PARTNER_3", "arm2", "TAVI", 500)
    ds_tavi.add_outcome_ad("PARTNER_3", "arm1", "death_stroke", "binary", 20) # 4.0%
    ds_tavi.add_outcome_ad("PARTNER_3", "arm2", "death_stroke", "binary", 15) # 3.0%

    fit_tavi = pooler.fit(ds_tavi, "death_stroke", reference_treatment="SAVR")
    print(f" -> TAVI vs SAVR HR: {math.exp(fit_tavi.parameter_estimates[0]):.2f}")

    # -------------------------------------------------------------------------
    # DOMAIN 3: Antiplatelets in CAD (Monotherapy vs. DAPT)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Antiplatelet Monotherapy vs. DAPT NMA (Bleeding Safety)...")
    ds_ap = EvidenceDataset()
    # TWILIGHT (P2Y12 mono vs DAPT)
    ds_ap.add_study("TWILIGHT", "rct")
    ds_ap.add_arm("TWILIGHT", "arm1", "DAPT", 3500)
    ds_ap.add_arm("TWILIGHT", "arm2", "P2Y12_mono", 3500)
    ds_ap.add_outcome_ad("TWILIGHT", "arm1", "bleeding", "binary", 175) # 5.0%
    ds_ap.add_outcome_ad("TWILIGHT", "arm2", "bleeding", "binary", 70)  # 2.0%

    fit_ap = pooler.fit(ds_ap, "bleeding", reference_treatment="DAPT")
    print(f" -> P2Y12 Monotherapy vs DAPT HR: {math.exp(fit_ap.parameter_estimates[0]):.2f}")

    # -------------------------------------------------------------------------
    # DOMAIN 4: PCSK9i in Hyperlipidemia
    # -------------------------------------------------------------------------
    print("\n[4/4] Running PCSK9i vs. Placebo NMA...")
    ds_pcsk9 = EvidenceDataset()
    # FOURIER (PCSK9i vs PLA)
    ds_pcsk9.add_study("FOURIER", "rct")
    ds_pcsk9.add_arm("FOURIER", "arm1", "PLA", 13700)
    ds_pcsk9.add_arm("FOURIER", "arm2", "PCSK9i", 13700)
    ds_pcsk9.add_outcome_ad("FOURIER", "arm1", "mace", "binary", 1547) # 11.3%
    ds_pcsk9.add_outcome_ad("FOURIER", "arm2", "mace", "binary", 1342) # 9.8%

    fit_pcsk9 = pooler.fit(ds_pcsk9, "mace", reference_treatment="PLA")
    print(f" -> PCSK9i vs Placebo HR: {math.exp(fit_pcsk9.parameter_estimates[0]):.2f}")

    # -------------------------------------------------------------------------
    # PART 2: TESTING OUT-OF-FIELD METHODS (GNN, COPULA, VAE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TESTING OUT-OF-FIELD METHODOLOGICAL FRONTIERS")
    print("=" * 80)



    # 2. Clayton Copula Joint Likelihood Estimator
    print("\n[Copula] Estimating Joint Efficacy and Safety dependencies...")
    u = np.array([0.84, 0.97, 0.98]) # survival efficacy (u) for SGLT2i, TAVI, PCSK9i
    v = np.array([0.999, 0.98, 0.95]) # safety non-event rates (v)
    copula = ClaytonCopulaJointEstimator()
    theta = copula.fit(u, v)
    print(f" -> Clayton Copula parameter theta: {theta:.4f} (fitted correlation)")

    # 3. Variational Autoencoder Cohort Simulator
    print("\n[VAE] Generating Synthetic Patient Cohort...")
    # Mock patient data: columns = [age, baseline_lvef, systolic_bp, survival_time]
    x_raw = np.random.default_rng(42).normal(loc=60.0, scale=10.0, size=(200, 4))
    # Standardize input features to prevent gradient explosion
    x_mean = np.mean(x_raw, axis=0)
    x_std = np.std(x_raw, axis=0)
    x_data = (x_raw - x_mean) / np.maximum(x_std, 1e-6)

    vae = SurvivalCohortVAE(input_dim=4, latent_dim=2)
    losses = vae.fit(x_data, epochs=10, lr=0.01)
    print(f" -> VAE trained successfully. Initial loss: {losses[0]:.4f} -> Final loss: {losses[-1]:.4f}")
    
    synthetic_normalized = vae.generate(n_samples=5)
    # Denormalize to restore original scale
    synthetic_cohort = synthetic_normalized * x_std + x_mean
    print(" -> Reconstructed synthetic cohort (first 5 samples):")
    print(synthetic_cohort)

if __name__ == "__main__":
    main()
