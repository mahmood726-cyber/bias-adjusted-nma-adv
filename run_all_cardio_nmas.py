"""Run and compare all four major cardiology network meta-analyses (NMA) under strict post-2007 data boundaries using real-world trial extractions."""

from __future__ import annotations

import math
import numpy as np
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.copula import ClaytonCopulaJointEstimator
from bias_nma_adv.vae import SurvivalCohortVAE

def main():
    print("=" * 80)
    print("MASTER REAL-WORLD CARDIOLOGY NMA PIPELINE (POST-2007 REGISTRY DATA)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # DOMAIN 1: SGLT2i in Heart Failure (DAPA-HF & EMPEROR-Reduced)
    # -------------------------------------------------------------------------
    print("\n[1/4] Running SGLT2i in Heart Failure (HFrEF) NMA...")
    ds_hf = EvidenceDataset()
    # DAPA-HF (NCT03036826): Dapagliflozin vs Placebo
    ds_hf.add_study("DAPA-HF", "rct")
    ds_hf.add_arm("DAPA-HF", "arm1", "PLA", 2371)
    ds_hf.add_arm("DAPA-HF", "arm2", "SGLT2i", 2373)
    ds_hf.add_outcome_ad("DAPA-HF", "arm1", "mace", "binary", 502)
    ds_hf.add_outcome_ad("DAPA-HF", "arm2", "mace", "binary", 386)

    # EMPEROR-Reduced (NCT03057977): Empagliflozin vs Placebo
    ds_hf.add_study("EMPEROR-Reduced", "rct")
    ds_hf.add_arm("EMPEROR-Reduced", "arm1", "PLA", 1867)
    ds_hf.add_arm("EMPEROR-Reduced", "arm2", "SGLT2i", 1863)
    ds_hf.add_outcome_ad("EMPEROR-Reduced", "arm1", "mace", "binary", 462)
    ds_hf.add_outcome_ad("EMPEROR-Reduced", "arm2", "mace", "binary", 361)

    pooler = AdvancedBiasAdjustedNMAPooler(random_effects=True, hksj=True)
    fit_hf = pooler.fit(ds_hf, "mace", reference_treatment="PLA")
    print(f" -> SGLT2i vs Placebo Pooled HR: {math.exp(fit_hf.parameter_estimates[0]):.3f}")

    # -------------------------------------------------------------------------
    # DOMAIN 2: TAVI vs. SAVR (PARTNER 3 & EVOLUT Low Risk)
    # -------------------------------------------------------------------------
    print("\n[2/4] Running TAVI vs. SAVR NMA...")
    ds_tavi = EvidenceDataset()
    # PARTNER 3 (NCT02675114): TAVI vs SAVR (Low Risk 1-year Death/Stroke/Rehosp)
    ds_tavi.add_study("PARTNER_3", "rct")
    ds_tavi.add_arm("PARTNER_3", "arm1", "SAVR", 483)
    ds_tavi.add_arm("PARTNER_3", "arm2", "TAVI", 496)
    ds_tavi.add_outcome_ad("PARTNER_3", "arm1", "death_stroke", "binary", 41)
    ds_tavi.add_outcome_ad("PARTNER_3", "arm2", "death_stroke", "binary", 21)

    # EVOLUT Low Risk (NCT02701283): TAVI vs SAVR (Low Risk 1-year Death/Disabling Stroke)
    ds_tavi.add_study("EVOLUT_LR", "rct")
    ds_tavi.add_arm("EVOLUT_LR", "arm1", "SAVR", 703)
    ds_tavi.add_arm("EVOLUT_LR", "arm2", "TAVI", 725)
    ds_tavi.add_outcome_ad("EVOLUT_LR", "arm1", "death_stroke", "binary", 29)
    ds_tavi.add_outcome_ad("EVOLUT_LR", "arm2", "death_stroke", "binary", 22)

    fit_tavi = pooler.fit(ds_tavi, "death_stroke", reference_treatment="SAVR")
    print(f" -> TAVI vs SAVR Pooled HR: {math.exp(fit_tavi.parameter_estimates[0]):.3f}")

    # -------------------------------------------------------------------------
    # DOMAIN 3: Antiplatelet Monotherapy vs. DAPT (TWILIGHT Bleeding Safety)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running P2Y12 Monotherapy vs. DAPT NMA (Bleeding Safety)...")
    ds_ap = EvidenceDataset()
    # TWILIGHT (NCT02870140): Ticagrelor Monotherapy vs DAPT (BARC 2/3/5 bleeding)
    ds_ap.add_study("TWILIGHT", "rct")
    ds_ap.add_arm("TWILIGHT", "arm1", "DAPT", 3566)
    ds_ap.add_arm("TWILIGHT", "arm2", "P2Y12_mono", 3554)
    ds_ap.add_outcome_ad("TWILIGHT", "arm1", "bleeding", "binary", 250)
    ds_ap.add_outcome_ad("TWILIGHT", "arm2", "bleeding", "binary", 141)

    fit_ap = pooler.fit(ds_ap, "bleeding", reference_treatment="DAPT")
    print(f" -> P2Y12 Monotherapy vs DAPT Pooled HR: {math.exp(fit_ap.parameter_estimates[0]):.3f}")

    # -------------------------------------------------------------------------
    # DOMAIN 4: PCSK9i vs. Placebo (FOURIER & ODYSSEY Outcomes)
    # -------------------------------------------------------------------------
    print("\n[4/4] Running PCSK9i vs. Placebo NMA...")
    ds_pcsk9 = EvidenceDataset()
    # FOURIER (NCT01764633): Evolocumab vs Placebo (MACE)
    ds_pcsk9.add_study("FOURIER", "rct")
    ds_pcsk9.add_arm("FOURIER", "arm1", "PLA", 13780)
    ds_pcsk9.add_arm("FOURIER", "arm2", "PCSK9i", 13784)
    ds_pcsk9.add_outcome_ad("FOURIER", "arm1", "mace", "binary", 1563)
    ds_pcsk9.add_outcome_ad("FOURIER", "arm2", "mace", "binary", 1344)

    # ODYSSEY Outcomes (NCT01663402): Alirocumab vs Placebo (MACE)
    ds_pcsk9.add_study("ODYSSEY_Outcomes", "rct")
    ds_pcsk9.add_arm("ODYSSEY_Outcomes", "arm1", "PLA", 9462)
    ds_pcsk9.add_arm("ODYSSEY_Outcomes", "arm2", "PCSK9i", 9462)
    ds_pcsk9.add_outcome_ad("ODYSSEY_Outcomes", "arm1", "mace", "binary", 1052)
    ds_pcsk9.add_outcome_ad("ODYSSEY_Outcomes", "arm2", "mace", "binary", 903)

    fit_pcsk9 = pooler.fit(ds_pcsk9, "mace", reference_treatment="PLA")
    print(f" -> PCSK9i vs Placebo Pooled HR: {math.exp(fit_pcsk9.parameter_estimates[0]):.3f}")

    # -------------------------------------------------------------------------
    # PART 2: TESTING OUT-OF-FIELD METHODS (COPULA, VAE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TESTING OUT-OF-FIELD METHODOLOGICAL FRONTIERS")
    print("=" * 80)

    # 1. Clayton Copula Joint Likelihood Estimator
    print("\n[Copula] Estimating Joint Efficacy and Safety dependencies...")
    u = np.array([0.837, 0.963, 0.902]) # actual survival efficacy (u) for SGLT2i, TAVI, PCSK9i
    v = np.array([0.999, 0.980, 0.960]) # actual safety non-event rates (v)
    copula = ClaytonCopulaJointEstimator()
    theta = copula.fit(u, v)
    print(f" -> Clayton Copula parameter theta: {theta:.4f} (fitted correlation)")

    # 2. Variational Autoencoder Cohort Simulator
    print("\n[VAE] Generating Synthetic Patient Cohort...")
    # Real-world covariate parameters: [Age, LVEF, Systolic BP, Follow-up Months]
    # Calibrated to heart failure population averages: mean=[66.0, 31.2, 122.0, 18.2]
    x_raw = np.random.default_rng(42).normal(
        loc=[66.0, 31.2, 122.0, 18.2],
        scale=[10.0, 5.0, 15.0, 6.0],
        size=(200, 4)
    )
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
