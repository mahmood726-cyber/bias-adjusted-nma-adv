"""Run cardiology demonstration analyses from inline event-count examples.

This script is not validation evidence. Source-backed real-meta benchmarks live
under validation/real_meta and are checked by tests.
"""

from __future__ import annotations

import math
import numpy as np
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.copula import ClaytonCopulaJointEstimator

def main():
    print("=" * 80)
    print("MASTER CARDIOLOGY NMA DEMONSTRATION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # DOMAIN 1: SGLT2i in Heart Failure (DAPA-HF & EMPEROR-Reduced)
    # -------------------------------------------------------------------------
    print("\n[1/4] Running SGLT2i in Heart Failure (HFrEF) NMA...")
    ds_hf = EvidenceDataset()
    # DAPA-HF: Dapagliflozin vs Placebo
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
    print(f" -> SGLT2i vs Placebo Pooled OR: {math.exp(fit_hf.parameter_estimates[0]):.3f}")

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
    print(f" -> TAVI vs SAVR Pooled OR: {math.exp(fit_tavi.parameter_estimates[0]):.3f}")

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
    print(f" -> P2Y12 Monotherapy vs DAPT Pooled OR: {math.exp(fit_ap.parameter_estimates[0]):.3f}")

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
    print(f" -> PCSK9i vs Placebo Pooled OR: {math.exp(fit_pcsk9.parameter_estimates[0]):.3f}")

    # -------------------------------------------------------------------------
    # PART 2: TESTING OUT-OF-FIELD METHODS (COPULA, VAE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TESTING OUT-OF-FIELD METHODOLOGICAL FRONTIERS")
    print("=" * 80)

    # 1. Clayton Copula Joint Likelihood Estimator
    print("\n[Copula] Estimating Joint Efficacy and Safety dependencies...")
    u = np.array([0.837, 0.963, 0.902]) # illustrative efficacy values
    v = np.array([0.999, 0.980, 0.960]) # illustrative safety non-event rates
    copula = ClaytonCopulaJointEstimator()
    theta = copula.fit(u, v)
    print(f" -> Clayton Copula parameter theta: {theta:.4f} (fitted correlation)")

    # 2. Symbolic Regression for Non-Proportional Hazards
    print("\n[Symbolic Regression] Finding closed-form time-varying hazard functions...")
    from bias_nma_adv.symbolic import SymbolicHazardRegressor
    sym_reg = SymbolicHazardRegressor()
    times = np.array([1.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0])
    hazards = 0.05 + 0.02 * np.sqrt(times)
    formula, coefs, mse = sym_reg.fit_best_formula(times, hazards)
    print(f" -> Discovered closed-form hazard: {formula} (MSE: {mse:.6f})")

    # 4. Conditional GAN for Physiological Patient Reconstruction
    print("\n[cGAN] Training Conditional GAN to simulate heart failure cohort...")
    from bias_nma_adv.gan import ConditionalGAN
    # 2 features: [Age, LVEF]
    hf_real = np.random.default_rng(42).normal(loc=[66.0, 31.0], scale=[8.0, 4.0], size=(100, 2))
    # Condition: 1 if severe HF (LVEF < 30), 0 otherwise
    severe_cond = np.where(hf_real[:, 1] < 30.0, 1.0, 0.0).reshape(-1, 1)
    
    gan = ConditionalGAN(noise_dim=2, cond_dim=1, out_dim=2)
    gan_loss = gan.train_step(hf_real, severe_cond, lr=0.01)
    print(f" -> cGAN step completed. Discriminator Loss: {gan_loss:.4f}")
    
    # Generate 3 samples for severe HF condition (cond = [[1.0]])
    gen_severe = gan.generate(np.array([[1.0], [1.0], [1.0]]))
    print(" -> Generated severe HF patient covariates (first 3 samples):")
    print(gen_severe)

    # 5. Bayesian Model Averaging for Network Consistency
    print("\n[BMA] Averaging treatment rankings over consistent and inconsistent models...")
    from bias_nma_adv.bma import BayesianModelAverager
    bma_solver = BayesianModelAverager()
    effects = np.array([-0.25, -0.32]) # [Consistent Model, Inconsistent Model]
    variances = np.array([0.015, 0.020])
    bics = np.array([85.4, 88.9]) # Consistent model (BIC=85.4) is preferred
    bma_eff, bma_var = bma_solver.average_effects(effects, variances, bics)
    print(f" -> BMA Pooled Effect: {bma_eff:.4f} (Pooled Variance: {bma_var:.4f})")


    # 6. Multinomial Logistic GLMM for Competing Endpoints
    print("\n[Multinomial GLMM] Modeling competing outcomes (CV Death vs. Non-CV Death vs. HF Hosp)...")
    from bias_nma_adv.multinomial import MultinomialGLMMSolver
    mult_solver = MultinomialGLMMSolver()
    hf_covs = np.random.default_rng(42).normal(loc=[66.0, 31.0], scale=[8.0, 4.0], size=(10, 2))
    # 3 classes: [CV Death, Non-CV Death, HF Hosp] one-hot encoded
    y_mult = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0],
        [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]
    ])
    mult_solver.fit(hf_covs, y_mult)
    mult_probs = mult_solver.predict_proba(hf_covs[:3])
    print(" -> Multinomial probabilities for first 3 patients:")
    print(mult_probs)

    # 7. Collaborative TMLE Causal Inference
    print("\n[C-TMLE] Estimating collaborative risk differences...")
    from bias_nma_adv.ctmle import CollaborativeTMLE
    ctmle_solver = CollaborativeTMLE()
    w_covs = np.random.default_rng(42).normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(50, 2))
    a_tr = np.random.default_rng(42).binomial(1, 0.5, size=50)
    y_out = np.random.default_rng(42).binomial(1, 0.4, size=50)
    c_rd = ctmle_solver.estimate_risk_difference(w_covs, a_tr, y_out)
    print(f" -> Collaborative TMLE Risk Difference: {c_rd:.4f}")

    # 8. No-U-Turn Sampler (NUTS)
    print("\n[NUTS] Running No-U-Turn Sampler for Bayesian posterior estimation...")
    from bias_nma_adv.nuts import NoUTurnSampler
    nuts_sampler = NoUTurnSampler(step_size=0.1, seed=42)
    def log_post_fn(theta):
        return -0.5 * theta[0]**2
    def grad_log_post_fn(theta):
        return -theta
    
    nuts_samples = nuts_sampler.sample(np.array([1.0]), log_post_fn, grad_log_post_fn, n_samples=5)
    print(" -> NUTS parameter samples:")
    print(nuts_samples)

    # 9. Sponsorship Bias and Attrition Auditor
    print("\n[Sponsor Auditor] Auditing trial collaborator classes and participant flow...")
    from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor
    sponsor_auditor = RegistrySponsorAuditor()
    # Register TWILIGHT (NCT02870140): Industry funded, low attrition (1% LTFU)
    sponsor_auditor.register_trial_flow("NCT02870140", "industry", randomized=7119, lost_to_follow_up=71)
    
    twilight_lar = sponsor_auditor.calculate_attrition_ratio("NCT02870140")
    twilight_sbs = sponsor_auditor.calculate_sponsorship_bias_score("NCT02870140")
    adj_quality = sponsor_auditor.adjust_doi_welton_quality("NCT02870140", base_quality=1.0)
    
    print(f" -> TWILIGHT (NCT02870140): Sponsor class: {twilight_sbs:.1f} (1.0=industry)")
    print(f" -> Attrition Ratio: {twilight_lar:.4f} (Lost-to-Follow-Up)")
    print(f" -> Adjusted Doi-Welton Quality Weight: {adj_quality:.4f} (Base: 1.0)")

if __name__ == "__main__":
    main()


