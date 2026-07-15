"""Demonstration of Exact Binomial GLMM NMA and Guyot IPD Reconstruction on real-world SGLT2i Heart Failure trials."""

from __future__ import annotations

import numpy as np
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.bayesian import BayesianNMAMCMCSampler
from bias_nma_adv.survival import SurvivalIPDReconstructor

def main():
    print("=" * 80)
    print("REAL-WORLD CARDIOLOGY NMA DEMONSTRATION: SGLT2 INHIBITORS IN HEART FAILURE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: RARE SAFETY ENDPOINT (Diabetic Ketoacidosis - DKA)
    # -------------------------------------------------------------------------
    print("\n--- PART 1: Safety NMA (DKA Rare Events with Double-Zero & Single-Zero Cells) ---")
    dataset = EvidenceDataset()

    # DAPA-HF (N=4744)
    # Dapa: 3 events in 2373 | Placebo: 0 events in 2371
    dataset.add_study("DAPA-HF", "rct")
    dataset.add_arm("DAPA-HF", "arm1", "Placebo", 2371)
    dataset.add_arm("DAPA-HF", "arm2", "Dapagliflozin", 2373)
    dataset.add_outcome_ad("DAPA-HF", "arm1", "DKA", "binary", 0.0)
    dataset.add_outcome_ad("DAPA-HF", "arm2", "DKA", "binary", 3.0)

    # EMPEROR-Reduced (N=3730)
    # Empa: 0 events in 1863 | Placebo: 0 events in 1867 (Double Zero!)
    dataset.add_study("EMPEROR-Reduced", "rct")
    dataset.add_arm("EMPEROR-Reduced", "arm1", "Placebo", 1867)
    dataset.add_arm("EMPEROR-Reduced", "arm2", "Empagliflozin", 1863)
    dataset.add_outcome_ad("EMPEROR-Reduced", "arm1", "DKA", "binary", 0.0)
    dataset.add_outcome_ad("EMPEROR-Reduced", "arm2", "DKA", "binary", 0.0)

    # SOLOIST-WHF (N=1222)
    # Sota: 2 events in 608 | Placebo: 1 event in 614
    dataset.add_study("SOLOIST-WHF", "rct")
    dataset.add_arm("SOLOIST-WHF", "arm1", "Placebo", 614)
    dataset.add_arm("SOLOIST-WHF", "arm2", "Sotagliflozin", 608)
    dataset.add_outcome_ad("SOLOIST-WHF", "arm1", "DKA", "binary", 1.0)
    dataset.add_outcome_ad("SOLOIST-WHF", "arm2", "DKA", "binary", 2.0)

    # Fit using the Exact Binomial GLMM Pooler (automatically triggers on zero-cells)
    pooler = AdvancedBiasAdjustedNMAPooler(exact_binomial=True)
    fit_result = pooler.fit(dataset, "DKA", reference_treatment="Placebo")

    print(f"Exact Binomial GLMM Active: {fit_result.exact_binomial_active}")
    print("Treatment Effects (Log-Odds vs. Placebo):")
    for trt, est in fit_result.treatment_effects.items():
        if trt != "Placebo":
            se = fit_result.treatment_ses[trt]
            print(f" - {trt}: {est:.4f} (SE: {se:.4f})")

    # -------------------------------------------------------------------------
    # PART 2: SURVIVAL IPD RECONSTRUCTION (Guyot Algorithm on primary efficacy)
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Survival IPD Reconstruction (Guyot Curve Reconstructor) ---")
    reconstructor = SurvivalIPDReconstructor()

    # Simulated coordinates representing DAPA-HF Placebo Arm KM Curve
    # Starts at 1.0, drops to 0.85 at 12 months, 0.75 at 24 months
    dapa_t = [0.0, 3.0, 6.0, 12.0, 18.0, 24.0]
    dapa_s = [1.0, 0.95, 0.91, 0.85, 0.79, 0.75]
    dapa_nar_t = [0.0, 12.0, 24.0]
    dapa_nar_v = [2371, 2015, 1780]

    reconstructor.add_arm_curve(
        arm_id=0, # Placebo
        times=dapa_t,
        survivals=dapa_s,
        n_risk_times=dapa_nar_t,
        n_risk_values=dapa_nar_v,
        total_n=2371
    )

    # Simulated coordinates representing DAPA-HF Dapagliflozin Arm KM Curve
    # Starts at 1.0, drops to 0.88 at 12 months, 0.81 at 24 months (HR ~ 0.74)
    dapa_t_active = [0.0, 3.0, 6.0, 12.0, 18.0, 24.0]
    dapa_s_active = [1.0, 0.96, 0.93, 0.88, 0.84, 0.81]
    dapa_nar_v_active = [2373, 2088, 1922]

    reconstructor.add_arm_curve(
        arm_id=1, # Dapagliflozin
        times=dapa_t_active,
        survivals=dapa_s_active,
        n_risk_times=dapa_nar_t,
        n_risk_values=dapa_nar_v_active,
        total_n=2373
    )

    t_arr, e_arr, arm_arr = reconstructor.get_combined_ipd()
    print(f"Reconstructed Individual Patient Records: {len(t_arr)}")
    print(f" - Placebo events: {np.sum((e_arr == 1) & (arm_arr == 0))}")
    print(f" - Dapagliflozin events: {np.sum((e_arr == 1) & (arm_arr == 1))}")

    # -------------------------------------------------------------------------
    # PART 3: BAYESIAN MCMC SAMPLING (MCMC credible intervals)
    # -------------------------------------------------------------------------
    print("\n--- PART 3: Bayesian NMA MCMC Posterior Sampling ---")
    
    # We will sample the safety dataset using our Metropolis-Hastings MCMC engine
    blocks = pooler._build_study_blocks(dataset, "DKA", "binary")
    unique_designs = ["rct"]
    design_to_idx = {"rct": 0}
    param_names = pooler._build_parameter_names(
        ("Dapagliflozin", "Empagliflozin", "Sotagliflozin"), (), [], False, ()
    )
    y_vec, x_mat, v_mat = pooler._assemble_design(
        blocks, param_names, reference_treatment="Placebo", reference_design="rct", cov_names=[], study_specific_bias=False
    )

    sampler = BayesianNMAMCMCSampler(n_samples=2000, burn_in=500, thinning=1)
    mcmc_result = sampler.fit(
        y=y_vec,
        x=x_mat,
        v=v_mat,
        param_names=param_names,
        blocks=blocks,
        unique_designs=unique_designs,
        design_to_idx=design_to_idx,
        bias_prior_sd=1.0,
        treatment_shrinkage_lambda=0.0
    )

    print(f"MCMC Acceptance Rate: {mcmc_result.acceptance_rate * 100:.1f}%")
    print("Posterior Credible Intervals (95% CrI vs. Placebo):")
    for name in param_names:
        mean_val = mcmc_result.posterior_means[name]
        ci = mcmc_result.credible_intervals[name]
        print(f" - {name}: {mean_val:.4f} (95% CrI: {ci[0]:.4f} to {ci[1]:.4f})")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
