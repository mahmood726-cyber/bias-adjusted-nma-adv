"""Simulation framework for benchmarking advanced bias-adjusted NMA methods."""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from typing import Any

from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler, ValidationError

def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def logit(p: float) -> float:
    p = max(1e-10, min(p, 1.0 - 1e-10))
    return math.log(p / (1.0 - p))

def generate_synthetic_nma_dataset(
    n_studies: int = 30,
    n_treatments: int = 4,
    design_ratio: float = 0.5,  # proportion of NRS vs RCT
    true_heterogeneity: float = 0.25,
    true_bias: float = 0.4,
    true_bias_interaction: float = 0.2,
    covariate_effect: float = 0.3,
    seed: int | None = None
) -> EvidenceDataset:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dataset = EvidenceDataset()
    treatments = [chr(65 + i) for i in range(n_treatments)]  # A, B, C, D...
    ref_treatment = "A"

    # True treatment effects vs A
    true_effects = {t: 0.0 for t in treatments}
    for i, t in enumerate(treatments):
        if t != ref_treatment:
            true_effects[t] = 0.2 + 0.15 * i  # B=0.35, C=0.50, D=0.65...

    for s_idx in range(n_studies):
        study_id = f"study_{s_idx}"
        is_nrs = (s_idx < int(n_studies * design_ratio))
        design = "nrs" if is_nrs else "rct"

        # rob_weight
        if design == "rct":
            rob_weight = float(np.random.beta(8, 2))
        else:
            rob_weight = float(np.random.beta(4, 6))

        # Covariate
        covariate_val = float(np.random.normal(0, 1.0))
        covariates = {"year": covariate_val}

        dataset.add_study(study_id, design, rob_weight, covariates)

        # Assign treatments to arms
        # Star network structure with occasional loops
        if s_idx % 4 == 0:
            # Loop between active arms
            assigned = np.random.choice(treatments[1:], size=2, replace=False).tolist()
        else:
            # Star comparison vs reference A
            assigned = [ref_treatment, np.random.choice(treatments[1:])]

        assigned = sorted(assigned)
        n_arms = len(assigned)

        # Baseline probability
        p0 = np.random.uniform(0.12, 0.28)
        logit_p0 = logit(p0)

        # Random effect
        eta_s = np.random.normal(0, true_heterogeneity)

        for a_idx, trt in enumerate(assigned):
            arm_id = f"arm_{a_idx}"
            n_patients = int(np.random.randint(60, 250))
            dataset.add_arm(study_id, arm_id, trt, n_patients)

            # Compute true probabilities
            eff = true_effects[trt]

            # Add design bias
            if design == "nrs" and trt != ref_treatment:
                eff += true_bias + true_bias_interaction * covariate_val

            # Add covariate effect
            if trt != ref_treatment:
                eff += covariate_effect * covariate_val

            p_arm = expit(logit_p0 + eff + eta_s)

            # Simulate events
            events = int(np.random.binomial(n_patients, p_arm))
            dataset.add_outcome_ad(study_id, arm_id, "O1", "binary", events)

    return dataset


def run_benchmark_simulation(
    n_iterations: int = 100,
    n_studies: int = 30,
    n_treatments: int = 4,
    design_ratio: float = 0.5,
    true_heterogeneity: float = 0.25,
    true_bias: float = 0.4,
    true_bias_interaction: float = 0.2,
    covariate_effect: float = 0.3,
    seed: int = 42
) -> dict[str, Any]:
    np.random.seed(seed)
    random.seed(seed)

    # Let's target treatment B vs A
    target_trt = "B"
    ref_trt = "A"
    true_contrast_effect = 0.35  # true B vs A effect

    # Storage for results
    methods = {
      "standard_nma": [],
      "standard_bias_adj": [],
      "hksj_bias_adj": [],
      "hksj_weighted_bias_adj": [],
      "full_advanced_bias_adj": [],
      "sandwich_bias_adj": [],
      "sandwich_weighted_bias_adj": []
    }

    iterations_run = 0

    for i in range(n_iterations):
        # Generate dataset
        dataset = generate_synthetic_nma_dataset(
            n_studies=n_studies,
            n_treatments=n_treatments,
            design_ratio=design_ratio,
            true_heterogeneity=true_heterogeneity,
            true_bias=true_bias,
            true_bias_interaction=true_bias_interaction,
            covariate_effect=covariate_effect
        )

        # 1. Standard NMA (no design adjustment, no HKSJ, no down-weighting)
        pooler_std = AdvancedBiasAdjustedNMAPooler(hksj=False, down_weight=False)
        try:
            fit_std = pooler_std.fit(
                dataset, "O1", reference_treatment=ref_trt, covariates=[]
            )
            est, se, cil, ciu = fit_std.contrast(target_trt, ref_trt)
            methods["standard_nma"].append((est, se, cil, ciu))
        except ValidationError:
            continue

        # 2. Standard Bias Adjusted NMA (prior bias, no HKSJ, no down-weighting)
        pooler_bias = AdvancedBiasAdjustedNMAPooler(hksj=False, down_weight=False)
        try:
            fit_bias = pooler_bias.fit(
                dataset, "O1", reference_treatment=ref_trt, reference_design="rct", covariates=[]
            )
            est, se, cil, ciu = fit_bias.contrast(target_trt, ref_trt)
            methods["standard_bias_adj"].append((est, se, cil, ciu))
        except ValidationError:
            # Remove last from std list to keep parallel length
            methods["standard_nma"].pop()
            continue

        # 3. HKSJ Bias Adjusted NMA (prior bias + HKSJ, no down-weighting)
        pooler_hksj = AdvancedBiasAdjustedNMAPooler(hksj=True, hksj_df="studies", down_weight=False)
        try:
            fit_hksj = pooler_hksj.fit(
                dataset, "O1", reference_treatment=ref_trt, reference_design="rct", covariates=[]
            )
            est, se, cil, ciu = fit_hksj.contrast(target_trt, ref_trt)
            methods["hksj_bias_adj"].append((est, se, cil, ciu))
        except ValidationError:
            methods["standard_nma"].pop()
            methods["standard_bias_adj"].pop()
            continue

        # 4. HKSJ + Downweighted Bias Adjusted NMA (prior bias + HKSJ + down-weighting)
        pooler_weighted = AdvancedBiasAdjustedNMAPooler(hksj=True, hksj_df="studies", down_weight=True)
        try:
            fit_weighted = pooler_weighted.fit(
                dataset, "O1", reference_treatment=ref_trt, reference_design="rct", covariates=[]
            )
            est, se, cil, ciu = fit_weighted.contrast(target_trt, ref_trt)
            methods["hksj_weighted_bias_adj"].append((est, se, cil, ciu))
        except ValidationError:
            methods["standard_nma"].pop()
            methods["standard_bias_adj"].pop()
            methods["hksj_bias_adj"].pop()
            continue

        # 5. Full Advanced Bias Adjusted NMA (prior bias + HKSJ + down-weighting + meta-regression)
        pooler_full = AdvancedBiasAdjustedNMAPooler(hksj=True, hksj_df="studies", down_weight=True)
        try:
            fit_full = pooler_full.fit(
                dataset, "O1", reference_treatment=ref_trt, reference_design="rct", covariates=["year"]
            )
            # Evaluate at covariate = 0
            est, se, cil, ciu = fit_full.contrast(target_trt, ref_trt, covariates={"year": 0.0})
            methods["full_advanced_bias_adj"].append((est, se, cil, ciu))
        except ValidationError:
            methods["standard_nma"].pop()
            methods["standard_bias_adj"].pop()
            methods["hksj_bias_adj"].pop()
            methods["hksj_weighted_bias_adj"].pop()
            continue

        # 6. Sandwich Bias Adjusted NMA (prior bias + robust sandwich variance, no downweighting, no meta-regression)
        pooler_sand = AdvancedBiasAdjustedNMAPooler(hksj=False, down_weight=False, variance_type="sandwich")
        try:
            fit_sand = pooler_sand.fit(
                dataset, "O1", reference_treatment=ref_trt, reference_design="rct", covariates=[]
            )
            est, se, cil, ciu = fit_sand.contrast(target_trt, ref_trt)
            methods["sandwich_bias_adj"].append((est, se, cil, ciu))
        except ValidationError:
            methods["standard_nma"].pop()
            methods["standard_bias_adj"].pop()
            methods["hksj_bias_adj"].pop()
            methods["hksj_weighted_bias_adj"].pop()
            methods["full_advanced_bias_adj"].pop()
            continue

        # 7. Sandwich + Downweighted Bias Adjusted NMA (prior bias + robust sandwich variance + down-weighting)
        pooler_sand_w = AdvancedBiasAdjustedNMAPooler(hksj=False, down_weight=True, variance_type="sandwich")
        try:
            fit_sand_w = pooler_sand_w.fit(
                dataset, "O1", reference_treatment=ref_trt, reference_design="rct", covariates=[]
            )
            est, se, cil, ciu = fit_sand_w.contrast(target_trt, ref_trt)
            methods["sandwich_weighted_bias_adj"].append((est, se, cil, ciu))
        except ValidationError:
            methods["standard_nma"].pop()
            methods["standard_bias_adj"].pop()
            methods["hksj_bias_adj"].pop()
            methods["hksj_weighted_bias_adj"].pop()
            methods["full_advanced_bias_adj"].pop()
            methods["sandwich_bias_adj"].pop()
            continue

        iterations_run += 1

    # Summarize results
    results_summary = {}
    for m_name, list_vals in methods.items():
        if not list_vals:
            continue
        arr = np.array(list_vals)
        estimates = arr[:, 0]
        ses = arr[:, 1]
        ci_lowers = arr[:, 2]
        ci_uppers = arr[:, 3]

        bias = float(np.mean(estimates - true_contrast_effect))
        mse = float(np.mean((estimates - true_contrast_effect) ** 2))
        rmse = float(math.sqrt(mse))
        coverage = float(np.mean((ci_lowers <= true_contrast_effect) & (true_contrast_effect <= ci_uppers)))
        mean_se = float(np.mean(ses))

        results_summary[m_name] = {
            "bias": bias,
            "mse": mse,
            "rmse": rmse,
            "coverage": coverage,
            "mean_se": mean_se
        }

    return {
        "iterations_attempted": n_iterations,
        "iterations_successful": iterations_run,
        "n_studies": n_studies,
        "n_treatments": n_treatments,
        "true_heterogeneity": true_heterogeneity,
        "true_bias": true_bias,
        "true_bias_interaction": true_bias_interaction,
        "covariate_effect": covariate_effect,
        "true_contrast_effect": true_contrast_effect,
        "methods_summary": results_summary
    }
import math
