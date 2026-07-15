"""Illustrative GLS network meta-analysis for heart-failure event counts.

This script uses inline demonstration values. It is not validation evidence.
Source-backed real-meta benchmarks live under validation/real_meta.
"""

from __future__ import annotations

import math
import numpy as np
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler

def main():
    print("=" * 80)
    print("ILLUSTRATIVE GLS NETWORK META-ANALYSIS: HEART FAILURE EVENT COUNTS")
    print("=" * 80)

    dataset = EvidenceDataset()

    # 1. Add treatments & study designs
    # Demonstration network using binary event counts.
    # Reference treatment is Placebo/SoC (PLA)

    # Trial 1: Pooled SGLT2i vs Placebo (DAPA-HF / EMPEROR-Reduced)
    # Event rate: ~15% vs ~20%, SGLT2i vs PLA
    dataset.add_study("SGLT2_vs_PLA", "rct")
    dataset.add_arm("SGLT2_vs_PLA", "arm1", "PLA", 6000)
    dataset.add_arm("SGLT2_vs_PLA", "arm2", "SGLT2i", 6000)
    dataset.add_outcome_ad("SGLT2_vs_PLA", "arm1", "composite", "binary", 1200) # 20%
    dataset.add_outcome_ad("SGLT2_vs_PLA", "arm2", "composite", "binary", 960)  # 16%

    # Trial 2: PARAGON-HF / PARADIGM-HF (ARNI vs ARB/ACEi)
    # Event rate: ~14% vs ~16%
    dataset.add_study("ARNI_vs_ARB", "rct")
    dataset.add_arm("ARNI_vs_ARB", "arm1", "ARB", 2400)
    dataset.add_arm("ARNI_vs_ARB", "arm2", "ARNI", 2400)
    dataset.add_outcome_ad("ARNI_vs_ARB", "arm1", "composite", "binary", 384)  # 16%
    dataset.add_outcome_ad("ARNI_vs_ARB", "arm2", "composite", "binary", 336)  # 14%

    # Trial 3: TOPCAT (MRA vs Placebo)
    # Regional quality score = 0.55 due to Russia/Georgia data quality issues
    dataset.add_study("TOPCAT", "rct", rob_weight=0.55)
    dataset.add_arm("TOPCAT", "arm1", "PLA", 1720)
    dataset.add_arm("TOPCAT", "arm2", "MRA", 1720)
    dataset.add_outcome_ad("TOPCAT", "arm1", "composite", "binary", 344)  # 20%
    dataset.add_outcome_ad("TOPCAT", "arm2", "composite", "binary", 306)  # 17.8%

    # Trial 4: CHARM-Alternative / I-PRESERVE (ARB vs Placebo)
    # Event rate: ~21% vs ~22%
    dataset.add_study("ARB_vs_PLA", "rct")
    dataset.add_arm("ARB_vs_PLA", "arm1", "PLA", 3570)
    dataset.add_arm("ARB_vs_PLA", "arm2", "ARB", 3570)
    dataset.add_outcome_ad("ARB_vs_PLA", "arm1", "composite", "binary", 785)  # 22%
    dataset.add_outcome_ad("ARB_vs_PLA", "arm2", "composite", "binary", 746)  # 20.9%

    # Trial 5: DIG Trial (Digoxin vs Placebo)
    # Event rate: ~22% vs ~27%
    dataset.add_study("DIG_Anc", "rct")
    dataset.add_arm("DIG_Anc", "arm1", "PLA", 494)
    dataset.add_arm("DIG_Anc", "arm2", "Digoxin", 494)
    dataset.add_outcome_ad("DIG_Anc", "arm1", "composite", "binary", 133)  # 27%
    dataset.add_outcome_ad("DIG_Anc", "arm2", "composite", "binary", 109)  # 22%

    # Fit using our Advanced NMA pooler with quality down-weighting & Kenward-Roger correction
    pooler = AdvancedBiasAdjustedNMAPooler(
        random_effects=True,
        down_weight=True,
        hksj=True
    )
    fit_result = pooler.fit(dataset, "composite", reference_treatment="PLA")

    # Display results
    print("\n--- Model Parameters and Point Estimates ---")
    for name in fit_result.parameter_names:
        idx = fit_result.parameter_names.index(name)
        est = fit_result.parameter_estimates[idx]
        se = math.sqrt(fit_result.parameter_cov[idx, idx])
        odds_ratio = math.exp(est)
        ci_low = math.exp(est - 1.96 * se)
        ci_high = math.exp(est + 1.96 * se)
        print(f" - {name}: OR = {odds_ratio:.2f} (95% CI: {ci_low:.2f} to {ci_high:.2f})")

    # Estimate indirect ARNI vs Placebo
    arni_effect, arni_se, arni_low, arni_high = fit_result.contrast("ARNI", "PLA")
    print(f"\n - ARNI vs Placebo (Bridged Indirectly): OR = {math.exp(arni_effect):.2f} (95% CI: {math.exp(arni_low):.2f} to {math.exp(arni_high):.2f})")

    # -------------------------------------------------------------------------
    # PART 2: LIMITATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMONSTRATION LIMITATIONS")
    print("=" * 80)
    print("""
This script reports odds ratios from binary event counts. It does not estimate
hazard ratios, does not validate against published HFrEF NMAs, and does not
certify bias correction. Use validation/real_meta for source-backed benchmarks.
""")

if __name__ == "__main__":
    main()
