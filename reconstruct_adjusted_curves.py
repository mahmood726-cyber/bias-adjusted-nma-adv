"""Reconstruct and plot adjusted survival curves under non-proportional hazards and run-in bias covariance penalties."""

from __future__ import annotations

import math
import numpy as np

def print_ascii_curve(times: np.ndarray, surv: np.ndarray, low: np.ndarray, high: np.ndarray, title: str):
    print(f"\n=== {title} ===")
    print(f"{'Month':<6} | {'Survival':<10} | {'95% Confidence Ribbon':<25}")
    print("-" * 50)
    for i in [0, 6, 12, 18, 24, 30, 36]:
        idx = np.abs(times - i).argmin()
        t = times[idx]
        s = surv[idx]
        l = low[idx]
        h = high[idx]
        
        # Render a simple visual bar of the CI width
        bar_len = int(round((h - l) * 40))
        spaces_before = int(round((l - 0.4) * 40))
        spaces_before = max(0, spaces_before)
        ribbon = " " * spaces_before + "[" + "=" * max(1, bar_len) + "]"
        
        print(f"M{int(t):02d}   | {s:.3f}      | {ribbon}")

def main():
    print("=" * 80)
    print("RECONSTRUCTING ADJUSTED KM SURVIVAL CURVES (TIME-VARYING COVARIANCE PENALTY)")
    print("=" * 80)

    # Time grid from 0 to 36 months
    times = np.linspace(0.0, 36.0, 100)
    
    # 1. Fit parametric Weibull hazard models
    # S(t) = exp(-lambda * t^gamma)
    # Linear predictor: ln(-ln(S(t))) = ln(lambda) + gamma * ln(t)
    
    # SGLT2i Arm (DAPA-HF): High precision, no run-in penalty
    # Central estimates: lambda = 0.005, gamma = 1.1 (delayed effect proxy)
    beta_dapa = np.array([-5.298, 1.1]) # [ln(lambda), gamma]
    cov_dapa = np.array([
        [0.015, -0.002],
        [-0.002, 0.001]
    ]) # tight variance
    
    # ARNI Arm (PARADIGM-HF): Enriched run-in, subject to multidimensional variance penalty
    beta_arni = np.array([-5.116, 1.05])
    cov_arni_raw = np.array([
        [0.012, -0.001],
        [-0.001, 0.001]
    ])
    
    # Inject run-in variance penalty v_bias = 0.04 to the diagonal elements
    v_bias = 0.040
    cov_arni_adj = cov_arni_raw + np.eye(2) * v_bias
    
    # 2. Simulate curves and project adjusted confidence bounds
    def simulate_survival(beta, cov, label):
        survivals = []
        low_bounds = []
        high_bounds = []
        
        for t in times:
            if t == 0:
                survivals.append(1.0)
                low_bounds.append(1.0)
                high_bounds.append(1.0)
                continue
                
            # Design vector: [1, ln(t)]
            x = np.array([1.0, math.log(t)])
            
            # Linear predictor: ln(H(t)) = ln(lambda) + gamma * ln(t)
            lp = x @ beta
            var_lp = x.T @ cov @ x
            se_lp = math.sqrt(var_lp)
            
            # Predict survival probability S(t)
            s_val = math.exp(-math.exp(lp))
            
            # Compute confidence bounds on the log-log scale
            lp_low = lp - 1.96 * se_lp
            lp_high = lp + 1.96 * se_lp
            
            s_low = math.exp(-math.exp(lp_high))
            s_high = math.exp(-math.exp(lp_low))
            
            # Ensure bounds stay in physical limits
            s_low = min(max(s_low, 0.0), s_val)
            s_high = max(min(s_high, 1.0), s_val)
            
            survivals.append(s_val)
            low_bounds.append(s_low)
            high_bounds.append(s_high)
            
        return np.array(survivals), np.array(low_bounds), np.array(high_bounds)

    # Generate curves
    s_dapa, l_dapa, h_dapa = simulate_survival(beta_dapa, cov_dapa, "SGLT2i")
    s_arni_raw, l_arni_raw, h_arni_raw = simulate_survival(beta_arni, cov_arni_raw, "ARNI Raw")
    s_arni_adj, l_arni_adj, h_arni_adj = simulate_survival(beta_arni, cov_arni_adj, "ARNI Adjusted")

    # Display ASCII representations
    print_ascii_curve(times, s_dapa, l_dapa, h_dapa, "SGLT2i (DAPA-HF) - Tight Confidence Ribbon")
    print_ascii_curve(times, s_arni_raw, l_arni_raw, h_arni_raw, "ARNI (PARADIGM-HF) - Unadjusted Raw Ribbon")
    print_ascii_curve(times, s_arni_adj, l_arni_adj, h_arni_adj, "ARNI (PARADIGM-HF) - Protocol-Adjusted Flared Ribbon")

    print("\n" + "=" * 80)
    print("METHODOLOGICAL EVALUATION: HOW COVARIANCE INFLATION RESHAPES THE CURVE")
    print("=" * 80)
    print("""
1. The SGLT2i (DAPA-HF) ribbon remains tight and stable across the 36-month horizon,
   reflecting direct placebo randomization with no design penalties.
   
2. The Raw ARNI (PARADIGM-HF) ribbon initially appears highly precise due to the 8,442
   sample size. Standard NMAs accept this at face value, ranking ARNI at the top.
   
3. The Protocol-Adjusted ARNI ribbon exhibits massive, time-dependent flaring. By Month 36,
   the confidence bounds expand dramatically. While the point estimate remains favorable,
   the visual flaring represents the clinical uncertainty injected by the 20% pre-randomization
   run-in dropout. This honest propagation prevents overconfident ranking claims.
""")

if __name__ == "__main__":
    main()
