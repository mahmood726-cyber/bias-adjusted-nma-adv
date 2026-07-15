> **Historical review artifact - not validation evidence.** This document is a critique/planning transcript only. It is not source extraction evidence, not reference-software parity evidence, not certification evidence, not clinical guidance, and not proof of superiority. Treat all numerical, clinical, guideline, and publication-status statements below as hypotheses unless they are backed by machine-verifiable artifacts in this repository.
# Hardcore Methodological Review (Round 14): Advanced Non-Proportional Hazards Integration

This document registers the transcript of the fourteenth-round multiperson adversarial review, evaluating the clinical and mathematical utility of fractional polynomials, multidimensional covariance penalties, causal transportability, Bayesian QBA, and G-computation target trial emulation in our advanced NMA engine.

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## 1. Methodological Evaluation of the Proposed Frameworks

### 1.1. Fractional Polynomial (FP) NMA & Multidimensional Bias Penalty
**Dr. Fiona Vance:**
> "The shift to Fractional Polynomials is mathematically necessary when proportional hazards (PH) fail, which is common in cardiovascular trials with delayed treatment effects (e.g. PARADIGM-HF, NCT01035541). 
> 
> When the treatment effect is modeled as a vector $\boldsymbol{\theta} = [\beta_1, \beta_2]^T$ (representing scale and shape parameters), a standard scalar bias penalty is no longer sufficient. Injecting the protocol bias penalty ($v_{bias}$) as a diagonal inflation of the covariance matrix:
> 
> \Sigma_{adjusted} = \Sigma_{original} + (I_k \times v_{bias})
> 
> is an elegant solution. By adding the penalty to the diagonals, we inflate early-phase and late-phase treatment uncertainty without introducing spurious covariance correlations. This allows the uncertainty of pre-randomization run-in exclusions (like the 20% dropout in PARADIGM-HF) to propagate dynamically across the survival curves."

---

### 1.2. Causal Transportability (CIMA) via IPSW
**Dr. Cynthia Registry:**
> "CIMA via Inverse Probability of Sampling Weighting (IPSW) represents the gold standard for adjusting covariate differences between selected and unselected populations. 
> 
> In heart failure networks, the baseline covariates of enriched trials (e.g., NCT01035541, which excluded patients who developed hypotension or hyperkalemia during the run-in) differ systematically from direct-randomization trials (e.g., DAPA-HF, NCT03036826). 
> 
> By extracting covariate means from ClinicalTrials.gov and applying IPSW to the reconstructed pseudo-IPD, we can analytically 'transport' the enriched trial's Kaplan-Meier curve to the target patient cohort, simulating how the drug would have performed in a real-world, unselected population. This directly addresses the transitivity limitation of standard NMAs."

---

### 1.3. Bayesian Quantitative Bias Analysis (QBA)
**Dr. Benjamin MCMC:**
> "Adding Bayesian QBA to our MCMC sampler is a major upgrade. In our current implementation, $v_{bias}$ is a fixed scalar. In reality, run-in dropout rates are subject to sampling variance. 
> 
> By defining $\beta_{bias}$ as a random variable parameterized by the observed dropout rate ($\beta_{bias} \sim \mathcal{N}(\mu_{dropout}, \sigma_{dropout}^2)$) and subtracting it from the observed treatment effect in the MCMC chain:
> 
> \delta_{true, i} = \delta_{obs, i} - \beta_{bias, i}
> 
> we allow the full probability distribution of the protocol bias to organically propagate through the network. This yields a more robust and honest credible interval for treatment comparisons."

---

### 1.4. Target Trial Emulation via G-Computation
**Dr. Cynthia Registry:**
> "Target Trial Emulation via G-computation bypasses the Hazard Ratio entirely. It is particularly useful for clinical communication. 
> 
> Instead of presenting abstract parameters like $\beta_1$ and $\beta_2$, G-computation simulates a single synthetic patient cohort (which we can generate using our **Survival VAE cohort simulator**) and pushes it through counterfactual treatment paths. This yields a direct estimate of the **Marginal Survival Difference** over time:
> 
> \text{Risk Diff}(t) = \frac{1}{N} \sum_{i=1}^N \left[ \hat{S}(t \mid X_i, A=1) - \hat{S}(t \mid X_i, A=0) \right]
> 
> This is highly preferred by Health Technology Assessment bodies (like NICE) because it translates complex non-proportional hazards into concrete clinical metrics (e.g., 'absolute difference in life expectancy at 24 months')."

---

## 2. Structural Implementation Blueprint

We propose integrating these advanced models into the advanced NMA repository:

```
src/bias_nma_adv/
  ├── survival.py     <-- Audits reconstruction via IAE Wasserstein.
  ├── model.py        <-- Fits exact GLMM and GLS poolers.
  ├── vae.py          <-- Generates synthetic cohort (G-computation target).
  ├── copula.py       <-- Models efficacy-safety joint likelihood.
  ├── gnn.py          <-- Regularizes topology using network embeddings.
  └── time_varying.py <-- NEW: Fits Fractional Polynomials, applies Covariance Penalties,
                          and runs Target Trial Emulation.
```

---

## Final Recommendation: Implement Time-Varying Extensions

**Panel Consensus:**
> "The proposed methods are highly valuable and directly address the breakdown of proportional hazards in cardiology. We recommend integrating the **Fractional Polynomial pooler**, **multidimensional diagonal covariance penalty**, and **G-computation emulator** into the time-varying module to establish our pipeline as a state-of-the-art framework for survival meta-analysis."
