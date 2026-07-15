# Hardcore Methodological Review (Round 5): The Hybrid IPD-AD Survival Dilemma

This document registers the transcript of the fifth-round multiperson adversarial review, addressing the prevalence of survival curves in open-access papers and the mathematical validity of running NMAs when only a fraction of trials have extractable Kaplan-Meier curves.

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## Dilemma 1: How many survival curves actually exist in the public domain?

**Dr. Cynthia Registry (Data Engineer):**
> "The availability of survival curves is highly domain-specific:
> *   **Oncology & Cardiology:** KM curves are the universal gold standard. For primary endpoints (Overall Survival, Progression-Free Survival, Time-to-Event MACE), almost **85% to 90%** of phase III trials published in open-access journals or supplementary appendix files contain extractable KM curves.
> *   **Rare Events & Adverse Events:** For safety/adverse event outcomes (e.g. stroke, bleeding, infection rates), KM curves are almost never published. Instead, these are reported as aggregate tabular event counts ($r/n$) in ct.gov registries.
> *   **Other Diseases:** In immunology, infectious diseases, or psychiatry, time-to-event curves are rare; continuous scale scores or binary responder rates dominate the literature."

---

## Dilemma 2: Can we still use KM curves if they are only available for a few trials?

**Dr. Fiona Vance (Frequentist):**
> "Yes, absolutely. You do **not** need survival curves for every study to exploit their power. If only a few studies (e.g., 3 out of 15 trials) have KM curves, we fit a **Hybrid Joint Likelihood Model** that combines arm-level reconstructed IPD with study-level aggregate data (AD):
> 
> \ln L_{\text{total}}(\theta) = \sum_{s \in \text{IPD}} \ln L_{\text{IPD}, s}(Y_{\text{IPD}, s} | \theta) + \sum_{s' \in \text{AD}} \ln L_{\text{AD}, s'}(Y_{\text{AD}, s'} | \theta)
> 
> Where:
> *   The studies with KM curves ($s \in \text{IPD}$) are modeled using the exact patient-level Cox or parametric survival likelihood.
> *   The studies with only aggregate hazard ratios ($s' \in \text{AD}$) are modeled using a normal likelihood on the reported $\ln(\text{HR})$ and standard errors.
> 
> Both pieces of the likelihood are linked to the same underlying treatment contrast parameters $\beta$. The IPD studies 'anchor' the baseline hazard shape, while the AD studies widen the treatment network and improve precision."

**Dr. Benjamin MCMC (Bayesian):**
> "Fiona's hybrid model is exactly how we solve the ecological fallacy in practice. Even a **single** reconstructed IPD trial in the network allows us to estimate patient-level covariate interactions ($\beta_{\text{cov}}$). Once that interaction coefficient is estimated from the IPD trial, it can be applied to adjust for population differences across the aggregate (AD) trials in the network regression. 
> 
> This means that having 'only a few' survival curves is not a limitation—it is a massive opportunity to calibrate and de-bias the rest of the aggregate trial network."

---

## Summary of Hybrid NMA Feasibility

| Proportion of IPD Curves | Feasible Model | Methodological Benefits |
|---|---|---|
| **High ( $\ge 75\%$ )** | Full IPD Network Meta-Analysis | Exact survival modeling, time-varying hazard ratios, full population adjustment. |
| **Moderate ( $25\% - 75\%$ )** | Hybrid Joint Likelihood NMA | Reconstructed IPD anchors baseline hazards; AD trials expand network connectivity. |
| **Low ( $< 25\%$ )** | Shared-Parameter Regression (ML-NMR) | A single IPD trial estimates the patient-level covariate coefficients, de-biasing the rest of the AD network. |
| **Zero ( $0\%$ )** | Standard Contrast-Based GLS | Fallback to standard aggregate-level NMA (continuous/binary outcomes only). |
