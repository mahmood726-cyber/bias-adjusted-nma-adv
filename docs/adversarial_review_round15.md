# Hardcore Methodological Review (Round 15): Methodological Pruning and Pipeline Consolidation

This document registers the transcript of the fifteenth-round multiperson adversarial review, systematically evaluating which advanced features add real clinical value and which are over-engineered or statistically unstable and should be pruned from the active pipeline.

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## 1. The Methodological Pruning Decision Table

The panel evaluates the advanced methods to identify their clinical utility, mathematical stability, and data requirements:

| Method / Model | Verdict | Clinical & Statistical Rationale |
|---|---|---|
| **Fractional Polynomials (NPH)** | **KEEP (High Value)** | Essential. Proportional hazards break down in almost all modern cardiology trials with delayed effects. This is the gold standard for Health Technology Assessments (like NICE). |
| **Multidimensional Covariance Penalty** | **KEEP (High Value)** | Essential. Inflating the diagonal parameters ($I_k \times v_{bias}$) propagates protocol bias uncertainty over the entire survival curve timeline without adding false correlation assumptions. |
| **Bayesian Quantitative Bias Analysis (QBA)** | **KEEP (High Value)** | Highly stable. Replacing a static penalty factor with a prior probability distribution ($\beta_{bias} \sim \mathcal{N}(\mu_{dropout}, \sigma_{dropout}^2)$) yields honest, probabilistic credible intervals. |
| **Target Trial Emulation (G-Computation)** | **KEEP (High Value)** | Highly intuitive. Translates complex time-varying hazards into a concrete Marginal Risk Difference (e.g. absolute difference in life expectancy), which clinicians understand. |
| **Clayton Copula joint Likelihood** | **KEEP (Secondary Utility)** | Useful only for linking safety (adverse events) and efficacy (survival) across separate trials without requiring patient-level IPD. Keep as an optional diagnostic module. |
| **Variational Autoencoder (VAE) Simulator** | **KEEP (Secondary Utility)** | Keep strictly as a covariate generator to feed G-computation. It must be normalized to avoid gradient explosion. |
| **Causal Transportability (IPSW)** | **KEEP (Theoretical Option)** | Promising but highly sensitive to data missingness. If trials do not report identical baseline covariate categories, transportability weights become highly unstable. Keep with warning flags. |
| **GCN Treatment Embeddings (GNN)** | **DISCARD (Over-engineered)** | **Prune/Remove.** Running a Graph Convolutional Network on a network with only 5 to 6 nodes is 'deep learning theater.' GCNs are designed for massive graphs; on small NMA networks, they introduce parameter instability and add zero precision over standard centrality-based topological priors. |

---

## The Panel's Consensual Discussion

**Dr. Fiona Vance:**
> "The GCN treatment embedding model is the clearest candidate for removal. A network meta-analysis of heart failure or hyperlipidemia typically has 3 to 6 treatments. Applying deep learning message-passing to a 6-node graph is mathematically absurd. It introduces weight initialization noise and ReLU dead-node issues for zero practical benefit. We should stick to our standard, closed-form topological regularization based on degree centrality."

**Dr. Benjamin MCMC:**
> "I agree. The GCN is over-engineering. On the other hand, **Bayesian QBA** and **Fractional Polynomials** are extremely valuable. They directly address the real-world issues of run-in exclusions and time-varying treatment effects, producing statistically honest confidence ribbons. The VAE is also worth keeping, but only as a helper to generate synthetic cohorts for G-computation."

**Dr. Cynthia Registry:**
> "CIMA (Causal Transportability) should remain in the codebase but flagged with a warning. If ClinicalTrials.gov results tables have missing covariate data (e.g., one trial reports mean eGFR while another reports categorical eGFR), the IPSW weights cannot be computed reliably. We must alert the user to this data reporting limitation."

---

## Action Plan: Pruning the GNN Module

We will:
1.  Remove GNN usage from `run_all_cardio_nmas.py`.
2.  Deprecate and remove `src/bias_nma_adv/gnn.py` from the active codebase.
3.  Ensure the test suite remains 100% green without the GNN model.
