> **Historical review artifact - not validation evidence.** This document is a critique/planning transcript only. It is not source extraction evidence, not reference-software parity evidence, not certification evidence, not clinical guidance, and not proof of superiority. Treat all numerical, clinical, guideline, and publication-status statements below as hypotheses unless they are backed by machine-verifiable artifacts in this repository.
# Hardcore Methodological Review (Round 34): NMA Engine Technical Specification

This document registers the transcript of the thirty-fourth-round joint adversarial and spec review, compiling the final technical specification of all active modules in our next-generation evidence synthesis engine.

---

## 1. Engine Architecture & Active Methodological Specification

Our engine contains the following core active modules:

### 1.1. Multinomial Competing-Risk GLMM (`src/bias_nma_adv/multinomial.py`)
*   **Purpose:** Fits exact likelihood multinomial logistic models for multi-category clinical endpoints (e.g. CV Death vs. Non-CV Death vs. Heart Failure Hospitalization).
*   **Mechanism:** Implements a stable softmax projection over $J$ competing event classes:
    $$p_j = \frac{\exp(x \beta_j)}{\sum_{k=1}^J \exp(x \beta_k)}$$
    Weights are optimized using gradient descent on the multinomial log-likelihood, preserving endpoint correlations.

### 1.2. Collaborative TMLE (C-TMLE) (`src/bias_nma_adv/ctmle.py`)
*   **Purpose:** Double-robust causal inference estimator for risk differences that avoids the variance inflation caused by extreme propensity score weights.
*   **Mechanism:** Forward-selects propensity score covariates step-by-step to minimize the mean squared error (MSE) of the fluctuated outcome regression $Q^*(A, W)$. Fluctuation updates use:
    $$H(A, W) = \frac{A}{g(W)} - \frac{1 - A}{1 - g(W)}$$
    $$\text{logit}(Q^*(A, W)) = \text{logit}(Q(A, W)) + \epsilon H(A, W)$$

### 1.3. No-U-Turn Sampler (NUTS) (`src/bias_nma_adv/nuts.py`)
*   **Purpose:** Efficiently samples posterior distributions of treatment parameters and random-effects variance ($\tau^2$) without manual step parameter tuning.
*   **Mechanism:** Dynamically builds a recursive binary leapfrog trajectory tree forward and backward in time, stopping automatically when the trajectory begins to double back on itself (the No-U-Turn criterion).

### 1.4. Conditional GAN Patient Simulator (`src/bias_nma_adv/gan.py`)
*   **Purpose:** Generates high-fidelity synthetic patient covariate matrices.
*   **Mechanism:** Generator and discriminator are trained concurrently in pure NumPy. Incorporates conditional disease indicators (e.g. LVEF severe vs. mild status) as auxiliary inputs to capture joint physiological correlations.

### 1.5. Symbolic Hazard Regressor (`src/bias_nma_adv/symbolic.py`)
*   **Purpose:** Discovers closed-form, non-proportional hazard functions over time.
*   **Mechanism:** Evaluates a basis set of mathematical operators ($1, t, \sqrt{t}, \ln(t), \exp(-t)$) to fit time-varying survival hazards, avoiding human-induced parametric misspecification bias.

### 1.6. Bayesian Model Averaging (BMA) (`src/bias_nma_adv/bma.py`)
*   **Purpose:** Pools treatment effects and rankings across consistent and inconsistent network sub-graphs.
*   **Mechanism:** Approximates posterior model probabilities using BIC weights:
    $$P(M_k|D) = \frac{\exp(-0.5 \cdot \text{BIC}_k)}{\sum_j \exp(-0.5 \cdot \text{BIC}_j)}$$
    Incorporates model-specification uncertainty directly into the pooled variance.

### 1.7. Registry-Based Bias Auditors (`src/bias_nma_adv/publication_bias.py`, `src/bias_nma_adv/sponsor_bias.py`)
*   **Purpose:** Audits and down-weights trials based on registry discrepancies, industry sponsorship, and participant flow dropouts.
*   **Mechanism:**
    *   **Unpublished Trial Ratio (UTR):** Quantifies registry-to-publication gaps.
    *   **Outcome Switching Bias Score (OSBS):** Detects primary outcome switching.
    *   **Sponsorship Bias Score (SBS):** Down-weights industry-sponsored trials by a sponsorship penalty factor (such as 0.80).
    *   **Loss-to-Follow-Up Attrition Ratio (LAR):** Down-weights trials with $>5\%$ dropouts.
