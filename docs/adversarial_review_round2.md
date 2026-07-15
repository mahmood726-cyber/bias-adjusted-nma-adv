> **Historical review artifact - not validation evidence.** This document is a critique/planning transcript only. It is not source extraction evidence, not reference-software parity evidence, not certification evidence, not clinical guidance, and not proof of superiority. Treat all numerical, clinical, guideline, and publication-status statements below as hypotheses unless they are backed by machine-verifiable artifacts in this repository.
# Hardcore Methodological Review (Round 2): The Refined NMA Engine Under Scrutiny

This document registers the transcript of the second-round multiperson adversarial review of the advanced NMA pooler, following the implementation of Kenward-Roger covariance corrections, Sweeting's adaptive corrections, directional priors, and weight sensitivity bounds.

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## Round 1: First-Order Kenward-Roger Variance Corrections

**Dr. Fiona Vance (Frequentist):**
> "While I congratulate you on implementing the first-order multivariate covariance correction ($\Sigma = \Phi + \sum \text{Cov}(\hat{\tau}_g^2, \hat{\tau}_h^2) \frac{\partial \Phi}{\partial \tau_g^2 \partial \tau_h^2}$) to address the plug-in fallacy, you have only corrected the *covariance matrix*. You have not corrected the **degrees of freedom** of the fixed effects. Standard Kenward-Roger or Satterthwaite adjustments derive a study-by-study degrees-of-freedom correction that reduces the effective $df$ under small samples. By still using the simple $n_{\text{studies}} - p$ or $n_{\text{contrasts}} - p$ for HKSJ critical values, your confidence intervals will still slightly under-cover in ultra-small samples ($k < 10$)."

**Dr. Benjamin MCMC (Bayesian):**
> "Fiona is correct. Furthermore, your estimation of the covariance of the heterogeneity parameters $\text{Cov}(\hat{\tau}_g^2, \hat{\tau}_h^2)$ relies on the expected Fisher Information from the REML projection matrix. When study counts are small, the true sampling distribution of $\hat{\tau}^2$ is heavily skewed and far from normal. Treating the Fisher Information inverse as the variance of $\hat{\tau}^2$ assumes asymptotic normality of the estimator, which fails when $\hat{\tau}^2$ lies close to the boundary (0.0). Only MCMC sampler algorithms can capture the true, skewed joint posterior of the heterogeneities."

---

## Round 2: Sweeting's Adaptive Continuity Correction

**Dr. Cynthia Registry (Data Engineer):**
> "Implementing Sweeting’s adaptive correction is a major improvement over adding $0.5$ blindly. However, you fixed the scaling constant $k = 1.0$. Why 1.0? In highly sparse networks with extremely rare adverse events, a scaling factor of $1.0$ can still over-correct and bias the treatment effect estimates toward the null. Cochrane methodologists often recommend comparing results at $k = 0.5$ or $k = 0.1$ as a sensitivity sweep."

**Dr. Fiona Vance (Frequentist):**
> "More fundamentally, it is still a continuity correction. It forces a normal approximation on what is naturally a discrete binomial process. If a trial has zero events in *both* arms, your log-odds ratio calculation still discards that study because the odds ratio is undefined. In a true exact binomial likelihood model, a 0/0 study still contributes information about the baseline event rate. By relying on any continuity correction, you are still throwing away information."

---

## Round 3: Fixed Directional Prior Means ($\mu$)

**Dr. Benjamin MCMC (Bayesian):**
> "Your new generalized Ridge estimator incorporates the prior mean offset $\mu$ to account for directional publication bias. But this $\mu$ is fixed by the user (e.g. `bias_prior_mean = 0.2`). This is a 'static' adjustment. If the true under-reporting bias in the literature is 0.6, your model will still under-correct. To make this scientifically robust, the prior mean $\mu$ should be modeled hierarchically as a random variable drawn from a meta-epidemiological distribution whose hyperparameters are estimated from historical registry data."

---

## Round 4: Local vs. Global Weight Sensitivity Bounds

**Dr. Cynthia Registry (Data Engineer):**
> "Your perturbation analysis refits the model 30 times under $\pm 10\%$ uniform noise to produce `weight_sensitivity_stds`. While this is computationally efficient, it is a local sensitivity analysis. Subjective risk-of-bias assessments are often categorical (e.g. shifting a trial from 'low risk' to 'high risk' changes $w_s$ from $1.0$ to $0.3$—a $70\%$ drop). A local $10\%$ noise sweep will not capture the impact of these discrete, high-leverage scoring boundaries. You should implement a **leave-one-out (LOO) sensitivity analysis** or a global boundary sweep to report true structural stability."

---

## Summary of Vulnerabilities & Mitigation Paths

| Vulnerability | Impact | Mitigation Path |
|---|---|---|
| **First-Order KR Only** | Critical values slightly too narrow in $k < 10$ | Implement the full Kenward-Roger degrees-of-freedom approximation to adjust the student-t denominator. |
| **Fixed Sweeting Scaling** | Potential over-correction toward the null | Allow the Sweeting scaling constant $k$ to be customized, and run a automated sensitivity sweep ($k \in [0.1, 1.0]$). |
| **Static Prior Mean** | Under-correction if prior mean is misspecified | Implement a hierarchical prior where the bias mean is estimated from historical meta-epidemiological databases. |
| **Local Perturbation Sweep** | Ignores categorical risk-of-bias boundary shifts | Implement global leave-one-out (LOO) sweeps and categorical RoB boundary tests. |
