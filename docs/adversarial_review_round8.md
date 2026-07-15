# Hardcore Methodological Review (Round 8): The Wasserstein Distance vs. Guyot in Survival Analysis

This document registers the transcript of the eighth-round multiperson adversarial review, addressing the question: *Is the Wasserstein distance itself better than the Guyot algorithm, and is it only useful when survival curves cross?*

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## 1. Defining the Roles: Guyot (Reconstructor) vs. Wasserstein (Metric)

**Dr. Cynthia Registry (Data Engineer):**
> "We must first clarify a common terminological confusion. The **Guyot algorithm** and the **Wasserstein distance** are not direct competitors because they perform entirely different tasks:
> *   **Guyot is an IPD Reconstructor:** It takes aggregate survival probabilities and risk tables, and outputs patient-level records ($t_i, d_i$). It does not calculate treatment effects; it merely prepares the data.
> *   **Wasserstein is a Distance Metric:** It calculates the mathematical distance between two survival distributions. The $L_1$ Wasserstein distance between two survival curves $S_1(t)$ and $S_2(t)$ is the integrated absolute difference between them:
> 
> W_1(S_1, S_2) = \int_{0}^{\tau} |S_1(t) - S_2(t)| dt
> 
> Therefore, you use Guyot to reconstruct the patient data, and you can use the Wasserstein distance to evaluate the mismatch between the reconstructed curve and the original digitized curve, or to compare treatment arms directly."

---

## 2. The Crossing Curves Dilemma (Proportional Hazards Violations)

**Dr. Fiona Vance (Frequentist):**
> "The user is correct: **Wasserstein becomes highly valuable when survival curves cross**. 
> 
> Under standard Proportional Hazards (PH), the Hazard Ratio (HR) is constant, and fitting a Cox model on Guyot-reconstructed IPD is the standard approach. But in modern immuno-oncology (e.g. checkpoint inhibitors vs. chemotherapy) and cardiovascular device trials, **curves frequently cross**. Delayed treatment effects cause chemotherapy curves to drop early, while immunotherapy curves flatten out later, leading to a crossing point.
> 
> When curves cross, the proportional hazards assumption is violated. The pooled HR becomes a mathematically unstable average that underestimates the long-term treatment benefit."

**Dr. Benjamin MCMC (Bayesian):**
> "Exactly. Furthermore, when curves cross, the standard **Restricted Mean Survival Time (RMST) difference** (which is the net area between the curves: $\int (S_1(t) - S_2(t)) dt$) can equal **zero** because the positive area before the crossing point cancels out the negative area after it. 
> 
> The **Wasserstein distance** resolves this by integrating the *absolute* difference $|S_1(t) - S_2(t)|$. It captures the total magnitude of the treatment difference and time-varying kinetics without cancellation. In this sense, Wasserstein is indeed mathematically superior when curves cross."

---

## 3. Clinical Utility & Interpretation limits

**Dr. Cynthia Registry (Data Engineer):**
> "While Wasserstein is mathematically elegant for crossing curves, it has a major weakness in clinical practice: **it lacks clinical interpretability**. 
> 
> A doctor cannot easily interpret a Wasserstein distance of `0.85 days-percent`. In contrast, RMST represents the difference in life expectancy (e.g., 'an average of 3.2 months of additional life gained up to 2 years'), which is highly intuitive. 
> 
> Therefore, if curves cross, clinical guidelines prefer reporting **RMST differences over a specific time horizon** or **time-varying Hazard Ratios** (e.g., HR before 6 months vs. HR after 6 months) rather than a raw Wasserstein distance metric."

---

## Comparative Summary: Guyot vs. Wasserstein vs. RMST

| Method / Metric | Mathematical Definition | Best Used When | Clinical Interpretability |
|---|---|---|---|
| **Guyot IPD** | Iterative event-censoring reconstruction. | Reconstructing individual patient records for regression. | High (enables standard Cox / survival modeling). |
| **Wasserstein Distance** | $\int \|S_1(t) - S_2(t)\| dt$ | Capturing total distributional mismatch (especially for crossing curves). | Low (abstract distance unit). |
| **RMST Difference** | $\int (S_1(t) - S_2(t)) dt$ | Quantifying difference in life expectancy up to time horizon $\tau$. | High ('months of life gained'). |
