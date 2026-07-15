<!-- sentinel:skip-file -->
# Technical Specification: Advanced Bias-Aware Network Meta-Analysis Platform

## 1. Scope and Design Principles

The platform performs Bayesian and frequentist network meta-analysis (NMA) of randomized and non-randomized comparative studies. The core engines preserve randomization, within-study treatment contrasts, correlations from multi-arm trials, and between-study heterogeneity.

Bias-related information is incorporated through transparent sensitivity analyses or explicitly specified probabilistic bias models. Study weights are not modified using arbitrary sponsorship, attrition or registry penalty constants.

All advanced modules are classified as:

1.  **Core NMA Estimators:** Bayesian and frequentist engines, including component, dose-response, multivariate, and survival models.
2.  **Optional Bias and Sensitivity Models:** Probabilistic bias models and registry-informed analyses.
3.  **IPD Extensions:** Multilevel Network Meta-Regression (ML-NMR) and causal individual-participant-data (IPD) adjustment.
4.  **Simulation and Validation Tools:** Benchmarks, simulation matrices, and privacy-preserving data generators.

---

## 2. Estimands, Evidence Designs and Analysis Governance

### 2.1 Estimand Specification
Before model fitting, each analysis must define the target estimand:
*   **Population:** Target clinical population baseline parameters.
*   **Treatment Strategies:** Set of eligible comparator interventions.
*   **Outcome:** Outcomes must have prespecified definitions, time horizons and estimands. Mutually exclusive and collectively exhaustive categories are required only for multinomial clinical-state models.
*   **Follow-Up Time:** Prespecified landmarks or continuous horizons.
*   **Effect Measure:** Relative risk, odds ratio, risk difference, or hazard ratio.
*   **Intercurrent Events:** Strategy for handling treatment discontinuation, crossovers, or rescue therapies (e.g. treatment-policy, hypothetical, composite, or while-on-treatment).
*   **Analysis Interpretation:** Intention-to-treat (ITT) versus per-protocol (PP).

Results from materially different estimands are not pooled without an explicit transformation or hierarchical model.

### 2.2 Evidence-Design Separation
Randomized and non-randomized studies are analysed separately by default. Joint synthesis is permitted only through a prespecified design-adjusted model containing design-specific parameters or bias distributions. The engine must report:
1.  randomized-evidence-only results;
2.  non-randomized-evidence-only results;
3.  combined design-adjusted results;
4.  sensitivity to the assumed bias parameters.

Synthetic observations are never classified as clinical evidence.

### 2.3 Prior Specification
Every Bayesian model must declare priors for:
*   treatment effects;
*   study baselines;
*   heterogeneity parameters;
*   correlation parameters;
*   inconsistency parameters;
*   bias parameters;
*   time-varying or spline coefficients.

The engine performs prior predictive checks and reports sensitivity to clinically plausible alternative priors.

---

## 3. Generalized Likelihood and Link Framework

The platform supports a generalized linear model (GLM) framework. The outcome likelihoods, links, and principal estimands are defined as follows:

| Outcome | Likelihood | Default Link | Principal Estimand |
| :--- | :--- | :--- | :--- |
| **Binary** | Binomial | Logit, log, or identity | OR, RR, or RD |
| **Counts** | Poisson or negative binomial | Log | Rate ratio |
| **Continuous** | Normal or Student-$t$ | Identity | Mean Difference (MD) or SMD |
| **Ordinal** | Multinomial | Cumulative logit | Common or category-specific OR |
| **Multinomial states** | Multinomial | Multinomial logit | State probabilities |
| **Time-to-event** | Flexible survival likelihood | Hazard / cumulative hazard | HR, RMST, or survival difference |

---

## 4. Frequentist NMA Engine

**Module:** `src/bias_nma/inference/frequentist_backend.py`

### 4.1 Estimation Methods
The frequentist engine implements:
*   **Contrast-Based Weighted Least Squares:** Solves the linear network equations using study-specific estimates and their covariance matrices.
*   **Graph-Theoretical NMA:** Analyzes the network structure using electrical network analogies (random walks, Laplacian matrices).
*   **Common & Random Effects:** Estimates between-study heterogeneity ($\tau^2$) using DerSimonian–Laird, Paule–Mandel, or Restricted Maximum Likelihood (REML).

### 4.2 Multi-Arm Trial Covariance
Multi-arm trial correlations are preserved by constructing the block-diagonal covariance matrix ($V$) of relative effects and solving:

$$
\hat{\boldsymbol{d}} = \left( X^T V^{-1} X \right)^{-1} X^T V^{-1} \boldsymbol{y}.
$$

### 4.3 Diagnostics
*   **Design-by-Treatment Interaction:** Evaluates global inconsistency across designs.
*   **Node Splitting:** Direct–indirect evidence comparisons for direct comparisons.
*   **Influence & Leverage Diagnostics:** Leverage values, hat matrix diagonals, and Cook's distance for detecting influential studies.
*   **Prediction Intervals:** Computes uncertainty intervals for a future study.

---

## 5. Multilevel Network Meta-Regression (ML-NMR)

**Module:** `src/bias_nma/models/mlnmr/`

### 5.1 Formulation
ML-NMR combines individual-participant data (IPD) and aggregate data (AgD). For participant $i$ in study $j$ on treatment $k$, the individual-level linear predictor is:

$$
\eta_{ijk} = \mu_j + d_k + \mathbf{x}_{ij}^T \boldsymbol{\beta} + \mathbf{x}_{ij}^T \boldsymbol{\gamma}_k,
$$

where:
*   $\boldsymbol{\beta}$ is the vector of prognostic covariate effects;
*   $\boldsymbol{\gamma}_k$ represents the treatment-covariate interaction vector.

### 5.2 Likelihood Integration
For aggregate studies where individual $\mathbf{x}_{ij}$ are unknown, the study-level aggregate likelihood contribution is obtained by integrating the individual-level model over the study-specific joint covariate distribution $F_j(\mathbf{x})$:

$$
p_{jk} = \int g^{-1}(\eta_{jk}(\mathbf{x})) \, dF_j(\mathbf{x}).
$$

This integration is performed numerically using quasi-Monte Carlo integration or Gaussian quadrature.

### 5.3 Outputs and Diagnostics
*   **Conditional and Average Effects:** Reports treatment effects for specific patient profiles and standardized target-population averages.
*   **Covariate Overlap:** Computes distance metrics (e.g., Mahalanobis distance) to detect covariate extrapolation.
*   **Numerical Integration Quality:** Monitors the error of the integration approximation.

---

## 6. Component Network Meta-Analysis (CNMA)

**Module:** `src/bias_nma/models/component/`

### 6.1 Additive and Interaction Models
For multi-component interventions, treatment effects ($d_t$) are decomposed into their constituent components ($C$):

$$
d_{A+B} = d_A + d_B + \gamma_{AB},
$$

where:
*   $d_A, d_B$ are the main additive effects of components $A$ and $B$;
*   $\gamma_{AB}$ is the interaction effect (set to zero in additive models).

### 6.2 Structural Diagnostics
*   **Identifiability Checks:** Verifies that component parameters are estimable from the treatment network design matrix.
*   **Connectedness:** Checks network connectivity at both the treatment level and the constituent component level.
*   **Goodness of Fit:** Compares residual deviance between CNMA and full-treatment NMA models to justify additive assumptions.

---

## 7. Dose-Response Network Meta-Analysis

**Module:** `src/bias_nma/models/dose_response/`

### 7.1 Models
The dose-response module models continuous dose variables using:
*   **Nonlinear Functions:** Emax, Hill, exponential, or fractional polynomial functions.
*   **Spline Functions:** Restricted cubic splines and monotonic splines to capture complex dose curves.
*   **Hierarchical Curves:** Class-level dose-response curves linking agents with similar mechanisms.

### 7.2 Constraints & Extrapolations
*   **Dose-Equivalence Assumptions:** Standardizes doses using predefined equivalence coefficients.
*   **Prediction:** Estimates effects at unobserved doses.
*   **Extrapolation Warnings:** Flags predictions outside the range of observed trial doses.

---

## 8. Genuine Multivariate Network Meta-Analysis

**Module:** `src/bias_nma/models/multivariate/`

### 8.1 Model Structure
For correlated, non-exclusive outcomes (e.g., mortality, hospitalization, and adverse events), the study-level relative effects $\boldsymbol{\theta}_i$ are modeled jointly:

$$
\boldsymbol{\theta}_i \sim \operatorname{MVN}\left(\boldsymbol{\Delta}_i, \boldsymbol{S}_i + \boldsymbol{\Sigma}\right),
$$

where:
*   $\boldsymbol{S}_i$ is the within-study covariance matrix (estimated or imputed);
*   $\boldsymbol{\Sigma}$ is the between-study covariance matrix.

### 8.2 Missing Outcomes & Borrowing of Strength
*   **Missing Outcomes:** Implements joint imputation of outcomes missing in specific trials.
*   **Borrow-of-Strength Diagnostics:** Computes the change in precision (shrinkage) across outcomes to evaluate the impact of multivariate modeling.

---

## 9. Rare-Event and Zero-Event NMA

**Module:** `src/bias_nma/models/standard_nma/`

### 9.1 Likelihoods
*   **Exact Binomial Likelihood:** Evaluates events without continuity corrections ($+0.5$).
*   **Mantel-Haenszel NMA:** Non-parametric estimation of fixed effects for sparse networks.
*   **Non-Central Hypergeometric Models:** Fits exact conditional likelihoods for multi-arm sparse networks.
*   **Penalized Regression:** Applies Firth's penalization to logistic models.

### 9.2 Zero-Event Policy
*   Studies with zero events in all arms are excluded from relative effect models but retained in absolute baseline risk estimations.
*   Studies with zero events in one arm are modeled using exact likelihoods to avoid bias introduced by continuity corrections.

---

## 10. Disconnected-Network Policy

The platform enforces strict rules for networks containing disconnected components:
1.  **Pre-Fit Detection:** Scans the adjacency matrix of the treatment network.
2.  **Default Action:** Halts standard relative-effect analysis and fits separate subnetwork models.
3.  **Reconnection Rule:** Reconnection is permitted only under:
    *   Informative priors on baseline risks or common components;
    *   Component NMA or dose-response models where shared components/dose curves bridge the gap.
4.  **Labeling:** Outputs for reconnected nodes are flagged as "structurally connected" (as opposed to empirically connected).

---

## 11. Time-to-Event and Competing-Risk Extension

**Module:** `src/bias_nma/survival/flexible_hazards.py`

Where individual event times or sufficiently detailed interval data are available, the engine may fit piecewise exponential models, restricted cubic splines, M-splines, Royston-Parmar models, fractional polynomials, treatment-specific time-varying coefficients, and competing-risk models.

Non-proportional hazards are evaluated using splines, fractional polynomials, or treatment-specific time-varying coefficients. Hazards are constrained to remain non-negative.

---

## 12. Bayesian Inference Engine

**Module:** `src/bias_nma/inference/stan_backend.py`

### 15.1 Stan Integration
CmdStanPy serves as the production inference runner. All Bayesian models are compiled to C++ code using CmdStan.
*   **Parameterizations:** Uses non-centered parameterizations for heterogeneity variables by default to avoid funnel geometries.
*   **Ranking:** Ranking metrics (SUCRA, rank probabilities) are calculated inside the `generated quantities` block of the Stan program to preserve posterior correlation structures.

---

## 13. Probabilistic Bias Analysis

**Module:** `src/bias_nma/models/bias_adjustment/`

Bias is represented as an explicit additive or multiplicative parameter rather than an arbitrary change in study weight.

For study ($i$):

$$
\theta_i^{\mathrm{observed}} = \theta_i^{\mathrm{true}}+b_i,
$$

where ($b_i$) is a bias parameter whose prior distribution is determined by the relevant bias domain, empirical evidence or prior distributions from expert elicitation.

Analyses include:

1. primary analysis without numerical bias correction;
2. exclusion of studies at high risk of bias;
3. domain-specific probabilistic bias adjustment;
4. prior sensitivity analyses;
5. worst-case and tipping-point analyses.

The direction and magnitude of every bias prior are reported.

---

## 14. Registry and Missing-Evidence Auditing

**Modules:**

* `src/bias_nma/registry/`
* `src/bias_nma/models/bias_adjustment/publication_bias.py`

### Registry audit outputs
The registry auditor identifies registered but unpublished studies, discrepancies in planned and reported outcomes, sample sizes, and delayed reporting. These are registered as qualitative metadata flags, not weight adjustments.

### Missing-evidence analyses
Where appropriate, the engine supports selection models, robust Bayesian meta-analysis, pattern-mixture sensitivity analyses, and ROB-MEN-compatible evidence summaries.

---

## 15. Collaborative TMLE IPD Extension

**Module:** `src/bias_nma/models/bias_adjustment/ctmle.py`

### Scope
C-TMLE is an optional individual-participant-data module for observational comparisons or randomized studies requiring adjustment for informative missingness, non-adherence or treatment crossover. It is not applied automatically to aggregate randomized-trial NMA.

### Diagnostics
*   **Propensity Weights:** Calculates the propensity cohort size induced by the estimated weights or clever covariate.
*   **Influence Curves:** Computes influence-curve variance and standard error.
*   **Truncation:** Monitors the proportion of propensity scores truncated.
*   **Cross-Validation:** Monitors cross-validated nuisance-model loss.

---

## 16. Model Averaging

**Module:** `src/bias_nma/models/bias_adjustment/bma.py`

Model averaging may be used for common versus random effects, alternative heterogeneity priors, proportional versus non-proportional hazards, and selected consistency and inconsistency structures.

Where BIC-derived weights are used:

$$
w_k= \frac{\exp(-\tfrac12(BIC_k-BIC_{\min}))} {\sum_j\exp(-\tfrac12(BIC_j-BIC_{\min}))},
$$

they are labelled approximate information-criterion weights.

---

## 17. Synthetic-Data Simulation

**Package:** `bias_nma_simulation/` (Isolated)

The synthetic-data simulation module is restricted to software testing, simulation studies, demonstration datasets, and privacy-preserving methodological development. Synthetic participants are never combined with real participants to artificially inflate sample size or narrow the treatment-contrast variance.

---

## 18. Governance, Model Certification and Evidence Certainty

### 18.1 Model Acceptance Criteria
A model is classified as successfully fitted only when:
*   **split-$\hat{R}$:** Rank-normalized split-$\hat{R} < 1.05$ for all parameters.
*   **MCMC ESS Metrics:** Bulk and tail ESS statistics meet prespecified thresholds (minimum $100$ draws per chain).
*   **Monte Carlo Error:** MCSE is $< 5\%$ of the posterior standard deviation.
*   **HMC warnings:** No unresolved divergent transitions or tree-depth saturation events remain.
*   **Checks:** Posterior predictive checks do not show major systematic misfit.

### 18.2 Certainty of Evidence and Contribution Matrices
The platform generates evidence certainty calculations compatible with CINeMA and ROB-MEN.
*   **Study Contribution Matrix:** Computes the contribution matrix ($H$) showing the percentage contribution of each study and direct comparison to each network effect estimate:

$$
H = \left( X^T V^{-1} X \right)^{-1} X^T V^{-1}.
$$

*   **Risk of Bias Propagation:** Projects risk-of-bias (RoB) categories (low, moderate, high) across the network using the contribution matrix.
*   **ROB-MEN Pairwise Tables:** Assesses publication bias and outcome reporting bias based on registry discrepancies.
*   **Equivalence Limits:** Evaluates confidence intervals against clinically important equivalence margins.

### 18.3 Model Certification Levels
Every model fitted by the platform receives a certification level:

| Level | Meaning |
| :--- | :--- |
| **Experimental** | Implemented but not validated. |
| **Numerically Verified** | Gradients and likelihoods agree with independent calculations. |
| **Reference Matched** | Reproduces established software (e.g., `netmeta`, `multinma`) on benchmark datasets. |
| **Simulation Validated** | Acceptable bias, coverage, and calibration across simulation matrices. |
| **Externally Reproduced** | Independently tested by an external group. |
| **Production Certified** | Suitable for clinical or HTA analyses. |

The user interface blocks clinical reporting for any model below the **Production Certified** level.

---

## 19. Global Benchmark and Validation Programme

For every model certified for production, the platform executes a benchmark programme:
1.  **Datasets:** Reproduce at least $20$ published reference clinical trials/networks.
2.  **Inference Matching:** Compare Bayesian outcomes against `multinma` and CmdStan reference models.
3.  **Frequentist Matching:** Compare frequentist outcomes against `netmeta`.
4.  **Dose-Response Matching:** Compare dose-response models against `MBNMAdose`.
5.  **Cross-Design Matching:** Compare cross-design synthesis against `crossnma`.
6.  **SBC:** Run simulation-based calibration (SBC) to verify posterior distribution shape.
7.  **Recovery:** Verify parameter recovery, bias, RMSE, and interval coverage.
8.  **Ranks:** Assess ranking calibration using calibration curves.
9.  **Stress Testing:** Test parameter estimation under sparse, highly heterogeneous, and disconnected configurations.
10. **Reproducibility:** Publish all benchmark datasets, scripts, and numerical tolerances.

---

## 20. Reproducibility, Reporting, and Architecture

### 20.1 Reproducible Analysis Bundle
Every execution yields a reproducible bundle containing:
*   Clean, un-redacted model code (compiled `.stan` files).
*   Cryptographic hash of the input dataset.
*   Priors, configuration parameters, and seeds.
*   Reviewer overrides and qualitative risk assessments.
*   Convergence diagnostics and contribution tables.

### 20.2 Directory Structure
The platform codebase is structured as:

```text
bias_nma/
├── data/
│   ├── schema.py
│   ├── validation.py
│   └── estimands.py
├── models/
│   ├── standard_nma/
│   ├── multinomial/
│   ├── multivariate/
│   ├── survival/
│   ├── component/
│   ├── dose_response/
│   ├── mlnmr/
│   ├── cross_design/
│   └── bias_adjustment/
├── inference/
│   ├── stan_backend.py
│   ├── frequentist_backend.py
│   └── diagnostics.py
├── evidence_quality/
│   ├── rob2.py
│   ├── robmen.py
│   ├── cinema.py
│   └── contributions.py
├── registry/
├── reporting/
├── validation/
└── experimental/
```
