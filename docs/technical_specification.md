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

The platform is described as "potentially world-leading" until its models achieve reference matching, simulation validation, and independent external reproduction.

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

Student-$t$ continuous models are specified to target individual observations when patient-level data are available. For aggregate summaries, they require an appropriate sampling model based on reported means, standard errors and degrees of freedom rather than treating the summary statistics as raw observations.

---

## 4. Core Bayesian Network Meta-Analysis Model

**Module:** `src/bias_nma/models/standard_nma/`

### 4.1 Arm-Level Formulation
For study $i$ and arm $a$, the outcome $y_{ia}$ is modeled via:

$$
y_{ia}\sim f(\theta_{ia},\phi),
$$

$$
g(\theta_{ia})=\mu_i+\delta_{ia},
$$

where:
*   $f$ is the outcome probability distribution (e.g. Binomial, Normal);
*   $\phi$ represents auxiliary parameters (e.g. outcome variances);
*   $g$ is the link function (e.g. logit, log, identity);
*   $\mu_i$ is the study-specific baseline parameter.

For the reference arm ($a=1$), $\delta_{i1}=0$. For arm $a > 1$, the relative treatment effect against the reference treatment $t_{i1}$ is modeled under a random-effects framework:

$$
\delta_{ia}=d_{t_{ia}}-d_{t_{i1}}+u_{ia},
$$

where:
*   $d_k$ is the network treatment effect for treatment $k$ relative to the network reference;
*   $u_{ia}$ represents study-specific deviations (heterogeneity).

For a multi-arm study with $A_i$ arms, the vector of random effects $\boldsymbol{u}_i = (u_{i2}, \ldots, u_{iA_i})^T$ is modeled using a multivariate normal distribution:

$$
\boldsymbol{u}_i \sim \operatorname{MVN}\left(\boldsymbol{0}, \boldsymbol{\Sigma}_i(\tau)\right),
$$

where $\boldsymbol{\Sigma}_i(\tau)$ has diagonal elements $\tau^2$ and off-diagonal elements $\tau^2/2$, representing the induced correlation between treatment contrasts sharing the reference treatment.

### 4.2 Contrast-Level Formulation
For studies reporting contrast-level effects directly, the vector of observed effects $\boldsymbol{y}_i$ for study $i$ is modeled via:

$$
\boldsymbol{y}_i \sim \operatorname{MVN}\left(\boldsymbol{X}_i \boldsymbol{d}, \boldsymbol{S}_i + \boldsymbol{\Sigma}_i(\tau)\right),
$$

where:
*   $\boldsymbol{S}_i$ is the known within-study covariance matrix;
*   $\boldsymbol{d}$ is the vector of basic treatment parameters;
*   $\boldsymbol{X}_i$ is the study design matrix;
*   $\boldsymbol{\Sigma}_i(\tau)$ represents between-study heterogeneity.

### 4.3 Formulations and Constraints
*   **Arm-Based and Contrast-Based:** Supports both arm-level data inputs (binomial counts, normal means) and contrast-level trial inputs (log odds-ratios, mean differences).
*   **Common- and Random-Effects:** Random-effects modeling is used by default; common-effects models (constraining $\tau = 0$) are available as comparator specifications.
*   **Treatment Reference Constraints:** Constrains the reference treatment effect to zero ($d_{\text{reference}} = 0$) to ensure network parameter identifiability.
*   **Baseline-Risk Modeling:** Models baselines $\mu_i$ either as study-specific independent fixed effects or hierarchically from a common baseline risk distribution. Hierarchical baseline-risk modeling is used mainly for absolute-risk prediction. It must not create unsupported links between disconnected treatments or introduce treatment–baseline associations unless explicitly modeled.
*   **Consistency and Inconsistency:** Fits both consistency models (retaining the transitivity relation $d_{BC} = d_C - d_B$) and inconsistency models (such as design-by-treatment or unrelated mean effects models) to evaluate network coherence.

---

## 5. Multinomial Clinical-State NMA

**Module:** `src/bias_nma/models/multinomial/`

### 5.1 Likelihood and Logits
Models mutually exclusive and collectively exhaustive clinical categories $j = 1, \ldots, J$ measured at a common follow-up time:

$$
\mathbf{y}_{ia}\sim \operatorname{Multinomial}\left(n_{ia},\mathbf{p}_{ia}\right).
$$

Using category $J$ as the reference category, the multinomial logit relation is:

$$
\log\left(\frac{p_{iaj}}{p_{iaJ}}\right) = \eta_{iaj} = \mu_{ij} + \delta_{iaj}, \qquad j=1,\ldots,J-1,
$$

where:
*   $\mu_{ij}$ is the study-specific baseline for outcome category $j$;
*   $\delta_{iaj}$ is the relative effect of the treatment in arm $a$ versus reference arm $1$ for outcome category $j$, with $\delta_{i1j} = 0$.

### 5.2 Network Treatment Effects
The relative effects $\delta_{iaj}$ are connected to basic category-specific treatment parameters via:

$$
\delta_{iaj} = d_{t_{ia}, j} - d_{t_{i1}, j} + u_{iaj},
$$

where:
*   $d_{t, j}$ is the treatment effect relative to the reference treatment for clinical-state category $j$;
*   $\boldsymbol{u}_{ia} = (u_{ia1}, \ldots, u_{ia, J-1})^T$ represents cross-category random effects following the selected covariance structure.

### 5.3 Covariance Structures
Correlations between category-specific treatment effects $\delta_{iaj}$ are evaluated under one of three covariance specifications:
1.  **Independent:** Category-specific random effects are assumed uncorrelated across endpoints.
2.  **Unstructured Correlated:** Estimates a full covariance matrix $\boldsymbol{\Sigma}_{\text{multi}}$ representing the correlations between outcomes. This structure is subject to network size and identifiability limits.
3.  **Factor-Analytic Correlated:** Models covariance through a lower-dimensional factor structure to preserve stability in sparse networks.

---

## 6. Frequentist NMA Engine

**Module:** `src/bias_nma/inference/frequentist_backend.py`

### 6.1 Estimation Methods
The frequentist engine implements:
*   **Contrast-Based Weighted Least Squares:** Solves the linear network equations using study-specific estimates and their covariance matrices.
*   **Graph-Theoretical NMA:** Analyzes the network structure using electrical network analogies (random walks, Laplacian matrices).
*   **Common & Random Effects:** Estimates between-study heterogeneity ($\tau^2$). The DerSimonian–Laird estimator is available mainly for legacy compatibility, while Restricted Maximum Likelihood (REML) or Paule–Mandel estimation is preferred and used by default.

### 6.2 Multi-Arm Trial Covariance
Multi-arm trial correlations are preserved by constructing the block-diagonal covariance matrix ($V$) of relative effects and solving:

$$
\hat{\boldsymbol{d}} = \left( X^T V^{-1} X \right)^{-1} X^T V^{-1} \boldsymbol{y}.
$$

### 6.3 Diagnostics
*   **Design-by-Treatment Interaction:** Evaluates global inconsistency across designs.
*   **Node Splitting:** Direct–indirect evidence comparisons for direct comparisons.
*   **Influence & Leverage Diagnostics:** Leverage values, hat matrix diagonals, and Cook's distance for detecting influential studies.
*   **Prediction Intervals:** Computes uncertainty intervals for a future study.
*   **Sparse Event Handling:** Implements validated conditional-likelihood or penalized-likelihood methods for sparse binary outcomes.

---

## 7. Multilevel Network Meta-Regression (ML-NMR)

**Module:** `src/bias_nma/models/mlnmr/`

### 7.1 Formulation
ML-NMR combines individual-participant data (IPD) and aggregate data (AgD). For participant $i$ in study $j$ on treatment $k$, the individual-level linear predictor is:

$$
\eta_{ijk} = \mu_j + d_k + \mathbf{x}_{ij}^T \boldsymbol{\beta} + \mathbf{x}_{ij}^T \boldsymbol{\gamma}_k,
$$

where:
*   $\boldsymbol{\beta}$ is the vector of prognostic covariate effects;
*   $\boldsymbol{\gamma}_k$ represents the treatment-covariate interaction vector.

### 7.2 Likelihood Integration
For aggregate studies where individual $\mathbf{x}_{ij}$ are unknown, the study-level aggregate likelihood contribution is obtained by integrating the individual-level model over the study-specific joint covariate distribution $F_j(\mathbf{x})$:

$$
p_{jk} = \int g^{-1}(\eta_{jk}(\mathbf{x})) \, dF_j(\mathbf{x}).
$$

This integration is performed numerically using quasi-Monte Carlo integration or Gaussian quadrature.

### 7.3 Covariate-Distribution Governance
To ensure integration validity when only marginal aggregate data are reported:
*   **Reconstruction Methods:** Users must specify whether joint IPD covariate distributions ($F_j$) are observed directly, or reconstructed under copula or parametric joint distribution assumptions.
*   **Correlation Assumptions:** The assumed correlations between covariates must be declared and subjected to prior sensitivity audits.
*   **Target Population Overlap:** Overlap between aggregate covariate bounds and target population distributions must be reported, flagging covariate extrapolation beyond the observed support.

### 7.4 Outputs and Diagnostics
*   **Conditional and Average Effects:** Reports treatment effects for specific patient profiles and standardized target-population averages.
*   **Covariate Overlap:** Computes distance metrics (e.g., Mahalanobis distance) to detect covariate extrapolation.
*   **Numerical Integration Quality:** Monitors the error of the integration approximation.

---

## 8. Component Network Meta-Analysis (CNMA)

**Module:** `src/bias_nma/models/component/`

### 8.1 Additive and Interaction Models
For multi-component interventions, treatment effects ($d_t$) are decomposed into their constituent components ($C$):

$$
d_{A+B} = d_A + d_B + \gamma_{AB},
$$

where:
*   $d_A, d_B$ are the main additive effects of components $A$ and $B$;
*   $\gamma_{AB}$ is the interaction effect (set to zero in additive models).

### 8.2 Structural Diagnostics
*   **Identifiability Checks:** Verifies that component parameters are estimable from the treatment network design matrix.
*   **Connectedness:** Checks network connectivity at both the treatment level and the constituent component level.
*   **Goodness of Fit:** Compares residual deviance between CNMA and full-treatment NMA models to justify additive assumptions.

---

## 9. Dose-Response Network Meta-Analysis

**Module:** `src/bias_nma/models/dose_response/`

### 9.1 Models
The dose-response module models continuous dose variables using:
*   **Nonlinear Functions:** Emax, Hill, exponential, or fractional polynomial functions.
*   **Spline Functions:** Restricted cubic splines and monotonic splines to capture complex dose curves.
*   **Hierarchical Curves:** Class-level dose-response curves linking agents with similar mechanisms.

### 9.2 Constraints & Extrapolations
*   **Dose-Equivalence Assumptions:** Standardizes doses using predefined equivalence coefficients. Equivalence coefficients should not always be treated as fixed. The system permits:

$$
e_k \sim p(e_k)
$$

when equivalence ratios are uncertain, propagating that uncertainty into treatment-effect estimates.
*   **Prediction:** Estimates effects at unobserved doses.
*   **Extrapolation Warnings:** Flags predictions outside the range of observed trial doses.

---

## 10. Genuine Multivariate Network Meta-Analysis

**Module:** `src/bias_nma/models/multivariate/`

### 10.1 Model Structure
For correlated, non-exclusive outcomes (e.g., mortality, hospitalization, and adverse events), the study-level relative effects $\boldsymbol{\theta}_i$ are modeled jointly:

$$
\boldsymbol{\theta}_i \sim \operatorname{MVN}\left(\boldsymbol{\Delta}_i, \boldsymbol{S}_i + \boldsymbol{\Sigma}\right),
$$

where:
*   $\boldsymbol{S}_i$ is the within-study covariance matrix (estimated or imputed);
*   $\boldsymbol{\Sigma}$ is the between-study covariance matrix.

### 10.2 Missing Outcomes & Borrowing of Strength
*   **Outcome Integration:** Integrates over unreported outcome components under the joint model and may produce posterior predictive distributions for those outcomes. Predictions are clearly distinguished from observed study results.
*   **Borrow-of-Strength Diagnostics:** Computes the change in precision (shrinkage) across outcomes to evaluate the impact of multivariate modeling.

### 10.3 Multivariate Covariance Sensitivity
When within-study correlations are unavailable, the engine enforces:
1.  a declared correlation imputation method;
2.  a plausible correlation range;
3.  a sensitivity analysis over that range;
4.  positive-definiteness checks on the combined covariance matrix;
5.  a comparison against separate univariate analyses.

---

## 11. Cross-Design Network Meta-Analysis

**Module:** `src/bias_nma/models/cross_design/`

### 11.1 Model Structure
For combining randomized controlled trials (RCTs) and non-randomized studies (NRS), the relative treatment effects $\theta_{ic}^{\mathrm{obs}}$ are modeled via:

$$
\theta_{ic}^{\mathrm{obs}} = \Delta_c + b_{d_i, c} + u_{ic},
$$

where:
*   $i$ indexes studies;
*   $c$ indexes treatment comparisons;
*   $d_i$ denotes the study design (RCT or NRS);
*   $\Delta_c$ is the target treatment effect;
*   $b_{d_i, c}$ is a comparison-specific design-bias term, with $b_{\text{RCT}, c}$ constrained to zero as the evidential anchor;
*   $u_{ic}$ represents study-specific residual heterogeneity.

Observational bias is comparison-specific rather than a single universal NRS parameter, reflecting differing confounding and data structures.

### 11.2 Borrowing & Diagnostics
*   **Hierarchical & Commensurate Borrowing:** Controls the amount of borrowing from NRS evidence using power priors or design-specific variance inflation factors.
*   **Design-Conflict Diagnostics:** Computes design-inconsistency statistics to evaluate conflicts between RCT and NRS estimates.
*   **Stratified Reporting:** Generates RCT-only, NRS-only, and combined estimates alongside sensitivity analyses relative to assumed design-bias priors.

---

## 12. Rare-Event and Zero-Event NMA

**Module:** `src/bias_nma/models/standard_nma/`

### 12.1 Likelihoods
*   **Exact Binomial Likelihood:** Evaluates events without continuity corrections ($+0.5$).
*   **Mantel-Haenszel NMA:** Fixed-effects Mantel-Haenszel estimators for sparse binary outcomes.
*   **Non-Central Hypergeometric Models:** Fits conditional likelihoods for multi-arm networks, subject to network size and computational tractability constraints.
*   **Penalized Regression:** Applies Firth's penalization to logistic models.

### 12.2 Zero-Event Policy
*   Double-zero studies contribute no direct contrast information under conventional relative-effect models. They may be retained in compatible arm-based likelihoods when they inform absolute risks or other hierarchical parameters. Their inclusion policy is reported for every analysis.
*   Studies with zero events in one arm are modelled using arm-level binomial likelihoods without continuity correction, or validated conditional likelihood methods where applicable.

---

## 13. Disconnected-Network Policy

The platform enforces strict rules for networks containing disconnected components:
1.  **Pre-Fit Detection:** Scans the adjacency matrix of the treatment network.
2.  **Default Action:** Halts standard relative-effect analysis and fits separate subnetwork models.
3.  **Reconnection Rule:** Reconnection requires structural assumptions such as shared components, shared dose-response functions, external relative-effect information, or exchangeable class effects. Baseline-risk exchangeability or prior baseline risk assumptions alone are not treated as sufficient to reconnect networks unless these identifying assumptions are explicitly justified and tested.
4.  **Labeling:** Outputs for reconnected nodes are flagged as "structurally connected" (as opposed to empirically connected).

---

## 14. Time-to-Event and Competing-Risk Extension

**Module:** `src/bias_nma/survival/flexible_hazards.py`

Where individual event times or sufficiently detailed interval data are available, the engine may fit piecewise exponential models, restricted cubic splines, M-splines, Royston-Parmar models, fractional polynomials, treatment-specific time-varying coefficients, and competing-risk models.

Non-proportional hazards are evaluated using splines, fractional polynomials, or treatment-specific time-varying coefficients. Hazards are constrained to remain non-negative.

---

## 15. Bayesian Inference Engine

**Module:** `src/bias_nma/inference/stan_backend.py`

### 15.1 Stan Integration
CmdStanPy serves as the production inference runner. All Bayesian models are compiled to C++ code using CmdStan.
*   **Parameterizations:** Uses non-centered parameterizations for heterogeneity variables by default to avoid funnel geometries.
*   **Ranking:** Draw-specific treatment ranks and clinically important benefit indicators are generated while preserving joint posterior draws. Rank probabilities, rankograms and SUCRA are then calculated from the complete posterior sample.

---

## 16. Probabilistic Bias Analysis

**Module:** `src/bias_nma/models/bias_adjustment/`

Bias parameters are defined on the model's linear-predictor or prespecified effect scale (e.g., log-OR, log-RR, or hazard scale). For study ($i$):

$$
\theta_i^{\mathrm{observed}} = \theta_i^{\mathrm{true}}+b_i,
$$

where ($b_i$) is a bias parameter whose direction, scale, distribution, and clinical interpretation are reported explicitly.

Analyses include:

1. primary analysis without numerical bias correction;
2. exclusion of studies at high risk of bias;
3. domain-specific probabilistic bias adjustment;
4. prior sensitivity analyses;
5. worst-case and tipping-point analyses.

---

## 17. Registry and Missing-Evidence Auditing

**Modules:**

* `src/bias_nma/registry/`
* `src/bias_nma/models/bias_adjustment/publication_bias.py`

### Registry audit outputs
The registry auditor identifies registered but unpublished studies, discrepancies in planned and reported outcomes, sample sizes, and delayed reporting. These are registered as qualitative metadata flags to serve as one input into wider risk assessment, not as weight adjustments.

### Missing-evidence analyses
Where appropriate, the engine supports selection models, robust Bayesian meta-analysis, pattern-mixture sensitivity analyses, and ROB-MEN-compatible evidence summaries.

---

## 18. Collaborative TMLE IPD Extension

**Module:** `src/bias_nma/models/bias_adjustment/ctmle.py`

### Scope
C-TMLE is an optional individual-participant-data module for observational comparisons or randomized studies requiring adjustment for informative missingness, non-adherence or treatment crossover. It is not applied automatically to aggregate randomized-trial NMA.

### Diagnostics
*   **Propensity Weights:** Reports propensity-adjusted equivalent sample size derived from explicit inverse-probability weights, where such weights are used. Clever-covariate distributions, positivity, truncation and influence-curve behaviour are assessed separately.
*   **Influence Curves:** Computes influence-curve variance and standard error.
*   **Truncation:** Monitors the proportion of propensity scores truncated.
*   **Cross-Validation:** Monitors cross-validated nuisance-model loss.

---

## 19. Model Averaging

**Module:** `src/bias_nma/models/bias_adjustment/bma.py`

Model averaging may be used for common versus random effects, alternative heterogeneity priors, proportional versus non-proportional hazards, and selected consistency and inconsistency structures.

Where BIC-derived weights are used:

$$
w_k= \frac{\exp(-\tfrac12(BIC_k-BIC_{\min}))} {\sum_j\exp(-\tfrac12(BIC_j-BIC_{\min}))},
$$

they are labelled approximate information-criterion weights.

---

## 20. Synthetic-Data Simulation

**Package:** `bias_nma_simulation/` (Isolated)

The synthetic-data simulation module is restricted to software testing, simulation studies, demonstration datasets, and privacy-preserving methodological development. Synthetic participants are never combined with real participants to artificially inflate sample size or narrow the treatment-contrast variance.

---

## 21. Governance, Model Certification and Evidence Certainty

### 21.1 Model Acceptance Criteria
A model is classified as successfully fitted only when:
*   **split-$\hat{R}$:** Rank-normalized split-$\hat{R} \leq 1.01$ for all material parameters (exceptions require documented justification).
*   **MCMC ESS Metrics:** Bulk and tail ESS statistics meet prespecified thresholds (bulk ESS $\geq 400$, tail ESS $\geq 400$ in total). Higher thresholds are enforced for ranking distributions and tail probabilities.
*   **Monte Carlo Error:** MCSE is sufficiently small relative to posterior uncertainty for the reported numerical precision.
*   **HMC warnings:** No unresolved divergent transitions or tree-depth saturation events remain.
*   **Checks:** Posterior predictive checks do not show major systematic misfit.
*   **Frequentist Convergence Requirements:** 
    1. design matrix is full-rank or utilizes a documented generalized-inverse solution;
    2. estimated covariance matrices are positive-semidefinite;
    3. optimization completes successfully without boundary or singular warnings;
    4. estimated standard errors are finite and non-zero;
    5. heterogeneity estimator converges within maximum iterations;
    6. influence and residual diagnostics are within acceptable boundaries;
    7. results are consistent under alternative $\tau^2$ estimators;
    8. warnings are generated when asymptotic approximations are deemed unreliable.

### 21.2 Certainty of Evidence and Contribution Matrices
The platform generates evidence certainty calculations compatible with CINeMA and ROB-MEN.
*   **Study Contribution Matrix:** The engine first computes the estimator or hat matrix and then derives non-negative study and direct-comparison contribution percentages using an explicitly documented evidence-flow algorithm. Raw signed matrix coefficients are not interpreted directly as percentage contributions.
*   **Risk of Bias Propagation:** Projects RoB 2 categories (**low risk**, **some concerns**, and **high risk**) across the network using the contribution percentages.
*   **ROB-MEN Assessments:** Integrates the registry auditor alongside structured assessments of known unpublished studies, selectively unavailable outcomes, small-study effects, comparison-specific reporting patterns, the likely direction of missing evidence, and direct-evidence contribution percentages.
*   **Equivalence Limits:** Evaluates confidence or credible intervals, as appropriate, against prespecified clinically important equivalence margins.

### 21.3 Model Certification Levels
Every model fitted by the platform receives a certification level:

| Level | Meaning |
| :--- | :--- |
| **Experimental** | Implemented but not validated. |
| **Numerically Verified** | Gradients and likelihoods agree with independent calculations. |
| **Reference Matched** | Reproduces established software (e.g., `netmeta`, `multinma`) on benchmark datasets. |
| **Simulation Validated** | Acceptable bias, coverage, and calibration across simulation matrices. |
| **Externally Reproduced** | Independently tested by an external group. |
| **Production Certified** | Suitable for clinical or HTA analyses. |

Methodological and research outputs from models below **Production Certified** status are watermarked with their certification level. Automated clinical recommendations are blocked, and exports suitable for clinical, regulatory, or HTA decision-making are disabled until the corresponding module has been certified.

---

## 22. Global Benchmark and Validation Programme

For every model certified for production, the platform executes a benchmark programme:
1.  **Datasets:** Reproduce at least $20$ published reference clinical trials/networks.
2.  **Inference Matching:** Compare Bayesian outcomes against `multinma` and CmdStan reference models.
3.  **Frequentist Matching:** Compare frequentist outcomes against `netmeta`.
4.  **Dose-Response Matching:** Compare dose-response models against `MBNMAdose`.
5.  **Cross-Design Matching:** Compare cross-design synthesis against `crossnma`.
6.  **SBC:** Run simulation-based calibration (SBC) to verify posterior distribution shape.
7.  **Recovery:** Verify parameter recovery, bias, RMSE, and interval coverage in simulation testing.
8.  **Ranks:** Assess ranking calibration using calibration curves.
9.  **Stress Testing:** Test parameter estimation under sparse, highly heterogeneous, and disconnected configurations.
10. **Reproducibility:** Publish all benchmark datasets, scripts, and numerical tolerances.

---

## 23. Reproducibility, Reporting, and Architecture

### 23.1 Reproducible Analysis Bundle
Every execution yields a reproducible bundle containing:
*   Original `.stan` source files.
*   Generated C++ source where retained.
*   Compiled executable checksum.
*   CmdStan version, compiler version and flags.
*   Operating system and processor architecture.
*   Dependency lockfile hash and model executable hash.
*   Cryptographic hash of the input dataset.
*   Priors, configuration parameters, and seeds.
*   Reviewer overrides and qualitative risk assessments.
*   Convergence diagnostics and contribution tables.

### 23.2 Directory Structure
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

---

## 24. Build and Validation Status Schema

This section is generated automatically by the continuous-integration workflow. It must not be edited manually. A build is described as verified only when the associated machine-readable reports and immutable artefacts are available.

### 24.1 Build Metadata

The build report records:

- complete Git commit SHA;
- repository and branch or tag;
- UTC build timestamp;
- continuous-integration provider and immutable run identifier;
- operating system and processor architecture;
- Python, R, CmdStan and compiler versions;
- dependency lockfile hash;
- container image digest, where applicable;
- validation-engine version.

Illustrative or placeholder identifiers must not be presented as actual build results.

### 24.2 Test-Suite Summary

The report separates:

- unit tests;
- integration tests;
- numerical-gradient tests;
- regression tests;
- reference-software comparisons;
- simulation-based calibration tests;
- performance and numerical-stress tests;
- failure-mode and invalid-input tests.

For each category, it reports:

- total tests;
- passed tests;
- failed tests;
- skipped or expected-failure tests;
- execution duration;
- associated report or artefact hash.

Code coverage is reported separately as line and branch coverage. High code coverage is not interpreted as evidence of statistical validity.

### 24.3 Model-Certification Evidence

Each certification entry contains:

- module and model version;
- certification level;
- reference implementation and exact version;
- reference datasets;
- analysis configuration;
- comparison metrics;
- prespecified numerical tolerances;
- observed numerical differences;
- simulation scenarios and performance criteria;
- validation-report checksum;
- validation date;
- independent reviewer or reproducing group, where applicable;
- known limitations and unsupported data conditions.

A model cannot receive a certification level based solely on passing unit tests or achieving high code coverage.

### 24.4 Deployment Status

The report states clearly which modules are:

- experimental;
- numerically verified;
- reference matched;
- simulation validated;
- externally reproduced;
- production certified.

Modules below Production Certified status are restricted to development, research or validation use and cannot generate outputs labelled as suitable for clinical, regulatory or health-technology-assessment decision-making.

If no module is Production Certified, the report must state:

> No modules in this build currently hold Production Certified status. Clinical and HTA reporting is disabled.

### 24.5 Reproducibility Artefacts

Each build publishes or securely retains:

- original `.stan` source files;
- generated C++ source where retained;
- compiled executable checksums;
- compiler options;
- complete dependency manifests;
- benchmark datasets permitted for redistribution;
- reference outputs;
- numerical-tolerance definitions;
- simulation seeds and configurations;
- code-coverage reports;
- static-analysis reports;
- benchmark and validation logs.

Any statement that the working tree is clean, tests have passed or validation checks are clear must be derived directly from the continuous-integration system and linked to an immutable report.

---
Current build and validation status is reported exclusively through the machine-generated CI artefacts described above.
