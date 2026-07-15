<!-- sentinel:skip-file -->
# Technical Specification: Advanced Bias-Aware Network Meta-Analysis Platform

## 1. Scope and Design Principles

The engine performs Bayesian network meta-analysis of randomized and non-randomized comparative studies. The core model preserves randomization, within-study treatment contrasts, correlations from multi-arm trials, and between-study heterogeneity.

Bias-related information is incorporated through transparent sensitivity analyses or explicitly specified probabilistic bias models. Study weights are not modified using arbitrary sponsorship, attrition or registry penalty constants.

All advanced modules are classified as:

1.  **Core NMA Estimators:** Bayesian and frequentist engines, including component, dose-response, multivariate, and survival models.
2.  **Optional Bias and Sensitivity Models:** Probabilistic bias models and registry-informed analyses.
3.  **IPD Extensions:** Multilevel Network Meta-Regression (ML-NMR) and causal individual-participant-data (IPD) adjustment.
4.  **Simulation and Validation Tools:** Benchmarks, simulation matrices, and privacy-preserving data generators.

---

## 2. Estimand Definition & Evidence Separations

### 2.1 Estimand Specification
Before model fitting, each analysis must define the target estimand:
*   **Population:** Target clinical population baseline parameters.
*   **Treatment Strategies:** Set of eligible comparator interventions.
*   **Outcome:** Primary and secondary endpoints (mutually exclusive and collectively exhaustive).
*   **Follow-Up Time:** Prespecified landmarks or continuous horizons.
*   **Effect Measure:** Relative risk, odds ratio, risk difference, or hazard ratio.
*   **Intercurrent Events:** Strategy for handling treatment discontinuation, crossovers, or rescue therapies (e.g. treatment-policy, hypothetical, composite, or while-on-treatment).
*   **Analysis Interpretation:** Intention-to-treat (ITT) versus per-protocol (PP).

Results from materially different estimands are not pooled without an explicit transformation or hierarchical model.

### 2.2 Separating Evidence Designs
Randomized and non-randomized evidence are analysed separately by default. Joint synthesis requires a prespecified design-adjusted hierarchical model with design-specific bias parameters and sensitivity analyses.

---

## 3. Core Network Meta-Analysis Model

### 3.1 Study and Treatment Structure

For study ($i$), arm ($a$), and treatment ($t_{ia}$), the linear predictor is:

$$
\eta_{ia}=\mu_i+\delta_{ia},
$$

where:

* $\mu_i$ is the study-specific baseline parameter;
* $\delta_{ia}$ is the relative effect of treatment ($t_{ia}$) against the study reference treatment;
* $\delta_{i1}=0$ for the reference arm.

Under a consistency model:

$$
\delta_{ia}=d_{t_{ia}}-d_{t_{i1}}+u_{ia},
$$

where ($d_t$) is the network treatment effect relative to the network reference and ($u_{ia}$) represents between-study heterogeneity.

Multi-arm trial random effects are modelled jointly so that the induced correlations between treatment contrasts are retained.

### 3.2 Heterogeneity

The default random-effects model assumes:

$$
u_i \sim \operatorname{MVN}(0,\Sigma_i(\tau)),
$$

with a common heterogeneity parameter ($\tau$), unless outcome-specific or comparison-specific heterogeneity is prespecified and supported by sufficient data.

The engine reports:

* posterior median and credible interval for ($\tau$);
* predictive intervals;
* prior-to-posterior sensitivity;
* comparison with a common-effect model.

### 2.3 Transitivity and Incoherence

Potential effect modifiers are summarized across treatment comparisons before model fitting.

Incoherence is evaluated using:

* unrelated mean-effects or design-by-treatment models;
* node-splitting or direct–indirect comparisons;
* residual deviance and posterior predictive checks;
* design-specific influence diagnostics.

These methods diagnose incoherence but do not establish which source of evidence is correct. The software does not automatically choose between direct or indirect evidence based solely on statistical significance.

---

## 4. Multinomial Clinical-State NMA

**Module:** `src/bias_nma_adv/multinomial.py`

### Purpose

Models mutually exclusive and collectively exhaustive clinical states measured at a common, prespecified follow-up time, such as:

* event-free survival;
* cardiovascular death;
* non-cardiovascular death;
* heart-failure hospitalization as the first event (where states are defined at a common landmark or through a multi-state framework).

This module is intended for fixed-time multinomial outcomes. It is not described as a full competing-risks survival model unless event times and censoring are modelled explicitly.

### Likelihood

For study ($i$), arm ($a$), and outcome category ($j$):

$$
\mathbf{y}_{ia}\sim \operatorname{Multinomial}\left(n_{ia},\mathbf{p}_{ia}\right).
$$

Using category ($J$) as the reference:

$$
p_{iaj} = \frac{\exp(\eta_{iaj})}{1+\sum_{k=1}^{J-1}\exp(\eta_{iak})}, \qquad j=1,\ldots,J-1,
$$

and

$$
p_{iaJ} = \frac{1}{1+\sum_{k=1}^{J-1}\exp(\eta_{iak})}.
$$

Each category has study-specific baseline parameters and network treatment effects. The module defines whether category-specific treatment effects are:
1.  **Independent:** Assuming no correlation between endpoints.
2.  **Correlated (Unstructured):** Using an unstructured covariance matrix (poorly identified in small networks).
3.  **Correlated (Factor):** Correlated through a lower-dimensional factor structure.

---

## 5. Time-to-Event and Competing-Risk Extension

**Module:** `src/bias_nma_adv/competing_risks.py`

Where individual event times or sufficiently detailed interval data are available, the engine may fit:

* cause-specific hazard models;
* flexible parametric survival models;
* piecewise exponential models;
* cumulative-incidence models.

Cause-specific hazards are indexed by event type and treatment. The output must distinguish hazard ratios from cumulative-incidence contrasts because these estimands are not interchangeable.

Non-proportional hazards are evaluated using splines (M-splines, cubic splines), fractional polynomials, or treatment-specific time-varying coefficients. Hazards are constrained to remain non-negative.

---

## 6. Bayesian Inference Using NUTS

**Module:** `src/bias_nma_adv/nuts.py`

### Purpose

Obtains posterior draws for treatment effects, heterogeneity parameters, inconsistency parameters and other model quantities.

### Implementation requirements

The preferred production backend is a validated automatic-differentiation framework such as Stan (via CmdStanPy). Any independent NUTS implementation must be verified against a reference implementation using analytically tractable and simulated models.

NUTS adapts the leapfrog trajectory length and, during warm-up, the step size and mass matrix.

### Mandatory diagnostics

The engine reports:

* split-$\hat R$;
* bulk and tail ESS statistics;
* divergent transitions;
* maximum tree-depth events;
* energy Bayesian fraction of missing information;
* chain trace plots;
* Monte Carlo standard errors;
* posterior predictive diagnostics.

A model is not marked as successfully fitted merely because sampling terminates.

---

## 7. Probabilistic Bias Analysis

**Module:** `src/bias_nma_adv/bias_model.py`

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

## 8. Registry and Missing-Evidence Auditing

**Modules:**

* `src/bias_nma_adv/publication_bias.py`
* `src/bias_nma_adv/registry_audit.py`
* `src/bias_nma_adv/sponsor_bias.py`

### Registry audit outputs

The registry auditor identifies:

* registered but unpublished studies;
* discrepancies in planned and reported primary outcomes;
* changes in outcome time points;
* differences in planned and analysed sample sizes;
* unexplained exclusions;
* delayed or incomplete reporting.

These outputs are flags and evidence summaries, not automatic study-weight multipliers.

### Missing-evidence analyses

Where appropriate, the engine supports:

* comparison-adjusted funnel plots;
* selection models;
* robust Bayesian meta-analysis;
* pattern-mixture sensitivity analyses;
* inclusion of registered but unpublished results when data are available;
* ROB-MEN-compatible evidence summaries.

### Sponsorship

Funding source and author conflicts are retained as study characteristics. They may be examined through subgroup analysis, meta-regression or probabilistic bias analysis. No universal sponsorship penalty is applied.

### Attrition

Attrition is evaluated in relation to:

* imbalance between arms;
* reasons for missingness;
* relationship between missingness and outcome;
* analysis population;
* handling of missing data.

A fixed dropout threshold is not used as an automatic weight adjustment.

---

## 9. Collaborative TMLE IPD Extension

**Module:** `src/bias_nma_adv/ctmle.py`

### Scope

C-TMLE is an optional individual-participant-data module for observational comparisons or randomized studies requiring adjustment for informative missingness, non-adherence or treatment crossover.

It is not applied automatically to aggregate randomized-trial NMA.

### Estimation

The module estimates a prespecified causal estimand using:

* an initial outcome regression ($Q(A,W)$);
* a treatment or censoring mechanism ($g(A\mid W)$);
* targeted fluctuation updates;
* cross-fitting or sample splitting;
* influence-curve-based uncertainty estimation.

Covariate selection is restricted to a prespecified candidate set. Selection performance is assessed using cross-validated loss rather than in-sample MSE alone.

### Diagnostics

* equivalent sample size after weighting;
* influence-curve variance and standard error;
* proportion of propensity scores truncated;
* cross-validated nuisance-model loss;
* confidence-interval coverage in simulation testing;
* positivity and propensity-score overlap;
* truncation sensitivity;
* nuisance-model performance;
* comparison with alternative estimators.

---

## 10. Model Averaging

**Module:** `src/bias_nma_adv/bma.py`

Model averaging may be used for prespecified alternatives such as:

* common versus random effects;
* alternative heterogeneity priors;
* proportional versus non-proportional hazards;
* selected consistency and inconsistency structures.

Where BIC-derived weights are used:

$$
w_k= \frac{\exp(-\tfrac12(BIC_k-BIC_{\min}))} {\sum_j\exp(-\tfrac12(BIC_j-BIC_{\min}))},
$$

they are labelled approximate information-criterion weights.

For fully Bayesian models, marginal-likelihood, stacking or predictive-weighting methods are preferred when computationally feasible.

The engine reports model-specific estimates as well as averaged estimates so that clinically important incoherence is not concealed.

---

## 11. Synthetic-Data Simulation

**Module:** `bias_nma_simulation/` (Isolated package)

The synthetic-data simulation module is restricted to:

* software testing;
* simulation studies;
* demonstration datasets;
* privacy-preserving methodological development.

Synthetic participants are never combined with real participants to artificially inflate sample size or narrow the treatment-contrast variance.

Validation includes:

* marginal-distribution agreement;
* correlation and conditional-dependence preservation;
* clinical-range checks;
* rare-category performance;
* train–test discrimination;
* membership-inference and disclosure-risk assessment;
* comparison with simpler parametric simulators.

---

## 12. Flexible Hazard-Function Discovery

**Module:** `src/bias_nma_adv/symbolic.py`

The module evaluates candidate time functions for non-proportional treatment effects or baseline hazards.

Candidate functions may include:

$$
1,\quad t,\quad \sqrt{t},\quad \log(t+c),\quad \exp(-\lambda t),
$$

restricted cubic splines and piecewise-constant functions.

Candidate models are evaluated using held-out predictive performance, information criteria or Bayesian model comparison. Constraints are applied to ensure valid survival and hazard functions.

The selected functional form, uncertainty due to model selection and extrapolation behaviour are reported.

---

## 13. Treatment Ranking

The engine reports:

* posterior rank probabilities;
* cumulative ranking curves or SUCRA where requested;
* probability that each treatment is best;
* probability of clinically important benefit or harm;
* rankograms;
* uncertainty intervals for absolute outcomes.

Rankings are accompanied by treatment-effect estimates and certainty assessments. Rankings are not presented as conclusive when effects are imprecise, incoherent or clinically similar.

---

## 14. Validation and Testing

Each estimator must pass:

* unit tests for likelihood and gradient calculations;
* simulation-based calibration;
* parameter-recovery studies;
* type-I error and interval-coverage assessments where applicable;
* comparison with established software;
* multi-arm trial tests;
* disconnected-network detection;
* zero-event and sparse-data tests;
* reproducibility tests using fixed random seeds;
* numerical stress tests.

Reference comparisons should include established Stan-based or validated NMA implementations where equivalent models are available.

---

## 15. Estimands, Evidence Designs and Analysis Governance

### 15.1 Estimand Definition
Before model fitting, each analysis must define:
*   the target population;
*   eligible treatment strategies;
*   outcome definition;
*   follow-up time or time horizon;
*   effect measure;
*   treatment-policy, hypothetical, composite or while-on-treatment strategy for intercurrent events;
*   intention-to-treat, per-protocol or observational causal interpretation.

Results from materially different estimands are not pooled without an explicit transformation or hierarchical model.

### 15.2 Evidence-Design Separation
Randomized and non-randomized studies are analysed separately by default.
Joint synthesis is permitted only through a prespecified design-adjusted model containing design-specific parameters or bias distributions. The engine must report:
1.  randomized-evidence-only results;
2.  non-randomized-evidence-only results;
3.  combined design-adjusted results;
4.  sensitivity to the assumed bias parameters.

Synthetic observations are never classified as clinical evidence.

### 15.3 Prior Specification
Every Bayesian model must declare priors for:
*   treatment effects;
*   study baselines;
*   heterogeneity parameters;
*   correlation parameters;
*   inconsistency parameters;
*   bias parameters;
*   time-varying or spline coefficients.

The engine performs prior predictive checks and reports sensitivity to clinically plausible alternative priors.

### 15.4 Model Acceptance Criteria
A model is classified as successfully fitted only when:
*   all material parameters have acceptable split-$\hat R$;
*   MCMC sample sizes meet prespecified thresholds;
*   Monte Carlo error is small relative to posterior uncertainty;
*   no unresolved divergent transitions remain;
*   maximum tree-depth events are absent or adequately investigated;
*   posterior predictive checks do not show major systematic misfit;
*   treatment effects are identifiable from the observed network.

Failure of any criterion produces a warning or failed-model status rather than silently returning treatment rankings.

### 15.5 Certainty of Evidence
For each treatment comparison and outcome, the reporting layer records assessments of:
*   within-study bias;
*   across-studies bias;
*   indirectness;
*   imprecision;
*   heterogeneity;
*   incoherence.

Automated calculations and flags support—but do not replace—reviewer judgements. Treatment rankings are never used as substitutes for comparison-specific certainty assessments.
