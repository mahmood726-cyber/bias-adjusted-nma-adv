# Technical Specification: Advanced Bias-Aware Network Meta-Analysis Engine

## 1. Scope and Design Principles

The engine performs Bayesian network meta-analysis of randomized and, where explicitly enabled, non-randomized comparative studies. The core model preserves randomization, within-study treatment contrasts, correlations from multi-arm trials and between-study heterogeneity.

Bias-related information is incorporated through transparent sensitivity analyses or explicitly specified probabilistic bias models. Study weights are not modified using arbitrary sponsorship, attrition or registry penalty constants.

All advanced modules are classified as:

1. core NMA estimators;
2. optional bias and sensitivity models;
3. IPD extensions;
4. simulation and validation tools.

---

## 2. Core Network Meta-Analysis Model

### 2.1 Study and Treatment Structure

For study (i), arm (a), and treatment (t_{ia}), the linear predictor is:

$$
\eta_{ia}=\mu_i+\delta_{ia},
$$

where:

* \mu_i is the study-specific baseline parameter;
* \delta_{ia} is the relative effect of treatment (t_{ia}) against the study reference treatment;
* \delta_{i1}=0 for the reference arm.

Under a consistency model:

$$
\delta_{ia}=d_{t_{ia}}-d_{t_{i1}}+u_{ia},
$$

where (d_t) is the network treatment effect relative to the network reference and (u_{ia}) represents between-study heterogeneity.

Multi-arm trial random effects are modelled jointly so that the induced correlations between treatment contrasts are retained.

### 2.2 Heterogeneity

The default random-effects model assumes:

$$
u_i \sim \operatorname{MVN}(0,\Sigma_i(\tau)),
$$

with a common heterogeneity parameter (\tau), unless outcome-specific or comparison-specific heterogeneity is prespecified and supported by sufficient data.

The engine reports:

* posterior median and credible interval for (\tau);
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

Incoherent evidence is displayed explicitly. It is not automatically averaged with the consistency model without a prespecified model-averaging analysis.

---

## 3. Multinomial Clinical-State NMA

**Module:** `src/bias_nma_adv/multinomial.py`

### Purpose

Models mutually exclusive clinical states measured at a common, prespecified follow-up time, such as:

* event-free survival;
* cardiovascular death;
* non-cardiovascular death;
* heart-failure hospitalization as the first event.

This module is intended for fixed-time multinomial outcomes. It is not described as a full competing-risks survival model unless event times and censoring are modelled explicitly.

### Likelihood

For study (i), arm (a), and outcome category (j):

$$
\mathbf{y}_{ia}\sim \operatorname{Multinomial}\left(n_{ia},\mathbf{p}_{ia}\right).
$$

Using category (J) as the reference:

$$
p_{iaj} = \frac{\exp(\eta_{iaj})}{1+\sum_{k=1}^{J-1}\exp(\eta_{iak})}, \qquad j=1,\ldots,J-1,
$$

and

$$
p_{iaJ} = \frac{1}{1+\sum_{k=1}^{J-1}\exp(\eta_{iak})}.
$$

Each category has study-specific baseline parameters and network treatment effects. Correlations between category-specific treatment effects may be estimated using a structured covariance matrix where the data are sufficient.

### Required safeguards

* stable log-sum-exp calculations;
* explicit reference-category handling;
* multi-arm trial covariance;
* sparse-cell and zero-cell handling;
* identifiability checks;
* posterior predictive checks;
* comparison against independently implemented reference models.

---

## 4. Time-to-Event and Competing-Risk Extension

**Module:** `src/bias_nma_adv/competing_risks.py`

Where individual event times or sufficiently detailed interval data are available, the engine may fit:

* cause-specific hazard models;
* flexible parametric survival models;
* piecewise exponential models;
* cumulative-incidence models.

Cause-specific hazards are indexed by event type and treatment. The output must distinguish hazard ratios from cumulative-incidence contrasts because these estimands are not interchangeable.

Non-proportional hazards are evaluated using prespecified time functions, splines or model averaging. Hazards are constrained to remain non-negative.

---

## 5. Bayesian Inference Using NUTS

**Module:** `src/bias_nma_adv/nuts.py`

### Purpose

Obtains posterior draws for treatment effects, heterogeneity parameters, inconsistency parameters and other model quantities.

### Implementation requirements

The preferred production backend is a validated automatic-differentiation framework such as Stan. Any independent NUTS implementation must be verified against a reference implementation using analytically tractable and simulated models.

NUTS adapts the leapfrog trajectory length and, during warm-up, the step size and mass matrix.

### Mandatory diagnostics

The engine reports:

* split-(\hat R);
* bulk and tail MCMC sample sizes;
* divergent transitions;
* maximum tree-depth events;
* energy Bayesian fraction of missing information;
* chain trace plots;
* Monte Carlo standard errors;
* posterior predictive diagnostics.

A model is not marked as successfully fitted merely because sampling terminates.

---

## 6. Probabilistic Bias Analysis

**Module:** `src/bias_nma_adv/bias_model.py`

Bias is represented as an explicit additive or multiplicative parameter rather than an arbitrary change in study weight.

For study (i):

$$
\theta_i^{\mathrm{observed}} = \theta_i^{\mathrm{true}}+b_i,
$$

where (b_i) is a bias parameter whose prior distribution is determined by the relevant bias domain, empirical evidence or structured expert elicitation.

Analyses include:

1. primary analysis without numerical bias correction;
2. exclusion of studies at high risk of bias;
3. domain-specific probabilistic bias adjustment;
4. prior sensitivity analyses;
5. worst-case and tipping-point analyses.

The direction and magnitude of every bias prior are reported.

---

## 7. Registry and Missing-Evidence Auditing

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

## 8. Collaborative TMLE IPD Extension

**Module:** `src/bias_nma_adv/ctmle.py`

### Scope

C-TMLE is an optional individual-participant-data module for observational comparisons or randomized studies requiring adjustment for informative missingness, non-adherence or treatment crossover.

It is not applied automatically to aggregate randomized-trial NMA.

### Estimation

The module estimates a prespecified causal estimand using:

* an initial outcome regression (Q(A,W));
* a treatment or censoring mechanism (g(A\mid W));
* targeted fluctuation updates;
* cross-fitting or sample splitting;
* influence-curve-based uncertainty estimation.

Covariate selection is restricted to a prespecified candidate set. Selection performance is assessed using cross-validated loss rather than in-sample MSE alone.

### Diagnostics

* positivity and propensity-score overlap;
* truncation sensitivity;
* nuisance-model performance;
* influence-curve mean;
* bulk and tail MCMC sample sizes;
* comparison with alternative estimators.

---

## 9. Model Averaging

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

## 10. Synthetic-Data Simulation

**Module:** `src/bias_nma_adv/gan.py`

The synthetic-data module is restricted to:

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

## 11. Flexible Hazard-Function Discovery

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

## 12. Treatment Ranking

The engine reports:

* posterior rank probabilities;
* cumulative ranking curves or SUCRA where requested;
* probability that each treatment is best;
* probability of clinically important benefit or harm;
* rankograms;
* uncertainty intervals for absolute outcomes.

Rankings are accompanied by treatment-effect estimates and certainty assessments. Rankings are not presented as conclusive when effects are imprecise, incoherent or clinically similar.

---

## 13. Validation and Testing

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

## 14. Reproducibility and Reporting

Every analysis exports:

* data and treatment-network checks;
* model equations;
* likelihood and prior definitions;
* software version and random seed;
* sampler settings and diagnostics;
* convergence status;
* heterogeneity and incoherence results;
* bias assumptions;
* sensitivity analyses;
* absolute and relative treatment effects;
* ranking uncertainty;
* machine-readable results and an audit log.

The software must distinguish clearly between:

* observed data;
* derived quantities;
* imputed values;
* simulated data;
* reviewer-entered risk-of-bias judgements;
* algorithmically generated bias flags.
