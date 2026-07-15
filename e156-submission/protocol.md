# E156 Protocol: Advanced Bias-Adjusted Network Meta-Analysis (NMA) Benchmark

**Author:** Mahmood Ahmad, Tahir Heart Institute
**ORCID:** 0000-0001-9107-3704
**Registration Date:** 2026-07-15
**Repository:** https://github.com/mahmood726-cyber/bias-adjusted-nma-adv.git

## Objective

This repository contains the complete analysis code, data, and manuscript materials for developing and benchmarking advanced design-stratified network meta-analysis (NMA) models. Naive pooling methods are prone to design-level confounding and overconfidence under small-sample regimes. This project evaluates an advanced estimator combining meta-regression covariate interactions, Hartung-Knapp-Sidik-Jonkman covariance scaling, and continuous risk-of-bias down-weighting.

## Methods

We simulated NMA datasets spanning star and loop topologies with design-level bias (RCT vs NRS) and study-level covariates. The analytical pipeline is pre-specified in this protocol and implemented in Python using a frequentist GLS framework. The primary contrast of interest is the bias-adjusted treatment effect (log-odds ratio) between active treatments and a reference treatment. Performance metrics (bias, RMSE, interval coverage) are evaluated over 200 simulation iterations.

## Availability

- Code: https://github.com/mahmood726-cyber/bias-adjusted-nma-adv.git
- Dashboard: https://mahmood726-cyber.github.io/bias-adjusted-nma-adv/

---
**AI Disclosure Statement**
This work represents a compiler-generated evidence micro-publication. AI was used as a constrained synthesis engine operating on structured inputs and predefined rules. All results were reviewed and verified by the author.
