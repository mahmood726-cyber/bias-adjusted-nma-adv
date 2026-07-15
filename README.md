# Advanced Bias-Adjusted Network Meta-Analysis (NMA) Benchmark

This repository contains the implementation and benchmarking framework for an advanced design-stratified network meta-analysis (NMA) pooling model. The model incorporates study-level covariate meta-regression, Hartung-Knapp-Sidik-Jonkman (HKSJ) covariance scaling, and scoped risk-of-bias sensitivity weighting.

The tier-one methods benchmark is documented in `docs/tier1_benchmark.md`. The current package should be read as a prototype plus validation harness, not as a claim that every native estimator already outperforms mature specialist software such as `netmeta`, `multinma`, `MBNMAdose`, or `crossnma`.

The portfolio reuse map in `docs/portfolio_reuse_map.md` identifies nearby repos that can strengthen the platform: `wasserstein` for OA Kaplan-Meier survival validation, `topo-transport-ma` for transportability support diagnostics, `allmeta` for netmeta parity fixtures, and `rct-extractor-v2` for source-backed ingestion patterns. These are planned inputs only; each contribution must be revalidated inside this repository before it supports a certification claim.

## Methods & Estimands

The package implements a frequentist contrast-based network meta-analysis model with the following advanced features:
1. **Design-Stratified Bias Adjustment:** Adjusts for systematic differences between evidence designs (e.g., Randomized Controlled Trials vs. Non-Randomized Studies).
2. **Context Meta-Regression:** Models treatment-by-covariate and design-by-covariate interactions to adjust for study-level confounders.
3. **Hartung-Knapp-Sidik-Jonkman (HKSJ) Covariance Scaling:** Adjusts parameter standard errors to reflect residual heterogeneity and prevent overconfidence under small-sample regimes.
4. **Scoped Down-Weighting:** Continuously inflates within-study variances based on study-level quality weights ($w_s \in (0, 1]$), decreasing the influence of high-risk studies.

### Primary Estimand
- **Bias-Adjusted Treatment Effect (log-odds ratio):** The relative treatment effect between active treatments and a reference treatment, adjusted for study design, quality, and meta-regression covariates.

## Repository Structure
- `src/bias_nma_adv/`: Core python package containing:
  - `data.py`: Structured data classes and validation schemas.
  - `ingestion.py`: Source-backed ingestion provenance checks for PubMed, ClinicalTrials.gov, and open-access paper rows.
  - `model.py`: Fitting engine implementing REML optimization, GLS with prior shrinkage, HKSJ covariance scaling, and down-weighting.
- `src/bias_nma_adv/pairwise.py`: Experimental pairwise meta-analysis bridge for FE, DL, Paule-Mandel, REML, HKSJ, and prediction-interval conventions.
- `src/bias_nma_adv/transportability.py`: Experimental effect-modifier transport meta-regression with collapsibility guards and topological support certificates.
- `src/bias_nma_adv/multiarm.py`: Experimental contrast-level NMA solver preserving multi-arm covariance for netmeta-style parity tests.
- `simulation.py`: Synthetic NMA dataset generator and benchmarking loop.
- `tests/`: Unit and integration test suite.
- `reproduce.py`: Benchmark script running 200 simulation iterations.
- `reproduce_real_meta.py`: Runs the source-backed SGLT2 heart-failure real-meta benchmark.
- `validation/reference_targets.toml`: Machine-readable reference targets required before tier-one parity or production certification claims.
- `validation/reference_runs/`: Machine-readable external reference-adapter run reports. Unavailable or failed reports cannot count as certification evidence.
- `validation/real_meta/`: Source-backed real meta-analysis fixtures constrained to ClinicalTrials.gov, PubMed abstracts, and open-access papers.
- `validation/source_checks/`: Public-source identity snapshots generated from ClinicalTrials.gov and PubMed APIs.
- `external/r/`: Optional R reference adapters for packages such as `metafor`, `meta`, and later tier-one NMA packages.
- `scripts/preflight_reference_adapters.py`: Regenerates external-adapter preflight reports without treating skipped reference software as a pass.
- `scripts/verify_real_meta_sources.py`: Regenerates live source-identity snapshots for real-meta manifests.
- `docs/portfolio_reuse_map.md`: Local portfolio scan describing reusable methods and hard stops before porting them.
- `e156-submission/`: Micro-publication artifacts.

## Benchmark Results

Running the simulation benchmark over 200 iterations (1,000 model fits) on synthetic datasets with star and loop topologies yields the following comparative performance for estimating treatment contrast `B vs A`:

| Method | Bias | RMSE | Coverage | Mean SE |
|---|---|---|---|---|
| Naive NMA | 0.1407 | 0.1988 | 83.50% | 0.1463 |
| Standard Bias-Adjusted NMA | 0.1407 | 0.1988 | 83.50% | 0.1463 |
| HKSJ Bias-Adjusted NMA | 0.1407 | 0.1988 | 87.00% | 0.1488 |
| HKSJ + Downweighted Bias-Adjusted NMA | 0.1187 | 0.1808 | 87.50% | 0.1413 |
| Full Advanced Bias-Adjusted NMA (HKSJ + Weighting + Regression) | **0.0925** | **0.1451** | **95.00%** | 0.1294 |

*Note: These are synthetic benchmark results for the repository's configured simulation design. They do not establish general superiority over specialist NMA packages; reference matching and broader simulation validation are required before any certification claim.*

## Quick Start

### Installation
Clone the repository and install in editable mode:
```bash
pip install -e .
```

### Running Tests
To execute the unit and integration tests:
```bash
pytest
```

### Running the Benchmark
To run the simulation and regenerate the benchmark metrics:
```bash
$env:PYTHONPATH="src"; python reproduce.py
```
This writes the full results payload to `output/simulation_results.json`.

## References
1. Hartung J, Knapp G. A refined method for the meta-analysis of random effects models with unequal variances. *Stat Med*. 2001;20(24):3875-3889.
2. Sidik K, Jonkman JN. Robust variance estimation for random effects meta-analysis. *Comput Stat Data Anal*. 2006;50(12):3681-3701.
3. Lu G, Ades AE. Combination of direct and indirect evidence in mixed treatment comparisons. *Stat Med*. 2004;23(21):3105-3124.
