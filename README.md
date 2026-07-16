# Advanced Bias-Adjusted Network Meta-Analysis (NMA) Benchmark

This repository contains the implementation and benchmarking framework for an advanced design-stratified network meta-analysis (NMA) pooling model. The model incorporates study-level covariate meta-regression, Hartung-Knapp-Sidik-Jonkman (HKSJ) covariance scaling, and scoped risk-of-bias sensitivity weighting.

The tier-one methods benchmark is documented in `docs/tier1_benchmark.md`. The current package should be read as a prototype plus validation harness, not as a claim that every native estimator already outperforms mature specialist software such as `netmeta`, `multinma`, `MBNMAdose`, or `crossnma`.

The portfolio reuse map in `docs/portfolio_reuse_map.md` identifies nearby repos that can strengthen the platform: `wasserstein` for OA Kaplan-Meier survival validation, `topo-transport-ma` for transportability support diagnostics, `allmeta` for netmeta parity fixtures, and `rct-extractor-v2` for source-backed ingestion patterns. These are planned inputs only; each contribution must be revalidated inside this repository before it supports a certification claim.

Historical adversarial-review and journal-review notes in `docs/` are governed by `docs/review_artifact_policy.md`; they are critique artifacts, not validation evidence or clinical guidance.

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
  - `ingestion.py`: Source-backed ingestion provenance checks and proof-carrying extracted-effect contracts for PubMed, ClinicalTrials.gov, and open-access paper rows. WHO ICTRP and other registries are protocol-only metadata sources and cannot supply model-ready effects.
  - `model.py`: Fitting engine implementing REML optimization, GLS with prior shrinkage, HKSJ covariance scaling, and down-weighting.
- `src/bias_nma_adv/pairwise.py`: Experimental pairwise meta-analysis bridge for FE, DL, Paule-Mandel, REML, HKSJ, and prediction-interval conventions.
- `src/bias_nma_adv/transportability.py`: Experimental effect-modifier transport meta-regression with collapsibility guards and topological support certificates.
- `src/bias_nma_adv/multiarm.py`: Experimental contrast-level NMA solver preserving multi-arm covariance for netmeta-style parity tests, with pre-fit design diagnostics, deterministic fit-attempt reports, diagnostic-only GLS leverage, residual, approximate Cook-distance, absolute mapping-contribution, and fail-closed covariance-validity outputs.
- `src/bias_nma_adv/node_splitting.py`: Experimental fixed-effect node-splitting diagnostic for closed-loop direct-versus-indirect contrast checks.
- `src/bias_nma_adv/km_reconstruction.py`: Fail-closed screen and deterministic curve-fidelity metrics for open-access Kaplan-Meier reconstruction results before they can enter survival validation artifacts.
- `src/bias_nma_adv/publication_bias.py`: Registry outcome-switching auditor plus diagnostic-only Egger small-study-effect regression.
- `src/bias_nma_adv/ctgov_hr_network.py`: Source-verified ClinicalTrials.gov reported-HR network benchmark support.
- `src/bias_nma_adv/benchmark_registry.py`: Machine-readable registry validator for local source-backed benchmark artifacts, source-check semantics, and pinned hashes.
- `src/bias_nma_adv/grand_benchmark_plan.py`: Validator for the prespecified grand-benchmark plan separating real source-backed lanes from simulation-only operating-characteristic scenarios.
- `src/bias_nma_adv/tier1_gap_register.py`: Validator for known blockers that prevent tier-one parity or superiority claims, including feature completeness, numerical stability, and Bayesian ecosystem integration.
- `src/bias_nma_adv/html_delivery_contract.py`: Validator for what can be delivered as HTML without replacing statistical engines, source verification, reference adapters, or CI gates.
- `src/bias_nma_adv/improvement_review.py`: Validator for the current improvement/polish review milestone, keeping the strategic goal active while documenting what passed current review.
- `src/bias_nma_adv/simulation_matrix.py`: Validator and runner for executable, non-certifying simulation smoke jobs tied to the grand-benchmark plan.
- `src/bias_nma_adv/portfolio_reuse.py`: Local portfolio reuse registry and scanner for candidate methods/source-ingestion components; it is non-certifying and takes roots at runtime.
- `src/bias_nma_adv/proof_effect_bundle.py`: Validator and writer support for proof-carrying extracted-effect bundles that bind reported effects to verified source manifests, source-check reports, and minimal source snippets.
- `src/bias_nma_adv/real_benchmark_atlas.py`: Non-certifying coverage atlas for registered source-backed real benchmark artifacts, source-check scopes, and current evidence gaps.
- `src/bias_nma_adv/r_reference_validation.py`: Validates optional R reference-output JSON from `metafor`, `meta`, and `netmeta` against local source-backed or algorithmic benchmark artifacts before any report can become an evidence candidate.
- `src/bias_nma_adv/review_ledger.py`: Machine-readable multiperson review ledger validator for source-boundary, statistical, implementation, and claims-governance review rounds.
- `src/bias_nma_adv/validation_status.py`: Unified validation-status report composing source-backed benchmark, reference-target, and reference-run gates without changing certification status.
- `simulation.py`: Synthetic NMA dataset generator and benchmarking loop.
- `tests/`: Unit and integration test suite.
- `reproduce.py`: Benchmark script running 200 simulation iterations.
- `reproduce_real_meta.py`: Runs the source-backed SGLT2 heart-failure real-meta benchmark.
- `validation/reference_targets.toml`: Machine-readable reference targets required before tier-one parity or production certification claims.
- `validation/grand_benchmark_plan.toml`: Prespecified source-bound validation plan for real-data lanes and simulation scenarios; it contains no benchmark results and has `certification_effect = "none"`.
- `validation/tier1_gap_register.toml`: Machine-readable blocker register for tier-one shortcomings. Current blockers are feature completeness, numerical stability, and Bayesian/Stan ecosystem integration.
- `validation/html_delivery_contract.toml`: Machine-readable delivery contract for HTML dashboards versus backend-required engines and gates.
- `validation/simulation_matrix.toml`: Executable simulation smoke matrix. It uses no real data and cannot support clinical or tier-one superiority claims.
- `validation/portfolio_reuse_sources.toml`: Machine-readable inventory of local portfolio repositories worth inspecting for reusable code patterns. It stores repo names and relative assets only, not local absolute paths.
- `validation/reference_runs/`: Machine-readable external reference-adapter run reports. Unavailable or failed reports cannot count as certification evidence.
- `validation/reviews/`: Non-certifying review ledgers recording multiperson review findings, actions, and next gates.
- `validation/reviews/improvement_review_2026_07_15.toml`: Current improvement-review ledger for tier-one blockers, source boundaries, HTML delivery, and implementation polish. It passes the current milestone but explicitly keeps the global goal incomplete.
- `validation/multiarm/`: Governed multi-arm GLS fixture data and local replay benchmark. These artifacts are algorithmic fixtures, not clinical evidence or `netmeta` certification.
- `validation/real_meta/`: Source-backed real meta-analysis fixtures constrained to ClinicalTrials.gov, PubMed abstracts, and open-access papers. Protocol-only registries such as WHO ICTRP may support registration metadata only.
- `validation/benchmark_registry.toml`: Canonical inventory of local source-backed benchmark artifacts. Every entry must retain `certification_effect = "none"` until an external reference run passes, and every source-check report is revalidated against its specialized schema.
- `validation/source_checks/`: Public-source identity snapshots and PubMed abstract event-count token checks.
- `validation/ingestion/`: Proof-carrying extracted-effect bundles. These are model-ingestion evidence contracts only and carry `certification_effect = "none"`.
- `validation/real_benchmark_atlas.json`: Generated coverage atlas over the registered real-data benchmark artifacts. It records current source-backed coverage and explicit non-claims; it is not certification evidence.
- `validation/survival/`: Source-backed survival HR benchmark manifests; current entries verify reported HR tokens from PubMed abstracts and explicitly do not claim KM reconstruction yet.
- `validation/networks/`: Source-backed network benchmark manifests and generated artifacts, including the CT.gov T2D MACE reported-HR star network.
- `validation/survival/km_reconstruction_policy.toml`: Static OA-only KM reconstruction policy; blocks text-only HR and synthetic-IPD fallbacks from validation evidence.
- `external/r/`: Optional R reference adapters for packages such as `metafor`, `meta`, and later tier-one NMA packages.
- `scripts/preflight_reference_adapters.py`: Regenerates external-adapter preflight reports without treating skipped reference software as a pass.
- `scripts/preflight_multiarm_netmeta_adapter.py`: Regenerates the non-certifying `netmeta` multi-arm adapter preflight report.
- `scripts/validate_r_reference_outputs.py`: Validates generated R reference-output JSON and writes separate passed candidate reports with output hashes and tolerances.
- `scripts/validate_benchmark_registry.py`: Validates every registered local source-backed benchmark and emits an optional JSON summary for CI/Overmind-style gates.
- `scripts/run_simulation_matrix.py`: Runs the non-certifying simulation matrix and emits a JSON report for operating-characteristic smoke checks.
- `scripts/scan_portfolio_reuse.py`: Scans user-supplied local roots for registered portfolio reuse candidates and reports dirty worktrees, missing assets, and import-review status.
- `scripts/write_proof_effect_bundle.py`: Regenerates the SGLT2 reported-HR proof-carrying extracted-effect bundle from verified manifests and live PubMed abstract text.
- `scripts/write_real_benchmark_atlas.py`: Regenerates the non-certifying real benchmark coverage atlas from the source-backed benchmark registry.
- `scripts/write_validation_status.py`: Emits the combined validation status JSON for CI/Overmind-style gates. Current reports keep clinical and HTA reporting disabled unless a module is Production Certified.
- `scripts/verify_real_meta_sources.py`: Regenerates live source-identity snapshots for real-meta manifests.
- `scripts/verify_pubmed_event_counts.py`: Regenerates PubMed abstract event-count token snapshots for real-meta arm counts.
- `scripts/verify_survival_sources.py`: Regenerates CT.gov/PubMed identity snapshots for survival reported-HR manifests.
- `scripts/verify_pubmed_survival_hrs.py`: Regenerates PubMed abstract reported-HR token snapshots for survival benchmark manifests.
- `scripts/write_multiarm_benchmark.py`: Writes the deterministic multi-arm GLS fixture replay artifact from the governed arm-level CSV.
- `scripts/write_survival_hr_benchmark.py`: Writes the deterministic reported-HR pairwise benchmark artifact from a verified survival source snapshot.
- `scripts/verify_ctgov_hr_network.py`: Regenerates CT.gov reported-HR source snapshots for network benchmark manifests.
- `scripts/write_ctgov_hr_network_benchmark.py`: Writes the deterministic CT.gov reported-HR network benchmark artifact from a verified CT.gov source snapshot.
- `.github/workflows/validation.yml`: CI workflow running pytest, the source-backed benchmark registry gate, the unified validation-status writer, and the simulation-matrix smoke runner. Uploaded artifacts are evidence of gate execution, not certification.
- `docs/portfolio_reuse_map.md`: Local portfolio scan describing reusable methods and hard stops before porting them.
- `docs/review_artifact_policy.md`: Scope guard for historical review transcripts that contain hypotheses or critique, not certified evidence.
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
