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
  - `ingestion.py`: Source-backed ingestion provenance checks and proof-carrying extracted-effect contracts for AACT/ClinicalTrials.gov, PubMed, open-access papers, rare result-level ICTRP/PACTR rows, and public FDA/EMA regulatory review rows with numeric per-trial provenance. Protocol-only registry rows cannot supply model-ready effects, but can feed registered-primary-outcome and completeness-denominator ledgers.
  - `model.py`: Fitting engine implementing REML optimization, GLS with prior shrinkage, HKSJ covariance scaling, and down-weighting.
- `src/bias_nma_adv/bayesian.py`: Experimental Metropolis-Hastings Bayesian sampler with local R-hat/ESS/MCSE diagnostic warnings, prior/posterior predictive checks, and draw-preserving treatment ranking; it is not a Stan/NUTS replacement.
- `src/bias_nma_adv/stan_backend.py`: Fail-closed CmdStanPy/CmdStan preflight contract exposing required NUTS diagnostics before any Stan-backed reference run can count.
- `src/bias_nma_adv/stan_reference_validation.py`: Validates the source-backed CmdStan/NUTS reference-output JSON for sampler diagnostics, source identifiers, and posterior mean alignment with the existing `metafor` fixed-effect reference.
- `external/stan/standard_binary_nma.stan`: Minimal fixed-effect arm-level binary NMA Stan model source used for the current source-backed CmdStan/NUTS SGLT2i reference candidate. This is not broad `multinma` parity.
- `src/bias_nma_adv/pairwise.py`: Experimental pairwise meta-analysis bridge for FE, DL, Paule-Mandel, REML, HKSJ, prediction-interval conventions, diagnostic-only leave-one-out and bounded exhaustive GOSH-style outlier-space screening, tau2 cross-checks, sparse/dominant-study stress reports, REML profile/local-minimum screens, stress-matrix reports, and bounded trim-and-fill sensitivity screening.
- `src/bias_nma_adv/redescending.py`: Default-off Tukey-biweight redescending pairwise sensitivity screen that reports transparent study weights; it is not a full redescending fraud-containment NMA core.
- `src/bias_nma_adv/sufficiency_fragility.py`: Deterministic E-value and binary event fragility-index sensitivity summaries for source-backed ratio estimates or event counts.
- `src/bias_nma_adv/rapidmeta_adapter.py`: Fail-closed importer for a strict RapidMeta-style app-index JSON contract; protocol-only registry rows and ambiguous multi-analysis exports are rejected before estimation.
- `src/bias_nma_adv/transportability.py`: Experimental effect-modifier transport meta-regression with collapsibility guards and topological support certificates.
- `src/bias_nma_adv/multiarm.py`: Experimental contrast-level NMA solver preserving multi-arm covariance for netmeta-style parity tests, with pre-fit design diagnostics, deterministic fit-attempt reports, diagnostic-only GLS leverage, residual, approximate Cook-distance, row-level/study-level/heatmap-ready absolute mapping-contribution, and fail-closed covariance-validity outputs.
- `src/bias_nma_adv/component_nma.py`: Narrow additive component-NMA core with weighted least-squares estimates and estimability checks, validated only on an algorithmic `netmeta::discomb` fixture so far.
- `src/bias_nma_adv/node_splitting.py`: Experimental fixed-effect node-splitting diagnostic for closed-loop direct-versus-indirect contrast checks.
- `src/bias_nma_adv/km_reconstruction.py`: Fail-closed screen, native Python Guyot-style reconstruction check, and deterministic curve-fidelity metrics for open-access Kaplan-Meier reconstruction results before they can enter survival validation artifacts.
- `src/bias_nma_adv/publication_bias.py`: Registry outcome-switching auditor plus diagnostic-only Egger small-study-effect regression and prespecified selection-weight sensitivity analysis.
- `src/bias_nma_adv/ctgov_hr_network.py`: Source-verified ClinicalTrials.gov reported-HR network benchmark support, including sparse design-bias diagnostics for underidentified cross-design borrowing.
- `src/bias_nma_adv/dose_response_benchmark.py`: Source-verified CT.gov/PubMed dose-response smoke benchmark support for dose-level effects and narrow `metafor` polynomial validation; it remains non-certifying until `MBNMAdose` reference matching.
- `src/bias_nma_adv/dose_response_coverage.py`: Fail-closed source-coverage check showing whether dose-response benchmark data are currently registered from AACT/ClinicalTrials.gov, ICTRP/PACTR result rows, PubMed abstracts, open-access papers, or public FDA/EMA regulatory review rows.
- `src/bias_nma_adv/dta.py`: Experimental bivariate logit-normal REML DTA prototype for algorithmic fixture validation against `mada::reitsma` and one source-backed open-access table; it is not clinical DTA evidence or certification.
- `src/bias_nma_adv/dta_coverage.py`: Fail-closed DTA coverage gate requiring source-backed TP/FP/FN/TN tables and bivariate GLMM/HSROC modeling before diagnostic-accuracy claims.
- `src/bias_nma_adv/benchmark_registry.py`: Machine-readable registry validator for local source-backed benchmark artifacts, source-check semantics, and pinned hashes.
- `src/bias_nma_adv/grand_benchmark_plan.py`: Validator for the prespecified grand-benchmark plan separating real source-backed lanes from simulation-only operating-characteristic scenarios.
- `src/bias_nma_adv/tier1_gap_register.py`: Validator for known blockers that prevent tier-one parity or superiority claims, including feature completeness, numerical stability, and Bayesian ecosystem integration.
- `src/bias_nma_adv/feature_parity_matrix.py`: Fine-grained parity ledger across netmeta, metafor/meta, multinma/Stan, MBNMAdose, crossnma, DTA, and evidence-certainty capabilities.
- `src/bias_nma_adv/large_scale_validation.py`: Dynamic gate comparing current source-backed benchmarks, reference runs, and simulation jobs against prespecified large-scale validation thresholds.
- `src/bias_nma_adv/html_delivery_contract.py`: Validator for what can be delivered as HTML without replacing statistical engines, source verification, reference adapters, or CI gates.
- `src/bias_nma_adv/improvement_review.py`: Validator for the current improvement/polish review milestone, keeping the strategic goal active while documenting what passed current review.
- `src/bias_nma_adv/simulation_matrix.py`: Validator and runner for executable, non-certifying simulation smoke jobs tied to the grand-benchmark plan.
- `src/bias_nma_adv/portfolio_reuse.py`: Local portfolio reuse registry and scanner for candidate methods/source-ingestion components; it is non-certifying and takes roots at runtime.
- `src/bias_nma_adv/proof_effect_bundle.py`: Validator and writer support for proof-carrying extracted-effect bundles that bind reported effects to verified source manifests, source-check reports, and minimal source snippets.
- `src/bias_nma_adv/real_benchmark_atlas.py`: Non-certifying coverage atlas for registered source-backed real benchmark artifacts, source-check scopes, and current evidence gaps.
- `src/bias_nma_adv/r_reference_validation.py`: Validates optional R reference-output JSON from `metafor`, `meta`, `netmeta`, and `mada` against local source-backed or algorithmic benchmark artifacts before any report can become an evidence candidate.
- `src/bias_nma_adv/review_ledger.py`: Machine-readable multiperson review ledger validator for source-boundary, statistical, implementation, and claims-governance review rounds.
- `src/bias_nma_adv/validation_status.py`: Unified validation-status report composing source-backed benchmark, reference-target, and reference-run gates without changing certification status.
- `simulation.py`: Synthetic NMA dataset generator and benchmarking loop.
- `tests/`: Unit and integration test suite.
- `reproduce.py`: Benchmark script running 200 simulation iterations.
- `reproduce_real_meta.py`: Runs the source-backed SGLT2 heart-failure real-meta benchmark.
- `validation/reference_targets.toml`: Machine-readable reference targets required before tier-one parity or production certification claims.
- `validation/grand_benchmark_plan.toml`: Prespecified source-bound validation plan for real-data lanes and simulation scenarios; it contains no benchmark results and has `certification_effect = "none"`.
- `validation/tier1_gap_register.toml`: Machine-readable blocker register for tier-one shortcomings. Current blockers are feature completeness, numerical stability, and Bayesian ecosystem breadth/reference parity.
- `validation/feature_parity_matrix.toml`: Fine-grained feature-parity matrix. It currently records reference candidates and local implementations, but no completed broad feature parity.
- `validation/large_scale_validation.toml`: Large-scale validation gate. It records thresholds and dynamically reports the current corpus as partial, not large-scale validation.
- `validation/html_delivery_contract.toml`: Machine-readable delivery contract for HTML dashboards versus backend-required engines and gates.
- `validation/dose_response_source_coverage.toml`: Machine-readable check for dose-response real-data coverage. Current status records one source-backed CT.gov/PubMed semaglutide dose-response smoke benchmark plus a narrow `metafor` polynomial reference candidate, and still blocks dose-response NMA parity or superiority claims.
- `validation/dta_source_coverage.toml`: Machine-readable DTA coverage gate. Current status records one source-backed open-access DTA 2x2 benchmark (`midkine_elisa_cancer_dta`) and one narrow `mada::reitsma` source-table smoke reference, while still blocking broad HSROC parity, clinical DTA validation, and superiority claims.
- `validation/simulation_matrix.toml`: Executable simulation smoke matrix. It uses no real data and cannot support clinical or tier-one superiority claims.
- `validation/portfolio_reuse_sources.toml`: Machine-readable inventory of local portfolio repositories worth inspecting for reusable code patterns. It stores repo names and relative assets only, not local absolute paths.
- `validation/reference_runs/`: Machine-readable external reference-adapter run reports. Unavailable or failed reports cannot count as certification evidence.
- `validation/reviews/`: Non-certifying review ledgers recording multiperson review findings, actions, and next gates.
- `validation/reviews/improvement_review_2026_07_15.toml`: Current improvement-review ledger for tier-one blockers, source boundaries, HTML delivery, and implementation polish. It passes the current milestone but explicitly keeps the global goal incomplete.
- `validation/multiarm/`: Governed multi-arm GLS fixture data and local replay benchmark. These artifacts are algorithmic fixtures, not clinical evidence or `netmeta` certification.
- `validation/component/`: Algorithmic additive component-NMA contrast fixture and local replay benchmark. These artifacts are not source-backed clinical evidence and do not remove the real-data component-NMA blocker.
- `validation/real_meta/`: Source-backed real meta-analysis fixtures constrained to AACT/ClinicalTrials.gov, PubMed abstracts, open-access papers, rare public result-level ICTRP/PACTR rows, and public FDA/EMA regulatory review rows when numeric per-trial result text is source-bound. Protocol-only registry rows remain metadata only.
- `validation/benchmark_registry.toml`: Canonical inventory of local source-backed benchmark artifacts. Every entry must retain `certification_effect = "none"` until an external reference run passes, and every source-check report is revalidated against its specialized schema.
- `validation/source_checks/`: Public-source identity snapshots and PubMed abstract event-count token checks.
- `validation/ingestion/`: Proof-carrying extracted-effect bundles. These are model-ingestion evidence contracts only and carry `certification_effect = "none"`.
- `validation/real_benchmark_atlas.json`: Generated coverage atlas over the registered real-data benchmark artifacts. It records current source-backed coverage and explicit non-claims; it is not certification evidence.
- `validation/survival/`: Source-backed survival HR benchmark manifests and effects CSVs; current entries verify reported HR tokens from PubMed abstracts and support narrow `metafor` fixed-effect reported-HR checks. KM reconstruction remains non-certifying and requires OA figure provenance before any real KM artifact can enter validation.
- `validation/dose_response/`: Source-backed semaglutide dose-response manifest, effects CSV, and generated local benchmark artifact from ClinicalTrials.gov results plus PubMed identity verification.
- `validation/dta/`: Source-backed open-access diagnostic test accuracy 2x2 manifest, CSV, and generated local benchmark artifact for the Midkine ELISA cancer diagnostic table.
- `validation/networks/`: Source-backed network benchmark manifests, effects CSVs, and generated artifacts, including the CT.gov T2D MACE reported-HR star network with a narrow `netmeta` fixed-effect reference check.
- `validation/survival/km_reconstruction_policy.toml`: Static OA-only KM reconstruction policy; blocks text-only HR and synthetic-IPD fallbacks from validation evidence.
- `external/r/`: Optional R reference adapters for packages such as `metafor`, `meta`, `netmeta`, and `mada`; current component-NMA smoke validation uses `netmeta::discomb`, current dose-response smoke validation uses `metafor`, while later targets include `multinma`, `MBNMAdose`, and `crossnma`.
- `scripts/preflight_reference_adapters.py`: Regenerates external-adapter preflight reports without treating skipped reference software as a pass.
- `scripts/preflight_stan_nuts_adapter.py`: Regenerates the non-certifying CmdStan/NUTS preflight report for the committed Stan model source.
- `scripts/run_stan_nuts_reference.py`: Runs the source-backed SGLT2i CmdStan/NUTS reference candidate and writes both JSON output and a hashed reference-run report. It does not certify `multinma` or broad Bayesian parity.
- `scripts/preflight_multiarm_netmeta_adapter.py`: Regenerates the non-certifying `netmeta` multi-arm adapter preflight report.
- `scripts/preflight_dta_mada_adapter.py`: Regenerates the non-certifying `mada::reitsma` DTA adapter preflight report.
- `scripts/validate_r_reference_outputs.py`: Validates generated R reference-output JSON and writes separate passed candidate reports with output hashes and tolerances.
- `scripts/validate_benchmark_registry.py`: Validates every registered local source-backed benchmark and emits an optional JSON summary for CI/Overmind-style gates.
- `scripts/run_simulation_matrix.py`: Runs the non-certifying simulation matrix and emits a JSON report for operating-characteristic smoke checks.
- `scripts/scan_portfolio_reuse.py`: Scans user-supplied local roots for registered portfolio reuse candidates and reports dirty worktrees, missing assets, and import-review status.
- `scripts/write_proof_effect_bundle.py`: Regenerates the SGLT2 reported-HR proof-carrying extracted-effect bundle from verified manifests and live PubMed abstract text.
- `scripts/write_real_benchmark_atlas.py`: Regenerates the non-certifying real benchmark coverage atlas from the source-backed benchmark registry.
- `scripts/write_validation_status.py`: Emits the combined validation status JSON for CI/Overmind-style gates and can optionally verify external source-artifact hash pins. Current reports keep clinical and HTA reporting disabled unless a module is Production Certified.
- `scripts/verify_reversal_source_artifacts.py`: Dedicated external answer-key pin verifier. Provided artifacts fail closed on hash drift, and missing artifacts are reported as `unavailable` unless explicitly allowed by the caller.
- `scripts/verify_real_meta_sources.py`: Regenerates live source-identity snapshots for real-meta manifests.
- `scripts/verify_pubmed_event_counts.py`: Regenerates PubMed abstract event-count token snapshots for real-meta arm counts.
- `scripts/verify_survival_sources.py`: Regenerates CT.gov/PubMed identity snapshots for survival reported-HR manifests.
- `scripts/verify_pubmed_survival_hrs.py`: Regenerates PubMed abstract reported-HR token snapshots for survival benchmark manifests.
- `scripts/write_multiarm_benchmark.py`: Writes the deterministic multi-arm GLS fixture replay artifact from the governed arm-level CSV.
- `scripts/write_survival_hr_benchmark.py`: Writes the deterministic reported-HR pairwise benchmark artifact from a verified survival source snapshot.
- `scripts/verify_ctgov_hr_network.py`: Regenerates CT.gov reported-HR source snapshots for network benchmark manifests.
- `scripts/write_ctgov_hr_network_benchmark.py`: Writes the deterministic CT.gov reported-HR network benchmark artifact from a verified CT.gov source snapshot.
- `scripts/verify_dose_response_sources.py`: Regenerates live CT.gov/PubMed source checks for the source-backed dose-response manifest and exits nonzero on failed verification.
- `scripts/write_dose_response_benchmark.py`: Writes the deterministic non-certifying dose-response smoke benchmark from a verified source check.
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
