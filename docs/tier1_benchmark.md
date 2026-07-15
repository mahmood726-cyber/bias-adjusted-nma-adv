<!-- sentinel:skip-file -->
# Tier-One Methods Benchmark

Checked: 2026-07-15

This note benchmarks the platform specification against mature specialist network meta-analysis methods. It is not evidence that the current Python implementation is superior to those methods.

## Bottom Line

The platform can plausibly become stronger than any single specialist package as an integrated evidence-synthesis environment. It should not claim that every native estimator beats the best specialist method in its domain until reference matching, simulation validation, and independent reproduction have been completed.

The correct claim is:

> Use tier-one methods as foundations and validation targets, then compete through integration, governance, diagnostics, reproducibility, and carefully validated methodological improvements.

The current blocker register is `validation/tier1_gap_register.toml`. It records three active blockers from multiperson review: feature completeness, numerical stability, and Bayesian ecosystem integration. While any of these remain `blocking`, the repository must not claim tier-one parity, tier-one superiority, production certification, clinical reporting, regulatory reporting, or HTA reporting.

The HTML delivery contract is `validation/html_delivery_contract.toml`. HTML dashboards are allowed for presenting and inspecting generated artifacts, but statistical estimation, live source verification, R/Stan/reference adapters, hash-producing validation, and CI/certification gates remain backend-required.

The current improvement review is `validation/reviews/improvement_review_2026_07_15.toml`. It records a four-role review of the tier-one blockers, source boundary, HTML delivery risk, and implementation polish. The review passes the current milestone, but explicitly does not mark the global objective complete.

## Specialist Reference Targets

| Domain | Strong reference method or package | What must be matched first | Current platform position |
| --- | --- | --- | --- |
| Frequentist NMA | `netmeta` | Treatment effects, multi-arm covariance, inconsistency diagnostics, sparse-event handling, contribution matrices, disconnected-network warnings | Spec target; current repository has a small frequentist prototype plus a local multi-arm fixture replay artifact, not full parity |
| Bayesian NMA | `multinma` and validated CmdStan models | Posterior means, credible intervals, rank probabilities, diagnostics, prior sensitivity | Spec target; current repository does not yet provide production Stan reference matching |
| ML-NMR | `multinma` | IPD plus aggregate likelihood integration, population standardization, covariate-distribution checks | Spec target only |
| Dose-response NMA | `MBNMAdose` | Functional dose-response families, class effects, dose extrapolation checks, consistency diagnostics | Spec target only |
| Cross-design RCT/NRS synthesis | `crossnma` | Bayesian cross-format and cross-design models for RCT/NRS and IPD/aggregate mixtures | Spec target only |
| Component NMA | `netmeta` additive CNMA and recent CNMA hierarchy workflows | Estimability checks, component additivity diagnostics, hierarchy constraints | Spec target only |
| Evidence certainty | CINeMA and ROB-MEN | Contribution-aware certainty and missing-evidence assessment | Spec target only |

## Where The Platform Can Beat Specialist Tools

1. Integration: one workflow can cover data validation, estimand definition, network checks, model fitting, bias analysis, registry comparison, certainty assessment, and reproducible reporting.
2. Governance: every module can carry an explicit certification level from Experimental through Production Certified.
3. Safety: complex models can be blocked when the data structure is disconnected, unidentified, sparse, or outside the model's support.
4. Bias-aware synthesis: probabilistic bias parameters, RCT/NRS separation, missing-evidence analysis, registry auditing, contribution matrices, and certainty reporting can be handled together rather than as disconnected post-processing.
5. Method routing: the platform can recommend the appropriate specialist class instead of blindly running the most complex model.

## Where It Cannot Claim Universal Superiority

There is no single estimator that is best under every network, outcome, and evidence-design structure.

Failure modes that must remain explicit:

- unstructured multivariate covariance can be worse than univariate analyses in small sparse networks;
- ML-NMR can mislead when aggregate covariate distributions are reconstructed incorrectly;
- CNMA can mislead when component additivity fails or disconnected networks depend on strong structural assumptions;
- dose-response models can improve precision while adding bias if the functional form is wrong;
- cross-design borrowing can degrade RCT estimates when observational bias is misspecified;
- narrower intervals are not proof of better inference.

## Staged Validation Strategy

The executable reference-target registry is `validation/reference_targets.toml`. It is intentionally conservative: every target starts as `planned`, and certification tests fail closed if a module is promoted without evidence artifacts. The grand-benchmark plan in `validation/grand_benchmark_plan.toml` separates real source-backed validation lanes from simulation-only operating-characteristic scenarios. Real-world effect evidence must come from ClinicalTrials.gov records, PubMed abstracts, or open-access papers only. WHO ICTRP and other registries may be used for protocol-only metadata such as registration, planned outcomes, eligibility criteria, and dates, but they cannot supply model-ready effects under the current contracts.

External reference adapters must also emit machine-readable run reports under `validation/reference_runs/`. A skipped or unavailable adapter is recorded as `certification_effect = "none"` and cannot count as reference-matching evidence. A report becomes a certification candidate only when it has `status = "passed"`, package versions, input and output artifact hashes, and a prespecified tolerance.

Real-data benchmarks must also be paired with source snapshots under `validation/source_checks/`. Source-identity snapshots verify that public ClinicalTrials.gov and PubMed records are reachable and match the manifest identifiers. Event-count and reported-HR token snapshots, when present, separately verify exact numeric tokens and nearby treatment terms in PubMed abstracts; they are still not a substitute for full open-access paper/table extraction, Kaplan-Meier digitization, or external reference-software parity. Source-check artifacts must carry `certification_effect = "none"`.

The canonical inventory for local source-backed real-data benchmarks is `validation/benchmark_registry.toml`, validated by `src/bias_nma_adv/benchmark_registry.py`. This registry pins artifact hashes, source-manifest hashes, source-check hashes, allowed evidence-source policy, source mode, required limitations, and `certification_effect = "none"`. It also revalidates source-check reports through their specialized schemas and checks that those reports reference the registered source manifests. A benchmark can be useful regression evidence while still being barred from tier-one parity or production certification until a separate external reference run passes.

Run `python scripts/validate_benchmark_registry.py --json` to execute this gate outside pytest and emit a machine-readable summary. A passing registry summary still reports `certification_effect = "none"`. Run `python scripts/write_real_benchmark_atlas.py` to regenerate the coverage atlas over registered real-data benchmark artifacts.

Run `python scripts/write_validation_status.py` to compose the source-backed benchmark registry, real benchmark atlas, reference-target registry, simulation-matrix metadata, and reference-adapter run reports into one machine-readable validation status report. The report is deliberately conservative: evidence-candidate reference-output reports do not change module certification status, and clinical/HTA reporting remains disabled when no module holds Production Certified status. Run `python scripts/run_simulation_matrix.py` to execute the non-certifying simulation smoke matrix. The same gates run in `.github/workflows/validation.yml`, which uploads the status JSON, real benchmark atlas, and simulation report as CI artifacts.

Current adapter preflights: `validation/reference_runs/pairwise_metafor_meta_preflight.toml` and `validation/reference_runs/multiarm_netmeta_preflight.toml` record dependency availability only. They do not execute parity adapters and remain `certification_effect = "none"`.

Current local R reference-output validations: `validation/reference_runs/pairwise_metafor_meta_reference.toml` validates the generated `metafor`/`meta` JSON output for study log-OR effects, fixed-effect pooling, REML tau2/Q/df, and the documented HKSJ floor difference. `validation/reference_runs/multiarm_netmeta_reference.toml` validates generated `netmeta` JSON output for the multi-arm fixed/random fixture estimates and standard errors. These reports are `evidence_candidate` artifacts for narrow reference matching checks only; the broad reference targets in `validation/reference_targets.toml` remain `planned`.

Current review ledger: `validation/reviews/multiperson_review_2026_07_15.toml` records source-boundary, statistical-methods, implementation-contract, and claims-governance review rounds. It is a process-control artifact only and carries `certification_effect = "none"`.

Current local multi-arm artifact: `validation/multiarm/netmeta_portfolio_multiarm_benchmark.toml` deterministically replays the governed arm-level fixture in `validation/multiarm/netmeta_portfolio_multiarm_arms.csv`. It is useful for regression testing multi-arm covariance handling, but it carries `certification_effect = "none"` until an external `netmeta` adapter actually runs and passes with package versions, output hashes, and a prespecified tolerance.

Current KM reconstruction guard: `validation/survival/km_reconstruction_policy.toml` and `src/bias_nma_adv/km_reconstruction.py` define the survival lane's first fail-closed screen and curve-fidelity metric layer. Open-access paper figure provenance is required, text-only HR outputs are rejected, synthetic IPD fallbacks are blocked, and KM curve comparison metrics are non-certifying before a result can enter a validation artifact.

Current proof-carrying ingestion bundle: `validation/ingestion/sglt2_hf_reported_hr_proof_effects.json` records four SGLT2 reported-HR effects as source-backed extracted-effect rows. The bundle is regenerated by `scripts/write_proof_effect_bundle.py`, verifies source-manifest/report hashes, validates PubMed abstract snippets containing the HR and CI tokens, and still carries `certification_effect = "none"` because source provenance does not prove model performance or tier-one parity.

Current real network artifact: `validation/networks/t2d_mace_ctgov_hr_network_benchmark.toml` recomputes a ClinicalTrials.gov-reported HR star network for type 2 diabetes cardiovascular outcome trials from `validation/networks/t2d_mace_ctgov_hrs.toml` and `validation/source_checks/t2d_mace_ctgov_hr_network_check.json`. It is useful as a source-governed multi-treatment regression test, but it carries `certification_effect = "none"` because the network is placebo-centered, has no closed loops for inconsistency assessment, uses analyst-defined class labels, and has not passed external `netmeta`/`multinma` reference matching.

### Phase 1: Match Tier One

For each module, reproduce the established implementation before claiming improvement.

Acceptance criteria:

- deterministic estimates match within a prespecified numerical tolerance;
- Bayesian posterior differences are no larger than expected Monte Carlo error;
- multi-arm covariance handling is equivalent or deliberately documented;
- disconnected and unidentified networks produce equivalent warnings or fail-closed behavior;
- differences from reference software are documented as intentional, tested, and clinically interpretable.

### Phase 2: Beat Them Operationally

After reference matching, compete through:

- automatic schema recognition;
- pre-fit estimand checks;
- model-routing diagnostics;
- fail-closed data validation;
- parallel Bayesian and frequentist sensitivity analyses;
- reproducible execution bundles;
- contribution-based certainty assessment;
- registry and missing-evidence integration.

### Phase 3: Beat Selected Methods Statistically

Only after phases 1 and 2 should native methodological improvements be claimed. Candidate improvements must be tested prospectively against reference methods across prespecified simulations for:

- bias;
- RMSE;
- interval coverage;
- calibration;
- convergence failure;
- runtime;
- robustness to misspecification.

## Grand Benchmark Design

A credible superiority benchmark should include thousands of scenarios across:

- binary, continuous, count, ordinal, multinomial, and survival outcomes;
- small and large networks;
- multi-arm trials;
- rare and zero events;
- low and high heterogeneity;
- coherent and incoherent networks;
- IPD plus aggregate evidence;
- effect-modifier imbalance;
- disconnected networks;
- multicomponent treatments;
- dose-response misspecification;
- RCT/NRS conflicts;
- missing outcomes and publication bias.

Winners should be reported separately for statistical accuracy, uncertainty calibration, robustness, runtime, diagnostic quality, reproducibility, and user safety.

## Static-Vs-Dynamic Hardcode Disclosure

| Item | Static or dynamic | Evidence source | Disclosure |
| --- | --- | --- | --- |
| Reference package list | Static | Public package documentation checked on 2026-07-15 | Used as validation targets, not embedded computation |
| Package versions | Dynamic | CRAN or package site metadata checked on 2026-07-15 | May drift and must be refreshed before release claims |
| Platform capability status | Static repository review | Current repository files and tests checked on 2026-07-15 | Current implementation is not certified as tier-one parity |
| Benchmark thresholds | Static specification | This document and `docs/technical_specification.md` | Must be implemented as machine-readable validation criteria before certification |
| Reference-run preflight | Dynamic local environment check | `validation/reference_runs/*_preflight.toml` | Failed or unavailable preflights are recorded but cannot certify a module |
| R reference-output validation | Dynamic local R execution plus Python output comparison | `validation/reference_runs/*_output.json`, `validation/reference_runs/*_reference.toml`, `src/bias_nma_adv/r_reference_validation.py`, and `tests/test_r_reference_validation.py` | Validates narrow local parity artifacts with package versions, output hashes, and tolerance; does not promote broad targets or enable clinical/HTA reporting |
| Multi-arm fixture replay | Static algorithmic fixture plus dynamic local recomputation | `validation/multiarm/*` and `tests/test_multiarm_artifact.py` | Validates local covariance handling on a small fixture; not clinical evidence and not external `netmeta` parity |
| KM reconstruction policy and metrics | Static screen plus local unit fixtures | `validation/survival/km_reconstruction_policy.toml`, `src/bias_nma_adv/km_reconstruction.py`, and `tests/test_km_reconstruction_policy.py` | Blocks text-only and synthetic-IPD survival outputs and computes non-certifying KM curve-fidelity metrics; does not digitize any OA curve yet |
| CT.gov reported-HR network | Static manifest plus dynamic CT.gov source snapshot | `validation/networks/t2d_mace_ctgov_hrs.toml`, `validation/source_checks/t2d_mace_ctgov_hr_network_check.json`, and `tests/test_ctgov_hr_network.py` | Verifies public CT.gov HR/CI records and recomputes a local star-network GLS artifact; not reference matching or clinical guidance |
| Proof-carrying ingestion contract | Static source-row/effect-row contract plus unit fixtures | `src/bias_nma_adv/ingestion.py` and `tests/test_ingestion.py` | Requires source-backed identifiers, snippets, uncertainty, and ratio-scale sanity before extracted effects can enter models; not numeric extraction validation |
| Protocol-only registry boundary | Static source-policy contract plus unit fixtures | `src/bias_nma_adv/evidence_sources.py`, `src/bias_nma_adv/ingestion.py`, `validation/grand_benchmark_plan.toml`, and `tests/test_evidence_sources.py` | Allows WHO ICTRP and other registries for protocol metadata only; these sources are rejected as model-ready effect evidence |
| Proof-carrying reported-HR bundle | Dynamic PubMed abstract read plus static manifest/report hash checks | `validation/ingestion/sglt2_hf_reported_hr_proof_effects.json`, `scripts/write_proof_effect_bundle.py`, and `tests/test_proof_effect_bundle.py` | Validates four SGLT2 reported HR extracted-effect rows against source manifests, source-check hashes, and minimal HR/CI snippets; not model-performance certification |
| Source-backed benchmark registry | Static registry plus dynamic hash checks | `validation/benchmark_registry.toml` and `tests/test_benchmark_registry.py` | Provides the canonical inventory of local real-data benchmark artifacts and enforces non-certification status until external reference artifacts pass |
| Real benchmark coverage atlas | Dynamic registry-derived summary | `validation/real_benchmark_atlas.json`, `src/bias_nma_adv/real_benchmark_atlas.py`, and `tests/test_real_benchmark_atlas.py` | Summarizes registered real-data benchmark coverage, source-check scopes, and non-claims; it is not reference matching or clinical certification |
| Grand benchmark plan | Static source-bound plan plus dynamic registry cross-check | `validation/grand_benchmark_plan.toml`, `src/bias_nma_adv/grand_benchmark_plan.py`, and `tests/test_grand_benchmark_plan.py` | Separates real source-backed lanes from simulation scenarios; contains no results and cannot certify superiority |
| Tier-one gap register | Static blocker register plus validator | `validation/tier1_gap_register.toml`, `src/bias_nma_adv/tier1_gap_register.py`, and `tests/test_tier1_gap_register.py` | Keeps feature completeness, numerical stability, and Bayesian ecosystem limitations as active blockers until machine-verifiable promotion evidence exists |
| HTML delivery contract | Static delivery contract plus validator | `validation/html_delivery_contract.toml`, `src/bias_nma_adv/html_delivery_contract.py`, and `tests/test_html_delivery_contract.py` | Allows HTML dashboards for read-only artifact inspection but blocks HTML-only replacement of engines, source verification, adapters, hashes, and certification gates |
| Improvement review ledger | Static milestone review plus validator | `validation/reviews/improvement_review_2026_07_15.toml`, `src/bias_nma_adv/improvement_review.py`, and `tests/test_improvement_review.py` | Records that the current governance/polish milestone passed role-based review while the global world-class objective remains incomplete |
| Simulation matrix | Static simulation matrix plus dynamic smoke run | `validation/simulation_matrix.toml`, `src/bias_nma_adv/simulation_matrix.py`, `scripts/run_simulation_matrix.py`, and `tests/test_simulation_matrix.py` | Runs executable operating-characteristic smoke checks; uses no real data and cannot certify real clinical performance |
| Portfolio reuse registry | Static repo-name/relative-asset registry plus optional local root scan | `validation/portfolio_reuse_sources.toml`, `src/bias_nma_adv/portfolio_reuse.py`, `scripts/scan_portfolio_reuse.py`, and `tests/test_portfolio_reuse.py` | Records reusable local methods and source-ingestion candidates; scans are non-certifying and do not import evidence |
| Multiperson review ledger | Static review disposition plus validator | `validation/reviews/multiperson_review_2026_07_15.toml`, `src/bias_nma_adv/review_ledger.py`, and `tests/test_review_ledger.py` | Records review findings, actions, and next gates; it is process governance, not validation evidence or model certification |
| Unified validation status | Dynamic local gate summary | `src/bias_nma_adv/validation_status.py`, `scripts/write_validation_status.py`, and `tests/test_validation_status.py` | Composes current gates, including the proof-effect bundle, into JSON; does not certify models or enable clinical/HTA reporting |
| CI validation workflow | Dynamic GitHub Actions run | `.github/workflows/validation.yml` plus uploaded `validation-status` and `simulation-matrix-report` artifacts | Runs tests and machine-readable gates; a green workflow is execution evidence, not production certification |
| Source-identity snapshots | Dynamic public API check | `validation/source_checks/*source_check.json` | Verifies public-record identity and reachability, not numeric extraction |
| Event-count snapshots | Dynamic public API check | `validation/source_checks/*event_counts.json` | Verifies exact abstract count tokens and nearby treatment terms, not full paper extraction |
| Reported-HR snapshots | Dynamic public API check | `validation/source_checks/*reported_hr_tokens.json` | Verifies HR/CI abstract tokens near a hazard-ratio anchor, not survival model fitting |
| Statistical results | Dynamic | Future benchmark artefacts | No superiority claim is made here without generated benchmark outputs |

## References

- `netmeta`: https://cran.r-project.org/package=netmeta
- `multinma`: https://dmphillippo.github.io/multinma/
- `MBNMAdose`: https://cran.r-project.org/package=MBNMAdose
- `crossnma`: https://cran.r-project.org/package=crossnma
- CINeMA: https://cinema.med.auth.gr/
- ROB-MEN: https://cinema.ispm.unibe.ch/rob-men/
- Wigle A, Beliveau A, Nikolakopoulou A, Lin L. Creating treatment and component hierarchies in component network meta-analysis. arXiv:2605.15142. https://arxiv.org/abs/2605.15142
