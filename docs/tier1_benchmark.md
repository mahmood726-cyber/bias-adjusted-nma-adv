<!-- sentinel:skip-file -->
# Tier-One Methods Benchmark

Checked: 2026-07-15

This note benchmarks the platform specification against mature specialist network meta-analysis methods. It is not evidence that the current Python implementation is superior to those methods.

## Bottom Line

The platform can plausibly become stronger than any single specialist package as an integrated evidence-synthesis environment. It should not claim that every native estimator beats the best specialist method in its domain until reference matching, simulation validation, and independent reproduction have been completed.

The correct claim is:

> Use tier-one methods as foundations and validation targets, then compete through integration, governance, diagnostics, reproducibility, and carefully validated methodological improvements.

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

The executable reference-target registry is `validation/reference_targets.toml`. It is intentionally conservative: every target starts as `planned`, and certification tests fail closed if a module is promoted without evidence artifacts. Real-world validation data must come from ClinicalTrials.gov records, PubMed abstracts, or open-access papers only.

External reference adapters must also emit machine-readable run reports under `validation/reference_runs/`. A skipped or unavailable adapter is recorded as `certification_effect = "none"` and cannot count as reference-matching evidence. A report becomes a certification candidate only when it has `status = "passed"`, package versions, input and output artifact hashes, and a prespecified tolerance.

Real-data benchmarks must also be paired with source snapshots under `validation/source_checks/`. Source-identity snapshots verify that public ClinicalTrials.gov and PubMed records are reachable and match the manifest identifiers. Event-count and reported-HR snapshots, when present, separately verify exact numeric tokens and nearby treatment terms in PubMed abstracts; they are still not a substitute for full open-access paper/table extraction, Kaplan-Meier digitization, or external reference-software parity. Source-check artifacts must carry `certification_effect = "none"`.

Current adapter preflights: `validation/reference_runs/pairwise_metafor_meta_preflight.toml` records that the planned `metafor`/`meta` pairwise adapter could not run because `Rscript` was not available on PATH in this environment. `validation/reference_runs/multiarm_netmeta_preflight.toml` records the same fail-closed status for the planned `netmeta` multi-arm adapter. These are honest skips, not parity claims.

Current local multi-arm artifact: `validation/multiarm/netmeta_portfolio_multiarm_benchmark.toml` deterministically replays the governed arm-level fixture in `validation/multiarm/netmeta_portfolio_multiarm_arms.csv`. It is useful for regression testing multi-arm covariance handling, but it carries `certification_effect = "none"` until an external `netmeta` adapter actually runs and passes with package versions, output hashes, and a prespecified tolerance.

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
| Reference-run preflight | Dynamic local environment check | `validation/reference_runs/*.toml` | Unavailable or failed adapters are recorded but cannot certify a module |
| Multi-arm fixture replay | Static algorithmic fixture plus dynamic local recomputation | `validation/multiarm/*` and `tests/test_multiarm_artifact.py` | Validates local covariance handling on a small fixture; not clinical evidence and not external `netmeta` parity |
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
