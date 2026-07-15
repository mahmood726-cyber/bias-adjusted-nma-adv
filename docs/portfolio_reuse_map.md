<!-- sentinel:skip-file -->
# Portfolio Reuse Map

Checked: 2026-07-15

This note records which nearby repositories can materially help this project become a stronger evidence-synthesis platform under the source constraint: ClinicalTrials.gov records, PubMed abstracts, and open-access papers only.

It is a reuse plan, not a certification claim. Any imported method must be ported or wrapped with source-backed tests before it can support a platform capability.

Current gates:

- Candidate source repositories are recorded in `validation/portfolio_reuse_sources.toml` using repo names and relative asset paths only. Run `python scripts/scan_portfolio_reuse.py --root <portfolio-root>` to produce a non-certifying local scan of present repos, dirty worktrees, and missing assets.
- Source-backed benchmark artifacts imported or inspired by these repositories must be registered in `validation/benchmark_registry.toml` or covered by an equivalent machine-readable validation gate. The registry must pin artifact hashes, source manifests, source-check reports, limitations, and `certification_effect = "none"` until a separate external reference run passes. Source-check files must pass their specialized validators, not just hash matching.

The portfolio scan requires four review rounds before any import: source-boundary review, statistical-methods review, implementation-contract review, and claims-governance review.

## Immediate Reuse Candidates

| Source repo | Reusable asset | How it helps this project | Reuse boundary |
| --- | --- | --- | --- |
| `C:\Projects\wasserstein` | Kaplan-Meier curve extraction, Guyot-style IPD reconstruction, HR validation reports | Adds a survival real-meta validation path from open-access KM figures and abstracts | Do not treat existing outputs as certified. Re-validate every PMID, NCT ID, curve, risk table, and HR against OA source text or source figures before use. |
| `C:\Projects\advanced-nma-pooling` | Config-driven NMA workflows, optional `netmeta`/`multinma` adapters, HKSJ-floor and PI conventions | Gives the closest existing Python implementation pattern for tier-one reference matching and publication-gate outputs | Reuse conventions and adapter shape, but re-run all parity artifacts under this repo's source policy before claiming any match. |
| `C:\Projects\complex-evidence-synthesis-map` | Self-audit verdicts, honest prediction-interval selection, weighted-likelihood multiverse, POTH rank guard | Provides the operational wrapper strategy: estimator outputs plus audit, calibration, and ranking-safety checks | Reuse as governance and diagnostics, not as proof that any estimator is better. Weighted-likelihood is for multiverse aggregation, not for pooling clinical studies. |
| `C:\Projects\aact-kit` | Local AACT/ClinicalTrials.gov table resolution, schema checks, and aggregation helpers | Can become the CT.gov ingestion backbone for source-backed benchmark discovery | Verify actual columns and lowercase intervention types for every query; do not assume a local AACT backend exists in CI. |
| `C:\Projects\sheaf-nma` | Real `netmeta` corpus export, inconsistency-localization tests, DBT/Bucher/node-splitting comparators | Gives a real NMA inconsistency validation corpus and comparator harness design | Built-in `netmeta` datasets are useful reference fixtures, but they are not enough for the OA-only clinical source policy unless linked back to admissible source records. |
| `C:\Projects\spec-collapse-atlas` | Weighted-likelihood aggregation and multiverse calibration tests against `metafor` | Supplies a tested way to report method-choice fragility without anti-conservative IV pooling of specs | Must be used for specification sensitivity only; never present a multiverse aggregate as a new clinical estimator without validation. |
| `C:\Projects\topo-transport-ma` | Effect-modifier meta-regression, collapsibility guard, topological support certificate | Adds population-transportability diagnostics and fail-closed warnings for unsupported target populations | Port concepts and tests, not raw claims. The OR refusal and baseline-risk sign-flip warning should become platform guards. |
| `C:\Projects\allmeta\shared\nma-multiarm-v1.js` | Multi-arm contrast-level GLS NMA with netmeta parity fixture | Gives a small audited algorithmic template for preserving multi-arm covariance | Port to Python only after reproducing the JS fixture and an independent `netmeta` run. Comments in JS files have encoding damage, and the local worktree was dirty during the 2026-07-15 scan, so do not copy prose or unreviewed values. |
| `C:\Projects\allmeta\shared\transported-nma-v1.js` | Entropy-balancing transported NMA approximation | Gives a population-specific NMA candidate and an ESS-loss diagnostic | Treat as experimental until compared against IPD or ML-NMR reference targets. The local allmeta worktree must be reviewed before any import. |
| `C:\Projects\allmeta\nma-pro-v2` | Netmeta parity fixtures and R scripts | Provides lightweight frequentist NMA parity examples | Fixture values can seed tests only if the source R script and package version are captured in the artifact. |
| `C:\Users\mahmo\code\rct-extractor-v2` | PubMed/OA/CT.gov extraction ensemble, outcome matching, value plausibility checks | Provides ingestion and adjudication patterns for real meta-analysis datasets | Do not trust rows wholesale. One inspected benchmark row points PMID 32865377 to a non-article fallback PDF, so every row needs source validation before use. |

## What To Build From These Repos

1. **Survival KM validation lane**
   - Use `wasserstein` as the starting point for OA paper KM extraction.
   - Required artifacts: PMID, NCT ID when present, OA URL or PMC ID, source figure/table location, extracted curve CSV, reconstructed IPD hash, reference HR source, reproduced HR, and discrepancy.
   - Current guard: this repository now has an OA-only KM reconstruction screen that blocks text-only HR and synthetic-IPD fallbacks before they can enter validation artifacts.
   - Certification target: match known HRs and confidence intervals across a prespecified set of OA survival RCTs.

2. **Transportability support lane**
   - Port the `topo-transport-ma` support certificate into a platform diagnostic module.
   - Required guards: refuse OR/logOR transport, warn on ratio-scale baseline-risk heterogeneity, separate point-estimate fitting from support grading.
   - Certification target: match metafor/statsmodels anchors and confirm topological support behavior on known fixtures before using real clinical covariate clouds.

3. **Frequentist NMA parity lane**
   - The multi-arm GLS fixture has now been ported into Python tests and a governed local replay artifact.
   - Current checks: full multi-arm covariance block, disconnected-network failure, incomplete multi-arm clique warning, FE and RE deterministic replay values.
   - Remaining certification target: run the external `netmeta` adapter and then pass multiple `netmeta` parity fixtures, not only a small hand-built network.

4. **Source-backed ingestion lane**
   - Reuse `rct-extractor-v2` design patterns for extraction ensembles and adjudication, but rebuild the validation gate here.
   - Required checks: typed PMID/NCT/DOI validation, OA status, source-snippet or figure provenance, no fallback PDF accepted as evidence unless it matches the PMID/article.
   - Certification target: turn source records into arm-level or contrast-level analysis rows with proof-carrying identifiers.

5. **Population-specific NMA lane**
   - Use `allmeta` transported NMA as an experimental baseline, then compare against ML-NMR and explicit IPD/aggregate integration when data permit.
   - Required output: source network result, transported target result, achieved target moments, ESS loss, and extrapolation warning.
   - Certification target: simulation plus real-data examples where target covariates are source-backed.

6. **Tier-one adapter lane**
   - Reuse `advanced-nma-pooling` adapter shape for `netmeta`, `multinma`, and later `MBNMAdose` / `crossnma`.
   - Required output: command, package version, input hash, reference output hash, numerical tolerance, skip-with-reason when R/package is unavailable.
   - Certification target: each module reaches Reference Matched only from immutable parity artifacts, not from local smoke tests.

7. **Method-choice robustness lane**
   - Reuse `complex-evidence-synthesis-map` and `spec-collapse-atlas` for self-audit, POTH, and multiverse reporting.
   - Required output: primary estimator, prespecified sensitivity grid, weighted-likelihood multiverse summary, and clear separation between clinical effect estimates and specification-fragility summaries.
   - Certification target: no "best treatment" or "robust" claim unless interval calibration, rank uncertainty, and self-audit gates pass.

## Hard Stops

- Existing portfolio outputs are not evidence unless their identifiers, dates, article links, and numerical values are rechecked.
- No benchmark row is accepted if it depends on closed full text, a non-article fallback PDF, a guessed outcome, or a hardcoded effect size.
- Simulations can test operating characteristics, but they cannot be presented as real clinical validation.
- Native methods are not called better than `netmeta`, `multinma`, `MBNMAdose`, `crossnma`, `metafor`, or `meta` until reference matching and broader validation artifacts exist.
- Portfolio methods are not imported as black boxes; every reused component needs a local contract, source policy check, and failure mode test.

## Static-Vs-Dynamic Hardcode Disclosure

| Item | Static or dynamic | Evidence source | Disclosure |
| --- | --- | --- | --- |
| Local repo paths | Static local inspection | Disk scan on 2026-07-15 | Paths are documented for development only and must not be hardcoded into shipped code. |
| Reusable method descriptions | Static review | README, source, and test files inspected locally | Descriptions guide porting work; they do not certify correctness. |
| Existing validation numbers in source repos | Dynamic and untrusted until reproduced | Local JSON/test files | Must be regenerated or independently checked before becoming platform evidence. |
| Trial identifiers and effect estimates | Dynamic source-backed fields | ClinicalTrials.gov, PubMed abstracts, and OA papers | Must be validated per record before use in code, tests, dashboards, or claims. |
| New validation targets | Static registry entries | `validation/reference_targets.toml` | Targets remain `planned` until machine-verifiable artifacts exist. |
| Portfolio reuse candidates | Static registry plus optional dynamic scan | `validation/portfolio_reuse_sources.toml` and `scripts/scan_portfolio_reuse.py` | Candidate repos are implementation inputs only; scan reports do not certify clinical evidence or method superiority. |

## Next Implementation Order

1. Keep the source-backed benchmark registry green as new real-meta, survival, or network artifacts are added.
2. Run and hash the external `netmeta` multi-arm adapter when R and `netmeta` are available; keep the current preflight as a skip until then.
3. Add the first real OA Kaplan-Meier source manifest and curve/IPD hashes that pass the new KM reconstruction screen.
4. Port the transportability collapsibility guards and support certificate behind an experimental API.
5. Add an ingestion provenance validator that rejects mismatched PMID/OA PDF rows.
6. Only then expand the real-meta benchmark set beyond the current SGLT2 heart-failure fixture.
