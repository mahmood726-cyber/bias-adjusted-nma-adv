<!-- sentinel:skip-file -->
# Portfolio Reuse Map

Checked: 2026-07-15

This note records which nearby repositories can materially help this project become a stronger evidence-synthesis platform under the source constraint: ClinicalTrials.gov records, PubMed abstracts, and open-access papers only.

It is a reuse plan, not a certification claim. Any imported method must be ported or wrapped with source-backed tests before it can support a platform capability.

## Immediate Reuse Candidates

| Source repo | Reusable asset | How it helps this project | Reuse boundary |
| --- | --- | --- | --- |
| `C:\Projects\wasserstein` | Kaplan-Meier curve extraction, Guyot-style IPD reconstruction, HR validation reports | Adds a survival real-meta validation path from open-access KM figures and abstracts | Do not treat existing outputs as certified. Re-validate every PMID, NCT ID, curve, risk table, and HR against OA source text or source figures before use. |
| `C:\Projects\topo-transport-ma` | Effect-modifier meta-regression, collapsibility guard, topological support certificate | Adds population-transportability diagnostics and fail-closed warnings for unsupported target populations | Port concepts and tests, not raw claims. The OR refusal and baseline-risk sign-flip warning should become platform guards. |
| `C:\Projects\allmeta\shared\nma-multiarm-v1.js` | Multi-arm contrast-level GLS NMA with netmeta parity fixture | Gives a small audited algorithmic template for preserving multi-arm covariance | Port to Python only after reproducing the JS fixture and an independent `netmeta` run. Comments in JS files have encoding damage, so do not copy prose. |
| `C:\Projects\allmeta\shared\transported-nma-v1.js` | Entropy-balancing transported NMA approximation | Gives a population-specific NMA candidate and an ESS-loss diagnostic | Treat as experimental until compared against IPD or ML-NMR reference targets. |
| `C:\Projects\allmeta\nma-pro-v2` | Netmeta parity fixtures and R scripts | Provides lightweight frequentist NMA parity examples | Fixture values can seed tests only if the source R script and package version are captured in the artifact. |
| `C:\Users\mahmo\code\rct-extractor-v2` | PubMed/OA/CT.gov extraction ensemble, outcome matching, value plausibility checks | Provides ingestion and adjudication patterns for real meta-analysis datasets | Do not trust rows wholesale. One inspected benchmark row points PMID 32865377 to a non-article fallback PDF, so every row needs source validation before use. |

## What To Build From These Repos

1. **Survival KM validation lane**
   - Use `wasserstein` as the starting point for OA paper KM extraction.
   - Required artifacts: PMID, NCT ID when present, OA URL or PMC ID, source figure/table location, extracted curve CSV, reconstructed IPD hash, reference HR source, reproduced HR, and discrepancy.
   - Certification target: match known HRs and confidence intervals across a prespecified set of OA survival RCTs.

2. **Transportability support lane**
   - Port the `topo-transport-ma` support certificate into a platform diagnostic module.
   - Required guards: refuse OR/logOR transport, warn on ratio-scale baseline-risk heterogeneity, separate point-estimate fitting from support grading.
   - Certification target: match metafor/statsmodels anchors and confirm topological support behavior on known fixtures before using real clinical covariate clouds.

3. **Frequentist NMA parity lane**
   - Port the multi-arm GLS fixture into Python tests for `src/bias_nma_adv`.
   - Required checks: full multi-arm covariance block, disconnected-network failure, incomplete multi-arm clique warning, FE and RE estimates against `netmeta`.
   - Certification target: pass multiple `netmeta` parity fixtures, not only a small hand-built network.

4. **Source-backed ingestion lane**
   - Reuse `rct-extractor-v2` design patterns for extraction ensembles and adjudication, but rebuild the validation gate here.
   - Required checks: typed PMID/NCT/DOI validation, OA status, source-snippet or figure provenance, no fallback PDF accepted as evidence unless it matches the PMID/article.
   - Certification target: turn source records into arm-level or contrast-level analysis rows with proof-carrying identifiers.

5. **Population-specific NMA lane**
   - Use `allmeta` transported NMA as an experimental baseline, then compare against ML-NMR and explicit IPD/aggregate integration when data permit.
   - Required output: source network result, transported target result, achieved target moments, ESS loss, and extrapolation warning.
   - Certification target: simulation plus real-data examples where target covariates are source-backed.

## Hard Stops

- Existing portfolio outputs are not evidence unless their identifiers, dates, article links, and numerical values are rechecked.
- No benchmark row is accepted if it depends on closed full text, a non-article fallback PDF, a guessed outcome, or a hardcoded effect size.
- Simulations can test operating characteristics, but they cannot be presented as real clinical validation.
- Native methods are not called better than `netmeta`, `multinma`, `MBNMAdose`, `crossnma`, `metafor`, or `meta` until reference matching and broader validation artifacts exist.

## Static-Vs-Dynamic Hardcode Disclosure

| Item | Static or dynamic | Evidence source | Disclosure |
| --- | --- | --- | --- |
| Local repo paths | Static local inspection | Disk scan on 2026-07-15 | Paths are documented for development only and must not be hardcoded into shipped code. |
| Reusable method descriptions | Static review | README, source, and test files inspected locally | Descriptions guide porting work; they do not certify correctness. |
| Existing validation numbers in source repos | Dynamic and untrusted until reproduced | Local JSON/test files | Must be regenerated or independently checked before becoming platform evidence. |
| Trial identifiers and effect estimates | Dynamic source-backed fields | ClinicalTrials.gov, PubMed abstracts, and OA papers | Must be validated per record before use in code, tests, dashboards, or claims. |
| New validation targets | Static registry entries | `validation/reference_targets.toml` | Targets remain `planned` until machine-verifiable artifacts exist. |

## Next Implementation Order

1. Add Python multi-arm GLS parity tests using the `allmeta` fixture and an explicit `netmeta` artifact.
2. Add a survival KM benchmark manifest schema before importing any `wasserstein` output.
3. Port the transportability collapsibility guards and support certificate behind an experimental API.
4. Add an ingestion provenance validator that rejects mismatched PMID/OA PDF rows.
5. Only then expand the real-meta benchmark set beyond the current SGLT2 heart-failure fixture.
