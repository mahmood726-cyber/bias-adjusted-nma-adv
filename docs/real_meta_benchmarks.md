<!-- sentinel:skip-file -->
# Real Meta-Analysis Benchmarks

Checked: 2026-07-15

This document records source-backed real-data benchmarks. These benchmarks are validation evidence only when the tests and generated artifacts pass; they are not superiority claims.

## Source Boundary

Real-world validation inputs are restricted to:

- ClinicalTrials.gov public records;
- PubMed abstracts;
- open-access papers.

Closed IPD, proprietary trial reports, secondary package fixtures, and unverified inline demo numbers are not admissible validation sources.

## Benchmark 1: SGLT2 Inhibitors In Heart Failure

Dataset: `validation/real_meta/sglt2_hf_primary_events.csv`

Source manifest: `validation/real_meta/sglt2_hf_primary_sources.toml`

Source identity snapshot: `validation/source_checks/sglt2_hf_primary_source_check.json`

PubMed abstract event-count snapshot: `validation/source_checks/sglt2_hf_primary_event_counts.json`

Reported survival HR manifest: `validation/survival/sglt2_hf_reported_hrs.toml`

Reported survival HR source-identity snapshot: `validation/source_checks/sglt2_hf_reported_hr_source_check.json`

Reported survival HR token snapshot: `validation/source_checks/sglt2_hf_reported_hr_tokens.json`

Reported survival HR benchmark: `validation/survival/sglt2_hf_reported_hr_benchmark.toml`

Outcome: primary composite outcome in each trial, harmonized as worsening heart failure or cardiovascular death / cardiovascular death or heart-failure hospitalization.

Scale currently tested: log odds ratio, because the current arm-level binary model estimates log-odds contrasts.

| Trial | NCT | PMID | Active events / n | Placebo events / n |
| --- | --- | --- | --- | --- |
| DAPA-HF | NCT03036124 | 31535829 | 386 / 2373 | 502 / 2371 |
| EMPEROR-Reduced | NCT03057977 | 32865377 | 361 / 1863 | 462 / 1867 |
| DELIVER | NCT03619213 | 36027570 | 512 / 3131 | 610 / 3132 |
| EMPEROR-Preserved | NCT03057951 | 34449189 | 415 / 2997 | 511 / 2991 |

Current tests:

- validate all rows against the allowed source boundary;
- validate a separate source manifest requiring matching trial labels, NCT IDs, PMIDs, outcomes, arm counts, and admissible PubMed/ClinicalTrials.gov URLs;
- validate a source-identity snapshot showing the CT.gov and PubMed public records were reachable and matched the manifest identifiers at verification time;
- validate a PubMed abstract event-count snapshot showing the exact arm-level count tokens and nearby active/control treatment terms were present in the public abstracts at verification time;
- validate a reported survival HR snapshot showing the abstract HR and confidence-interval tokens are present near the hazard-ratio anchor and treatment terms;
- derive log-HR study effects from the verified reported HR and 95% CI tokens;
- run the experimental pairwise bridge as fixed effect and REML-HKSJ on the reported log-HR scale;
- compute an independent inverse-variance fixed-effect log-odds-ratio reference;
- compute source-backed study-level log-odds-ratio effects with NCT/PMID provenance;
- run the experimental pairwise bridge as fixed effect and REML-HKSJ with prediction interval;
- run the frequentist candidate model on the same rows;
- run the Bayesian MCMC candidate model on the same rows;
- require the pairwise fixed-effect result to match the independent fixed-effect reference exactly within numerical tolerance;
- require the frequentist result to match the independent fixed-effect reference exactly within numerical tolerance;
- require the Bayesian posterior mean to be directionally and numerically compatible with the same reference.

Current local benchmark contract: `validation/real_meta/sglt2_hf_primary_benchmark.toml`.

External reference preflight: `validation/reference_runs/pairwise_metafor_meta_preflight.toml`.

The external `metafor`/`meta` parity adapter is planned at `external/r/pairwise_metafor_meta.R`, but the current preflight is `unavailable` because `Rscript` is not available on PATH in this environment. This benchmark therefore remains a local source-backed validation artifact, not a reference-matched artifact.

| Engine | Estimate | SE / posterior SD | 95% interval |
| --- | ---: | ---: | ---: |
| Independent fixed-effect logOR | -0.268984 | 0.036297 | -0.340125 to -0.197843 |
| Experimental pairwise fixed effect | -0.268984 | 0.036297 | -0.340125 to -0.197843 |
| Experimental pairwise REML-HKSJ | -0.268984 | 0.036297 | -0.384497 to -0.153470 |
| Candidate frequentist model | -0.268984 | 0.036297 | -0.340125 to -0.197843 |
| Candidate Bayesian MCMC | -0.260798 | 0.049688 | -0.357244 to -0.164967 |

Reported HR benchmark on the log-HR scale:

| Engine | Estimate | SE | 95% interval |
| --- | ---: | ---: | ---: |
| Experimental pairwise fixed effect | -0.250426 | 0.033071 | -0.315244 to -0.185609 |
| Experimental pairwise REML-HKSJ | -0.250426 | 0.033071 | -0.355672 to -0.145181 |

Limitations:

- this is a pairwise class meta-analysis, not a full multi-treatment NMA;
- the arm-level model benchmark uses first-event binary counts; the separate reported-HR benchmark uses published HR tokens but does not fit a survival NMA;
- the reported HR snapshot verifies source tokens only; it does not yet fit a survival NMA or reconstruct IPD from Kaplan-Meier figures;
- the REML heterogeneity estimate is zero on this four-study fixture, so this is not evidence of random-effects superiority;
- PubMed/CT.gov source identity is verified by public API snapshots, and the arm counts are checked against exact PubMed abstract count tokens; full paper/table extraction, time-to-event HR extraction, and independent dual extraction remain future work;
- this benchmark does not yet have a passed external `metafor`, `meta`, `netmeta`, `multinma`, `MBNMAdose`, or `crossnma` reference run.

## Static-Vs-Dynamic Hardcode Disclosure

| Item | Static or dynamic | Evidence source | Disclosure |
| --- | --- | --- | --- |
| Trial arm counts | Static fixture | `sglt2_hf_primary_events.csv` plus `sglt2_hf_primary_sources.toml` | Treated as extracted source-backed data, not simulated output |
| Source manifest | Static fixture | PubMed abstract URLs and ClinicalTrials.gov record URLs | Machine-checked for identifier, outcome, source-type, and arm-count consistency |
| Source identity snapshot | Dynamic public API check | `scripts/verify_real_meta_sources.py` against ClinicalTrials.gov API and PubMed EFetch | Verifies identity and reachability only, not event-count extraction |
| Event-count snapshot | Dynamic public API check | `scripts/verify_pubmed_event_counts.py` against PubMed EFetch abstracts | Verifies exact `events of n` tokens and nearby active/control terms in abstracts, not full paper extraction |
| Reported HR snapshot | Dynamic public API check | `scripts/verify_pubmed_survival_hrs.py` against PubMed EFetch abstracts | Verifies HR/CI tokens near the hazard-ratio anchor and treatment terms, not KM digitization |
| Reported HR source identity | Dynamic public API check | `scripts/verify_survival_sources.py` and `validation/source_checks/sglt2_hf_reported_hr_source_check.json` | Verifies CT.gov NCT IDs and PubMed PMIDs before reported HR tokens are benchmarked |
| Reported HR benchmark | Dynamic computation | `scripts/write_survival_hr_benchmark.py` plus `bias_nma_adv.survival_benchmark` | Recomputes log-HR study effects and pairwise pooling from verified identity and reported-HR source snapshots |
| Source-check certification effect | Static contract | `certification_effect = "none"` in source-check artifacts | Source-token verification cannot certify model performance or tier-one parity |
| Independent fixed-effect reference | Dynamic computation | `bias_nma_adv.real_meta.fixed_effect_log_or_reference` | Recomputed by tests from the CSV rows |
| Pairwise bridge result | Dynamic computation | `bias_nma_adv.pairwise.fit_pairwise_meta` | Recomputed by tests from source-backed study-level effects |
| External pairwise reference run | Dynamic local environment preflight | `validation/reference_runs/pairwise_metafor_meta_preflight.toml` | Recorded as unavailable and has `certification_effect = "none"` |
| Candidate frequentist result | Dynamic computation | `AdvancedBiasAdjustedNMAPooler` | Recomputed by tests from the CSV rows |
| Candidate Bayesian result | Dynamic seeded computation | `BayesianNMAMCMCSampler` with seed 20260715 | Recomputed by tests with tolerance for sampler behavior |
| Certification status | Static contract | `sglt2_hf_primary_benchmark.toml` | `certification_effect = "none"` until external reference matching exists |
