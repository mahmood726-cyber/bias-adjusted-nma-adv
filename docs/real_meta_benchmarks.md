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

## Portfolio Reuse Recon

Portfolio recon was run with `find-related-repos.py "evidence synthesis benchmark meta-analysis wasserstein"`, followed by targeted inspection of the local `wasserstein` repository. The selected reusable patterns were:

- `ma-workbench`: reused the honest benchmark pattern of source-backed fixtures, explicit pass/fail artifacts, and no threshold relaxation after a miss.
- `wasserstein`: reused the method-governance pattern for survival/KM work: provenance hashes, CI containment checks, and outputs marked uncertified unless independently validated. No Wasserstein extraction outputs are imported as evidence here.
- `fragility-atlas`: reserved as a future pattern for multiverse/sensitivity stress testing. It is not used as a source for the current PCSK9 or SGLT2 effects.

Net-new work in this repository is the NMA-oriented source contract: PubMed abstract HR-token verification, CT.gov/PubMed identity checks, source-bound survival HR manifests, and benchmark artifacts that preserve `certification_effect = "none"` until external reference matching exists.

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

## Benchmark 2: PCSK9 Inhibitors And Major Cardiovascular Events

Reported survival HR manifest: `validation/survival/pcsk9_mace_reported_hrs.toml`

Reported survival HR source-identity snapshot: `validation/source_checks/pcsk9_mace_reported_hr_source_check.json`

Reported survival HR token snapshot: `validation/source_checks/pcsk9_mace_reported_hr_tokens.json`

Reported survival HR benchmark: `validation/survival/pcsk9_mace_reported_hr_benchmark.toml`

Outcome: trial-defined major adverse cardiovascular event composite in FOURIER and ODYSSEY Outcomes.

Scale currently tested: log hazard ratio, derived from reported PubMed abstract HR and 95% CI tokens.

| Trial | NCT | PMID | Active treatment | Control | Reported HR | 95% CI |
| --- | --- | --- | --- | --- | ---: | --- |
| FOURIER | NCT01764633 | 28304224 | evolocumab | placebo | 0.85 | 0.79 to 0.92 |
| ODYSSEY-Outcomes | NCT01663402 | 30403574 | alirocumab | placebo | 0.85 | 0.78 to 0.93 |

Current tests:

- validate both records against the allowed source boundary;
- validate CT.gov and PubMed source identity snapshots before using the HR tokens;
- validate PubMed abstract HR/CI tokens near the hazard-ratio anchor and active/control treatment terms;
- derive log-HR study effects from the verified reported HR and 95% CI tokens;
- recompute the generated benchmark artifact from the manifest, token snapshot, and identity snapshot;
- require `certification_effect = "none"` because source verification does not prove model performance or tier-one parity.

Reported HR benchmark on the log-HR scale:

| Engine | Estimate | SE | 95% interval |
| --- | ---: | ---: | ---: |
| Experimental pairwise fixed effect | -0.162519 | 0.029377 | -0.220096 to -0.104942 |
| Experimental pairwise REML-HKSJ | -0.162519 | 0.029377 | -0.535783 to 0.210745 |

Limitations:

- this is a two-study pairwise class benchmark, not a full multi-treatment survival NMA;
- the public PubMed abstracts provide the reported HR/CI tokens used here, but Kaplan-Meier curves are not digitized;
- the two reported HR point estimates are identical, so this fixture tests provenance and artifact reproducibility more than heterogeneity behavior;
- with only two studies, the HKSJ interval is intentionally conservative and should not be treated as a superiority result;
- this benchmark does not yet have a passed external `metafor`, `meta`, `netmeta`, `multinma`, `MBNMAdose`, or `crossnma` reference run.

## Static-Vs-Dynamic Hardcode Disclosure

| Item | Static or dynamic | Evidence source | Disclosure |
| --- | --- | --- | --- |
| Portfolio reuse | Static documentation | `ma-workbench`, `wasserstein`, and `fragility-atlas` repository inspection | Reuses process patterns only; prior repo outputs are not treated as validation data for this platform |
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
