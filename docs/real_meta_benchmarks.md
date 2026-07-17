<!-- sentinel:skip-file -->
# Real Meta-Analysis Benchmarks

Checked: 2026-07-17

This document records source-backed real-data benchmarks. These benchmarks are validation evidence only when the tests and generated artifacts pass; they are not superiority claims.

## Source Boundary

Real-world validation inputs are restricted to:

- ClinicalTrials.gov public records;
- PubMed abstracts;
- open-access papers;
- public FDA/EMA regulatory review rows when numeric per-trial result text is source-bound.

Protocol-only ICTRP, PACTR, and other trial registries are allowed for metadata such as registration, planned outcomes, eligibility criteria, dates, registered-primary-outcome anchors, and completeness denominators. Downloaded ICTRP or PACTR result rows are admissible only when public numeric result text is source-bound; protocol-only rows remain barred from model-ready effects.

Closed IPD, proprietary trial reports, secondary package fixtures, and unverified inline demo numbers are not admissible validation sources.

## Portfolio Reuse Recon

Portfolio recon was run with `find-related-repos.py "evidence synthesis benchmark meta-analysis wasserstein"`, followed by targeted inspection of the local `wasserstein` repository. The selected reusable patterns were:

- `ma-workbench`: reused the honest benchmark pattern of source-backed fixtures, explicit pass/fail artifacts, and no threshold relaxation after a miss.
- `wasserstein`: reused the method-governance pattern for survival/KM work: provenance hashes, CI containment checks, and outputs marked uncertified unless independently validated. No Wasserstein extraction outputs are imported as evidence here.
- `fragility-atlas`: reserved as a future pattern for multiverse/sensitivity stress testing. It is not used as a source for the current PCSK9 or SGLT2 effects.

Net-new work in this repository is the NMA-oriented source contract: PubMed abstract HR-token verification, CT.gov/PubMed identity checks, source-bound survival HR manifests, and benchmark artifacts that preserve `certification_effect = "none"` until external reference matching exists.

The 2026-07-15 Wasserstein inspection found extracted-summary patterns such as `text_hr_pair_fallback` with warnings that the curve-derived HR diverged and the pipeline used the text HR. The KM reconstruction policy now blocks those fallback methods and warning terms before any OA KM artifact can enter validation.

The generated coverage atlas `validation/real_benchmark_atlas.json` summarizes the current registered real-data benchmark surface: 10 benchmark artifacts, 76 study-effect rows, 23 unique NCT IDs, and 15 unique PMIDs. It is a coverage and governance artifact only; it does not certify tier-one parity, clinical superiority, KM reconstruction accuracy, dose-response NMA parity, cross-design parity, component-NMA parity, broad closed-loop inconsistency performance, or production use.

## Benchmark 1: SGLT2 Inhibitors In Heart Failure

Dataset: `validation/real_meta/sglt2_hf_primary_events.csv`

Source manifest: `validation/real_meta/sglt2_hf_primary_sources.toml`

Source identity snapshot: `validation/source_checks/sglt2_hf_primary_source_check.json`

PubMed abstract event-count snapshot: `validation/source_checks/sglt2_hf_primary_event_counts.json`

Reported survival HR manifest: `validation/survival/sglt2_hf_reported_hrs.toml`

Reported survival HR source-identity snapshot: `validation/source_checks/sglt2_hf_reported_hr_source_check.json`

Reported survival HR token snapshot: `validation/source_checks/sglt2_hf_reported_hr_tokens.json`

Reported survival HR benchmark: `validation/survival/sglt2_hf_reported_hr_benchmark.toml`

Proof-carrying reported-HR extraction bundle: `validation/ingestion/sglt2_hf_reported_hr_proof_effects.json`

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
- validate a proof-carrying reported-HR extraction bundle with source-manifest/report hashes, PubMed abstract source identifiers, minimal HR/CI snippets, and complete CI uncertainty;
- derive log-HR study effects from the verified reported HR and 95% CI tokens;
- run the experimental pairwise bridge as fixed effect and REML-HKSJ on the reported log-HR scale;
- compute an independent inverse-variance fixed-effect log-odds-ratio reference;
- compute source-backed study-level log-odds-ratio effects with NCT/PMID provenance;
- run the experimental pairwise bridge as fixed effect and REML-HKSJ with prediction interval;
- emit a pairwise FE/DL/PM/REML tau2 cross-check summary, including sign and null-crossing diagnostics;
- run the frequentist candidate model on the same rows;
- record the frequentist candidate's declared `tau_method` and dropped-study count;
- run the Bayesian MCMC candidate model on the same rows;
- require the pairwise fixed-effect result to match the independent fixed-effect reference exactly within numerical tolerance;
- require the frequentist result to match the independent fixed-effect reference exactly within numerical tolerance;
- require the Bayesian posterior mean to be directionally and numerically compatible with the same reference.

Current local benchmark contract: `validation/real_meta/sglt2_hf_primary_benchmark.toml`.

External reference preflight: `validation/reference_runs/pairwise_metafor_meta_preflight.toml`.

Validated local R output: `validation/reference_runs/pairwise_metafor_meta_reference.toml` validates `validation/reference_runs/pairwise_metafor_meta_output.json` against this source-backed benchmark. It matches study-level log-OR effects and the fixed-effect pooled result within the stated tolerance, and it records the expected HKSJ difference: `metafor` uses the unfloored KNHA factor here, while this Python artifact applies the prespecified HKSJ floor. This is a narrow evidence-candidate parity artifact, not broad production certification.

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
- the proof-carrying extraction bundle validates model-ingestion provenance only; it does not certify source extraction accuracy beyond the checked abstract HR/CI tokens and does not certify any estimator;
- the REML heterogeneity estimate is zero on this four-study fixture, so this is not evidence of random-effects superiority;
- PubMed/CT.gov source identity is verified by public API snapshots, and the arm counts are checked against exact PubMed abstract count tokens; full paper/table extraction, time-to-event HR extraction, and independent dual extraction remain future work;
- this benchmark now has a narrow local `metafor`/`meta` output-validation report for fixed-effect and documented HKSJ-convention checks, but it does not yet satisfy all planned pairwise, `netmeta`, `multinma`, `MBNMAdose`, or `crossnma` reference targets.

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

## Benchmark 3: SGLT2 Inhibitors And Chronic Kidney Disease Progression

Reported survival HR manifest: `validation/survival/sglt2_ckd_reported_hrs.toml`

Reported survival HR source-identity snapshot: `validation/source_checks/sglt2_ckd_reported_hr_source_check.json`

Reported survival HR token snapshot: `validation/source_checks/sglt2_ckd_reported_hr_tokens.json`

Reported survival HR benchmark: `validation/survival/sglt2_ckd_reported_hr_benchmark.toml`

External `metafor` reference output: `validation/reference_runs/sglt2_ckd_survival_hr_metafor_reference.toml` validates `validation/reference_runs/sglt2_ckd_survival_hr_metafor_output.json`.

Trials: CREDENCE (`NCT02065791`, PMID 30990260), DAPA-CKD (`NCT03036150`, PMID 32970396), and EMPA-KIDNEY (`NCT03594110`, PMID 36331190).

Outcome: trial-defined kidney disease progression composite or renal/cardiovascular death composite, as reported in the PubMed abstract.

Scale currently tested: log hazard ratio, derived from reported PubMed abstract HR and 95% CI tokens.

The REML-HKSJ artifact estimates `tau2 = 0.00107019168474418`, so this benchmark satisfies the positive-heterogeneity coverage gate. This is a heterogeneity-stress signal only, not evidence of survival NMA parity or clinical superiority.

Limitations:

- this is a three-study pairwise class benchmark, not a multi-treatment survival NMA;
- the public PubMed abstracts provide the reported HR/CI tokens used here, but Kaplan-Meier curves are not digitized;
- the outcome definitions are related but not identical across trials;
- the `metafor` reference report validates fixed-effect reported-HR pooling only; it does not certify REML superiority, KM reconstruction, or clinical/HTA use.

## Benchmark 4: Type 2 Diabetes MACE-Class Star Network

CT.gov reported-HR network manifest: `validation/networks/t2d_mace_ctgov_hrs.toml`

CT.gov reported-HR source snapshot: `validation/source_checks/t2d_mace_ctgov_hr_network_check.json`

CT.gov reported-HR network benchmark: `validation/networks/t2d_mace_ctgov_hr_network_benchmark.toml`

Outcome: trial-defined major adverse cardiovascular event or MACE-like cardiovascular composite. TECOS is explicitly recorded as the CT.gov MACE Plus per-protocol analysis because that is the public CT.gov record matching the verified HR and CI.

Scale currently tested: log hazard ratio, derived from ClinicalTrials.gov reported HR and 95% CI records.

Network: placebo-centered star network with three analyst-defined active class labels.

| Class | Trials |
| --- | --- |
| DPP-4 inhibitor | TECOS, CARMELINA |
| GLP-1 RA | EXSCEL, REWIND, HARMONY Outcomes, PIONEER-6 |
| SGLT2 inhibitor | CANVAS, EMPA-REG OUTCOME, VERTIS-CV, CREDENCE |

Current tests:

- validate the manifest schema, source boundary, treatment labels, NCT IDs, HR/CI containment, and no certification effect;
- verify every HR/CI against live ClinicalTrials.gov API results snapshots using NCT identity, completed status, outcome-title terms, exact HR/CI fields, and drug/placebo source terms;
- recompute study-level log-HR effects from the verified CT.gov HR/CI values;
- fit a fixed-effect contrast-GLS NMA and a generalized-DL random-effects contrast-GLS NMA;
- recompute the generated benchmark artifact from the manifest and source snapshot;
- require `certification_effect = "none"` because source verification does not prove model performance or tier-one parity.

Current local benchmark results:

| Model | Treatment class vs placebo | HR | 95% interval |
| --- | --- | ---: | ---: |
| Fixed GLS | DPP-4 inhibitor | 0.995 | 0.915 to 1.082 |
| Fixed GLS | GLP-1 RA | 0.869 | 0.822 to 0.918 |
| Fixed GLS | SGLT2 inhibitor | 0.895 | 0.827 to 0.967 |
| Random GLS | DPP-4 inhibitor | 0.995 | 0.915 to 1.082 |
| Random GLS | GLP-1 RA | 0.869 | 0.822 to 0.918 |
| Random GLS | SGLT2 inhibitor | 0.895 | 0.827 to 0.967 |

The random-effects artifact currently estimates `tau2 = 0.0` with `Q = 6.8785` and `df = 7`, so random and fixed GLS estimates coincide in this fixture.

Limitations:

- this is a placebo-centered star network, so closed-loop inconsistency and node-splitting cannot be assessed;
- class labels are analyst-defined groupings and do not prove class exchangeability or clinical superiority;
- composite outcome definitions are similar but not identical across trials;
- CT.gov results records are verified and fixed-effect class estimates have a narrow local `netmeta` reference check, but this is not broad `netmeta`, `multinma`, or CmdStan parity;
- no clinical, regulatory, or HTA decision claim is made from this local artifact.

## Benchmark 5: Psoriasis PASI 90 Closed-Loop Binary Network

CT.gov arm-count network manifest: `validation/networks/psoriasis_pasi90_ctgov_binary_network.toml`

CT.gov/PubMed source snapshot: `validation/source_checks/psoriasis_pasi90_ctgov_binary_network_check.json`

Local benchmark artifact: `validation/networks/psoriasis_pasi90_ctgov_binary_network_benchmark.toml`

External `netmeta` reference output: `validation/reference_runs/psoriasis_pasi90_ctgov_binary_network_netmeta_reference.toml` validates `validation/reference_runs/psoriasis_pasi90_ctgov_binary_network_netmeta_output.json`.

Trials: FIXTURE (`NCT01358578`, PMID 25007392) and UNCOVER-3 (`NCT01646177`, PMID 26072109).

Outcome: PASI 90 responders at week 12, using CT.gov arm-level participant counts and denominators.

Scale currently tested: log odds ratio, generated from all pairwise contrasts within each multi-arm trial with no continuity correction.

Current tests:

- verify CT.gov NCT identity, completed status, outcome text, arm group IDs, responder counts, and denominators;
- verify PubMed publication identity for both trials;
- require all arm counts to satisfy `0 < events < n` because zero-cell continuity correction is not used;
- fit fixed-effect and generalized-DL random-effect multi-arm GLS models with shared-arm covariance handling;
- run a local fixed-effect node-splitting smoke diagnostic, which currently returns non-estimable after direct-edge removal because the contrast-level solver drops incomplete multi-arm cliques;
- match the source-backed multi-arm estimates and standard errors against local R `netmeta` output within deterministic tolerance.

The random-effects artifact estimates `tau2 = 0.0` with `Q = 0.4747` and `df = 1`.

Limitations:

- this is one dermatology endpoint family, not a broad closed-loop corpus;
- node-splitting is not reference matched and is not estimable on this artifact with the current contrast-level incomplete-clique policy;
- the `netmeta` reference report is a narrow evidence candidate, not full `netmeta`, `multinma`, CINeMA, or ROB-MEN parity;
- no clinical, regulatory, or HTA decision claim is made from this local artifact.

## Benchmark 6: Sitagliptin/Pioglitazone Factorial Component Smoke Benchmark

Component-NMA manifest: `validation/component/sitagliptin_pioglitazone_component.toml`

Component-NMA source snapshot: `validation/source_checks/sitagliptin_pioglitazone_component_check.json`

Component-NMA local benchmark: `validation/component/sitagliptin_pioglitazone_component_source_benchmark.toml`

Trial: NCT00722371, verified against ClinicalTrials.gov API v2.

PubMed identity: PMID 23909985, verified against PubMed ESummary.

Outcome: change from baseline in hemoglobin A1C at week 24, reported as CT.gov least-squares means with 95% confidence intervals.

Scale currently tested: percentage-point A1C change contrasts between seven single-component or combination arms.

Current tests:

- validate the manifest schema, NCT ID, PMID, CT.gov URL, PubMed URL, treatment components, and no certification effect;
- verify CT.gov trial identity, completed status, outcome title, outcome parameter type, dispersion type, arm counts, LS means, confidence limits, and treatment terms;
- verify PubMed article identity by PMID and title terms for sitagliptin, pioglitazone, and factorial design;
- derive 21 pairwise component-treatment contrasts from the seven verified CT.gov arms;
- fit the narrow fixed-effect additive component WLS core with rank and estimability checks;
- require the generated artifact to retain `certification_effect = "none"` and state that same-trial covariance is not modeled.

Limitations:

- this is a single CT.gov factorial trial, not a multi-study component network;
- arm-level LS mean contrasts are derived from reported confidence intervals;
- same-trial arm covariance is not modeled;
- the artifact is not broad `netmeta` CNMA parity and cannot support component hierarchy or clinical superiority claims.

## Benchmark 7: SGLT2 RCT/NRS Cross-Design Routing Smoke Benchmark

Cross-design manifest: `validation/cross_design/sglt2_rct_nrs_cross_design.toml`

Cross-design source snapshot: `validation/source_checks/sglt2_rct_nrs_cross_design_check.json`

Cross-design local benchmark: `validation/cross_design/sglt2_rct_nrs_cross_design_benchmark.toml`

Evidence source: PubMed abstract reported hazard ratios with NCT identifiers. The two randomized rows are DAPA-HF (NCT03036124, PMID 31535829) and EMPEROR-Reduced (NCT03057977, PMID 32865377). The two observational rows are CVD-REAL (NCT02993614, PMID 28522450) and CVD-REAL-2 (NCT02993614, PMID 29540325).

Scale currently tested: log hazard ratio derived from reported HR and 95% CI tokens.

Current tests:

- validate the manifest schema, study designs, NCT IDs, PMIDs, comparator labels, population labels, outcomes, and no certification effect;
- verify PubMed article identity and exact HR/CI tokens near a hazard-ratio or HR anchor before effects are computed;
- derive log-HR study effects separately for randomized and non-randomized rows;
- compute separated inverse-variance fixed-effect summaries by design;
- require `combined_borrowing_allowed = false` when comparator, population, or outcome definitions differ across designs;
- require the generated artifact to retain `certification_effect = "none"` and state that it is not `crossnma` reference matching.

Limitations:

- this is a cross-design routing and governance benchmark, not a Bayesian cross-design synthesis model;
- DAPA-HF and EMPEROR-Reduced are heart-failure RCTs, while CVD-REAL rows are diabetes real-world cohorts, so the estimands differ materially;
- combined RCT/NRS borrowing is blocked by design in this artifact;
- the artifact is not broad `crossnma` parity and cannot support clinical or HTA decision claims.

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
| Proof-carrying reported-HR bundle | Dynamic PubMed abstract read plus static manifest/report hash checks | `scripts/write_proof_effect_bundle.py` and `validation/ingestion/sglt2_hf_reported_hr_proof_effects.json` | Validates model-ready extracted HR records with source identity, source-check hashes, minimal source snippets, and CI uncertainty; not model-performance certification |
| Reported HR benchmark | Dynamic computation | `scripts/write_survival_hr_benchmark.py` plus `bias_nma_adv.survival_benchmark` | Recomputes log-HR study effects and pairwise pooling from verified identity and reported-HR source snapshots |
| CT.gov reported-HR network manifest | Static fixture | `validation/networks/t2d_mace_ctgov_hrs.toml` | Stores NCT IDs, class labels, drug terms, outcome-search terms, and HR/CI values that must be verified before use |
| CT.gov reported-HR network snapshot | Dynamic public API check | `scripts/verify_ctgov_hr_network.py` against ClinicalTrials.gov API v2 | Verifies NCT identity, completed status, exact HR/CI analysis fields, outcome-title terms, and drug/placebo terms |
| CT.gov reported-HR network benchmark | Dynamic computation | `scripts/write_ctgov_hr_network_benchmark.py` plus `bias_nma_adv.ctgov_hr_network` | Recomputes log-HR study effects and fixed/random contrast-GLS NMA from the verified CT.gov source snapshot |
| CT.gov/PubMed closed-loop binary NMA benchmark | Dynamic computation | `scripts/verify_ctgov_binary_network_sources.py`, `scripts/write_ctgov_binary_network_benchmark.py`, `validation/networks/psoriasis_pasi90_ctgov_binary_network_benchmark.toml`, and `tests/test_ctgov_binary_network.py` | Verifies two public CT.gov PASI 90 arm-count records plus PubMed article identities, recomputes all within-trial log-OR contrasts, and matches local `netmeta` multi-arm estimates; broad closed-loop inconsistency parity remains blocked |
| CT.gov/PubMed component-NMA smoke benchmark | Dynamic computation | `scripts/verify_component_sources.py`, `scripts/write_component_benchmark.py`, `validation/component/sitagliptin_pioglitazone_component_source_benchmark.toml`, and `tests/test_component_benchmark.py` | Verifies one factorial CT.gov/PubMed component benchmark and recomputes additive WLS contrasts; same-trial covariance and broad CNMA parity remain blocked |
| PubMed cross-design routing benchmark | Dynamic computation | `scripts/verify_cross_design_sources.py`, `scripts/write_cross_design_benchmark.py`, `validation/cross_design/sglt2_rct_nrs_cross_design_benchmark.toml`, and `tests/test_cross_design_benchmark.py` | Verifies four PubMed abstract reported-HR rows and recomputes separated RCT/NRS summaries; combined borrowing, sparse hierarchical shrinkage, and broad `crossnma` parity remain blocked |
| Proof-carrying extracted effect | Static contract plus unit fixtures | `bias_nma_adv.ingestion.ProofCarryingEffectRecord` | Blocks model-ready extracted effects unless source provenance, source snippet, uncertainty, and effect-scale sanity checks pass |
| Registry/regulatory/protocol boundary | Static source-policy contract plus unit fixtures | `bias_nma_adv.source_type_policy`, `bias_nma_adv.evidence_sources`, `bias_nma_adv.ingestion`, and source-boundary tests | Allows AACT/ClinicalTrials.gov, public numeric ICTRP/PACTR result rows, and public FDA/EMA regulatory review rows as effect sources when numeric per-trial provenance is present; rejects protocol-only registry records as model-ready effects while allowing metadata ledgers |
| Real benchmark coverage atlas | Dynamic registry-derived summary | `validation/real_benchmark_atlas.json`, `scripts/write_real_benchmark_atlas.py`, and `tests/test_real_benchmark_atlas.py` | Summarizes registered real-data benchmark coverage and explicit non-claims; not tier-one parity or clinical evidence certification |
| Source-check certification effect | Static contract | `certification_effect = "none"` in source-check artifacts | Source-token verification cannot certify model performance or tier-one parity |
| Independent fixed-effect reference | Dynamic computation | `bias_nma_adv.real_meta.fixed_effect_log_or_reference` | Recomputed by tests from the CSV rows |
| Pairwise bridge result | Dynamic computation | `bias_nma_adv.pairwise.fit_pairwise_meta` | Recomputed by tests from source-backed study-level effects |
| External pairwise reference preflight | Dynamic local environment preflight | `validation/reference_runs/pairwise_metafor_meta_preflight.toml` | Records dependency availability only and has `certification_effect = "none"` |
| External pairwise reference output | Dynamic local R execution plus Python validation | `validation/reference_runs/pairwise_metafor_meta_output.json`, `validation/reference_runs/pairwise_metafor_meta_reference.toml`, and `src/bias_nma_adv/r_reference_validation.py` | Validates narrow `metafor`/`meta` output parity for study log-OR effects, fixed-effect pooling, REML tau2/Q/df, and documented HKSJ floor differences; not production certification |
| Candidate frequentist result | Dynamic computation | `AdvancedBiasAdjustedNMAPooler` | Recomputed by tests from the CSV rows |
| Candidate Bayesian result | Dynamic seeded computation | `BayesianNMAMCMCSampler` with seed 20260715 | Recomputed by tests with tolerance for sampler behavior |
| Certification status | Static contract | `sglt2_hf_primary_benchmark.toml` | `certification_effect = "none"` until external reference matching exists |
