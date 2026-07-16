<!-- sentinel:skip-file -->
# Review Artifact Policy

Checked: 2026-07-15

This repository contains historical adversarial-review and journal-review notes under:

- `docs/adversarial_review*.md`
- `docs/journal_peer_review.md`
- `docs/methods_evaluation.md`

These files are critique and planning artifacts. They are not validation evidence, not source extraction evidence, not reference-software parity evidence, not certification evidence, not clinical guidance, and not claims that the current implementation is superior to tier-one methods.

Any statement in those files about treatment ranking, clinical recommendations, numerical effect estimates, hazard ratios, guideline implications, superiority over standard software, or production readiness must be treated as a hypothesis until it is backed by machine-verifiable artifacts in this repository.

Acceptable evidence artifacts are limited to:

- source manifests constrained to AACT/ClinicalTrials.gov, numeric ICTRP/PACTR result rows, PubMed abstracts, open-access papers, or public FDA/EMA regulatory review rows with numeric per-trial provenance;
- source-verification snapshots under `validation/source_checks/`;
- benchmark contracts under `validation/real_meta/` or `validation/survival/`;
- reference-run reports under `validation/reference_runs/`;
- tests that recompute or validate the claimed value from the source artifact;
- future CI reports satisfying the build and validation schema in `docs/technical_specification.md`.

Historical review notes can guide what to test next. They cannot upgrade certification status, justify a clinical or HTA export, or support a claim that a native estimator beats `netmeta`, `multinma`, `MBNMAdose`, `crossnma`, `metafor`, or `meta`.

## Static-Vs-Dynamic Hardcode Disclosure

| Item | Static or dynamic | Evidence source | Disclosure |
| --- | --- | --- | --- |
| Review-artifact file patterns | Static | Local docs directory checked on 2026-07-15 | Used to scope non-evidence files, not to certify claims |
| Claims inside review notes | Untrusted until reproduced | Historical review transcripts | Must be validated through source-backed artifacts before use |
| Certification status | Dynamic | `validation/reference_targets.toml` plus reference-run reports | Review notes cannot change status |
| Clinical or HTA suitability | Dynamic | Future production-certified CI artifacts only | No current review note authorizes clinical or HTA reporting |
