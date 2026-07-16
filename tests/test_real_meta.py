from pathlib import Path
import json

import pytest

from bias_nma_adv.real_meta import (
    ArmEventRow,
    build_dataset_from_arm_events,
    fixed_effect_log_or_reference,
    load_arm_event_rows,
    run_real_meta_benchmark,
    sha256_file,
    study_log_or_effects,
    validate_real_meta_source_manifest,
)
from bias_nma_adv.data import ValidationError
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler


ROOT = Path(__file__).resolve().parents[1]
SGLT2_EVENTS = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_events.csv"
SGLT2_SOURCES = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_sources.toml"
PAIRWISE_METAFOR_OUTPUT = ROOT / "validation" / "reference_runs" / "pairwise_metafor_meta_output.json"


def test_text_artifact_sha256_is_line_ending_canonical(tmp_path):
    lf = tmp_path / "artifact_lf.json"
    crlf = tmp_path / "artifact_crlf.json"
    lf.write_bytes(b'{\n  "nct_id": "NCT03036124"\n}\n')
    crlf.write_bytes(b'{\r\n  "nct_id": "NCT03036124"\r\n}\r\n')

    assert sha256_file(lf) == sha256_file(crlf)


def test_sglt2_hf_fixture_is_source_backed_real_data():
    rows = load_arm_event_rows(SGLT2_EVENTS)

    assert len(rows) == 8
    assert {row.study_id for row in rows} == {
        "DAPA-HF",
        "EMPEROR-Reduced",
        "DELIVER",
        "EMPEROR-Preserved",
    }
    assert {row.nct_id for row in rows} == {
        "NCT03036124",
        "NCT03057977",
        "NCT03619213",
        "NCT03057951",
    }
    assert {row.pmid for row in rows} == {
        "31535829",
        "32865377",
        "36027570",
        "34449189",
    }


def test_fixed_effect_reference_is_clinically_directional():
    rows = load_arm_event_rows(SGLT2_EVENTS)
    reference = fixed_effect_log_or_reference(rows)

    assert reference.estimate < 0.0
    assert reference.ci_low < reference.estimate < reference.ci_high


def test_study_log_or_effects_preserve_source_identifiers():
    rows = load_arm_event_rows(SGLT2_EVENTS)
    effects = study_log_or_effects(rows)

    assert [effect.study_id for effect in effects] == [
        "DAPA-HF",
        "DELIVER",
        "EMPEROR-Preserved",
        "EMPEROR-Reduced",
    ]
    assert {effect.nct_id for effect in effects} == {
        "NCT03036124",
        "NCT03619213",
        "NCT03057951",
        "NCT03057977",
    }
    assert {effect.pmid for effect in effects} == {
        "31535829",
        "36027570",
        "34449189",
        "32865377",
    }
    by_study = {effect.study_id: effect for effect in effects}
    dapa = by_study["DAPA-HF"]
    assert dapa.active_treatment == "SGLT2i"
    assert dapa.control_treatment == "Placebo"
    assert dapa.active_events == 386
    assert dapa.active_n == 2373
    assert dapa.control_events == 502
    assert dapa.control_n == 2371
    assert dapa.effect_direction == "active_vs_control"
    assert dapa.effect_scale == "log_or"
    assert dapa.continuity_correction == "none_zero_cells_fail_closed"
    assert all(effect.variance > 0.0 and effect.se > 0.0 for effect in effects)
    assert all(abs(effect.se**2 - effect.variance) < 1e-15 for effect in effects)


def test_sglt2_source_manifest_matches_csv_rows_and_allowed_sources():
    rows = load_arm_event_rows(SGLT2_EVENTS)
    summary = validate_real_meta_source_manifest(
        rows,
        SGLT2_SOURCES,
        dataset_path=SGLT2_EVENTS,
    )

    assert summary["n_studies"] == 4
    assert summary["n_sources"] == 8
    assert summary["source_types"] == ["clinicaltrials_gov", "pubmed_abstract"]


def test_real_meta_rows_fail_closed_on_within_study_identifier_mismatch():
    rows = load_arm_event_rows(SGLT2_EVENTS)
    bad_rows = []
    for row in rows:
        if row.study_id == "DAPA-HF" and row.arm_role == "control":
            bad_rows.append(
                ArmEventRow(
                    study_id=row.study_id,
                    trial=row.trial,
                    nct_id="NCT03057977",
                    pmid=row.pmid,
                    outcome_id=row.outcome_id,
                    outcome_label=row.outcome_label,
                    arm_role=row.arm_role,
                    treatment=row.treatment,
                    events=row.events,
                    n=row.n,
                )
            )
        else:
            bad_rows.append(row)

    from bias_nma_adv.real_meta import _validate_study_pairs

    with pytest.raises(ValidationError, match="mixed nct_id"):
        _validate_study_pairs(bad_rows)


def test_real_meta_rows_fail_closed_on_duplicate_treatments_in_contrast():
    rows = load_arm_event_rows(SGLT2_EVENTS)
    bad_rows = []
    for row in rows:
        if row.study_id == "DAPA-HF" and row.arm_role == "control":
            bad_rows.append(
                ArmEventRow(
                    study_id=row.study_id,
                    trial=row.trial,
                    nct_id=row.nct_id,
                    pmid=row.pmid,
                    outcome_id=row.outcome_id,
                    outcome_label=row.outcome_label,
                    arm_role=row.arm_role,
                    treatment="SGLT2i",
                    events=row.events,
                    n=row.n,
                )
            )
        else:
            bad_rows.append(row)

    from bias_nma_adv.real_meta import _validate_study_pairs

    with pytest.raises(ValidationError, match="distinct treatments"):
        _validate_study_pairs(bad_rows)


def test_source_manifest_fails_closed_on_arm_count_mismatch(tmp_path):
    manifest = tmp_path / "bad_sources.toml"
    text = SGLT2_SOURCES.read_text(encoding="utf-8").replace("events = 386", "events = 999", 1)
    manifest.write_text(text, encoding="utf-8")
    rows = load_arm_event_rows(SGLT2_EVENTS)

    with pytest.raises(ValidationError, match="events does not match"):
        validate_real_meta_source_manifest(rows, manifest, dataset_path=SGLT2_EVENTS)


def test_source_manifest_fails_closed_on_source_identifier_mismatch(tmp_path):
    rows = load_arm_event_rows(SGLT2_EVENTS)

    bad_ctgov = tmp_path / "bad_ctgov_sources.toml"
    bad_ctgov.write_text(
        SGLT2_SOURCES.read_text(encoding="utf-8").replace(
            'identifier = "NCT03036124"',
            'identifier = "NCT03057977"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="identifier does not match CSV NCT ID"):
        validate_real_meta_source_manifest(rows, bad_ctgov, dataset_path=SGLT2_EVENTS)

    bad_pubmed = tmp_path / "bad_pubmed_sources.toml"
    bad_pubmed.write_text(
        SGLT2_SOURCES.read_text(encoding="utf-8").replace(
            'identifier = "31535829"',
            'identifier = "32865377"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="identifier does not match CSV PMID"):
        validate_real_meta_source_manifest(rows, bad_pubmed, dataset_path=SGLT2_EVENTS)


def test_source_manifest_fails_closed_on_missing_pubmed_event_count_terms(tmp_path):
    rows = load_arm_event_rows(SGLT2_EVENTS)
    manifest = tmp_path / "bad_source_terms.toml"
    manifest.write_text(
        SGLT2_SOURCES.read_text(encoding="utf-8").replace(
            'active_source_terms = ["dapagliflozin"]',
            'active_source_terms = []',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="active_source_terms"):
        validate_real_meta_source_manifest(rows, manifest, dataset_path=SGLT2_EVENTS)


def test_real_meta_benchmark_runs_frequentist_and_bayesian_models():
    result = run_real_meta_benchmark(
        SGLT2_EVENTS,
        source_manifest_path=SGLT2_SOURCES,
        mcmc_samples=300,
    )

    assert result["n_studies"] == 4
    assert result["effect_scale"] == "log_or"
    assert result["source_manifest"]["n_sources"] == 8
    assert result["model_config"]["reference_treatment"] == "Placebo"
    assert result["model_config"]["candidate_treatment"] == "SGLT2i"
    assert result["model_config"]["pairwise_prediction_interval_df"] == "k_minus_1"

    ref = result["reference"]
    freq = result["frequentist"]
    pairwise = result["pairwise"]
    assert abs(freq["estimate"] - ref["estimate"]) < 1e-10
    assert abs(freq["se"] - ref["se"]) < 1e-10
    assert freq["ci_low"] < freq["estimate"] < freq["ci_high"]
    assert freq["warnings"] == []
    assert abs(pairwise["fixed_effect"]["estimate"] - ref["estimate"]) < 1e-10
    assert abs(pairwise["fixed_effect"]["se"] - ref["se"]) < 1e-10
    assert pairwise["fixed_effect"]["tau2"] == 0.0
    assert pairwise["fixed_effect"]["warnings"] == []
    assert pairwise["reml_hksj"]["method"] == "REML"
    assert pairwise["reml_hksj"]["hksj"] is True
    assert pairwise["reml_hksj"]["hksj_q_factor"] >= 1.0
    assert pairwise["reml_hksj"]["pi_low"] < pairwise["reml_hksj"]["estimate"] < pairwise["reml_hksj"]["pi_high"]

    bayes = result["bayesian"]
    assert bayes["credible_interval"][0] < bayes["posterior_mean"] < bayes["credible_interval"][1]
    assert bayes["posterior_sd"] > 0.0
    assert 0.05 <= bayes["acceptance_rate"] <= 0.95
    assert abs(bayes["posterior_mean"] - ref["estimate"]) < 0.25


def test_unadjusted_pooler_matches_metafor_fixed_effect_golden_reference():
    rows = load_arm_event_rows(SGLT2_EVENTS)
    dataset = build_dataset_from_arm_events(rows)
    assert all(study.rob_weight == 1.0 for study in dataset.studies.values())
    assert all(not study.covariates for study in dataset.studies.values())

    fit = AdvancedBiasAdjustedNMAPooler(hksj=False, random_effects=False).fit(
        dataset,
        "hf_primary",
        reference_treatment="Placebo",
    )
    reference = json.loads(PAIRWISE_METAFOR_OUTPUT.read_text(encoding="utf-8"))["metafor"][
        "fixed_effect"
    ]

    assert fit.treatment_effects["SGLT2i"] == pytest.approx(reference["estimate"], abs=1e-12)
    assert fit.treatment_ses["SGLT2i"] == pytest.approx(reference["se"], abs=1e-12)
    assert fit.taus == {"rct": 0.0}
    assert fit.warnings == ()
