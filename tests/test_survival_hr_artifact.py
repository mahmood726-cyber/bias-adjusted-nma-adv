import csv
from pathlib import Path
import tomllib

import pytest

from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.survival_benchmark import (
    run_survival_hr_benchmark,
    survival_hr_log_effects,
    load_survival_hr_manifest,
    load_survival_hr_verification_report,
    validate_survival_hr_source_bundle,
    validate_survival_hr_identity_bundle,
)
from bias_nma_adv.source_verification import load_source_verification_report


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = tuple(
    sorted((ROOT / "validation" / "survival").glob("*_reported_hr_benchmark.toml"))
)


@pytest.mark.parametrize("artifact_path", ARTIFACTS)
def test_survival_hr_benchmark_artifact_recomputes_from_verified_source_tokens(artifact_path):
    with artifact_path.open("rb") as handle:
        artifact = tomllib.load(handle)

    manifest_path = ROOT / artifact["source_manifest"]
    source_check_path = ROOT / artifact["source_verification_report"]
    identity_check_path = ROOT / artifact["source_identity_report"]
    assert sha256_file(manifest_path) == artifact["source_manifest_sha256"]
    assert sha256_file(source_check_path) == artifact["source_verification_report_sha256"]
    assert sha256_file(identity_check_path) == artifact["source_identity_report_sha256"]
    assert artifact["schema_version"] == "survival_hr_benchmark/v1"
    assert artifact["status"] == "local_pass"
    assert artifact["certification_effect"] == "none"
    assert artifact["effect_scale"] == "log_hr"
    assert "not a multi-treatment survival NMA" in " ".join(artifact["limitations"])

    manifest = load_survival_hr_manifest(manifest_path)
    assert artifact["n_studies"] == len(manifest.studies)
    assert artifact["benchmark_id"] == manifest.benchmark_id
    assert artifact["evidence_mode"] == manifest.evidence_mode
    report = load_survival_hr_verification_report(source_check_path)
    identity_report = load_source_verification_report(identity_check_path)
    source_bundle = validate_survival_hr_source_bundle(manifest, report)
    identity_bundle = validate_survival_hr_identity_bundle(manifest, identity_report)
    assert source_bundle == artifact["source_bundle"]
    assert identity_bundle == artifact["source_identity_bundle"]

    recomputed = run_survival_hr_benchmark(
        manifest_path,
        verification_report_path=source_check_path,
        identity_report_path=identity_check_path,
    )
    assert recomputed["source_manifest_sha256"] == artifact["source_manifest_sha256"]
    assert recomputed["source_verification_report_sha256"] == artifact["source_verification_report_sha256"]
    assert recomputed["source_identity_report_sha256"] == artifact["source_identity_report_sha256"]
    assert recomputed["model_config"] == artifact["model_config"]

    expected_effects = {effect.study_id: effect for effect in survival_hr_log_effects(manifest)}
    artifact_effects = {effect["study_id"]: effect for effect in artifact["study_effects"]}
    assert set(artifact_effects) == set(expected_effects)
    for study_id, expected in expected_effects.items():
        observed = artifact_effects[study_id]
        for field in (
            "study_id",
            "trial",
            "nct_id",
            "pmid",
            "outcome_id",
            "outcome_label",
            "active_treatment",
            "control_treatment",
            "effect_direction",
            "effect_scale",
            "variance_source",
        ):
            assert observed[field] == getattr(expected, field)
        for field in ("reported_hr", "ci_lower", "ci_upper", "estimate", "variance", "se"):
            assert abs(observed[field] - getattr(expected, field)) < 1e-14

    for key in ("pairwise_fixed_effect", "pairwise_reml_hksj"):
        observed = artifact["candidate"][key]
        expected = recomputed["candidate"][key]
        for field in ("method", "df", "hksj", "warnings"):
            assert observed[field] == expected[field]
        for field in ("estimate", "se", "ci_low", "ci_high", "tau2", "q", "hksj_q_factor"):
            assert abs(observed[field] - expected[field]) < 1e-14
        assert observed["weights"] == expected["weights"]
        if "pi_low" in expected:
            assert abs(observed["pi_low"] - expected["pi_low"]) < 1e-14
            assert abs(observed["pi_high"] - expected["pi_high"]) < 1e-14

    if artifact["benchmark_id"] == "sglt2_ckd_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "glp1_mace_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "parp_firstline_ovarian_pfs_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "rcc_firstline_pfs_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "nsclc_firstline_pfs_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "hfref_therapies_primary_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "doac_af_primary_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "tavi_savr_primary_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "melanoma_pfs_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "osimertinib_nsclc_pfs_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "lipid_cv_outcomes_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0
    if artifact["benchmark_id"] == "her2_breast_pfs_reported_hr":
        assert artifact["candidate"]["pairwise_reml_hksj"]["tau2"] > 0.0


@pytest.mark.parametrize("artifact_path", ARTIFACTS)
def test_survival_hr_effects_csv_matches_source_backed_artifact(artifact_path):
    with artifact_path.open("rb") as handle:
        artifact = tomllib.load(handle)
    csv_path = ROOT / "validation" / "survival" / f"{artifact['benchmark_id']}_effects.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected = {row["study_id"]: row for row in artifact["study_effects"]}
    observed = {row["study_id"]: row for row in rows}
    assert set(observed) == set(expected)
    for study_id, expected_row in expected.items():
        observed_row = observed[study_id]
        assert observed_row["nct_id"] == expected_row["nct_id"]
        assert observed_row["pmid"] == str(expected_row["pmid"])
        assert float(observed_row["estimate"]) == pytest.approx(expected_row["estimate"])
        assert float(observed_row["se"]) == pytest.approx(expected_row["se"])
        assert float(observed_row["variance"]) == pytest.approx(expected_row["variance"])
