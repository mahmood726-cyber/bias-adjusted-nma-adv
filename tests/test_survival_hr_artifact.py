from pathlib import Path
import tomllib

from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.survival_benchmark import (
    run_survival_hr_benchmark,
    survival_hr_log_effects,
    load_survival_hr_manifest,
    load_survival_hr_verification_report,
    validate_survival_hr_source_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "survival" / "sglt2_hf_reported_hr_benchmark.toml"


def test_survival_hr_benchmark_artifact_recomputes_from_verified_source_tokens():
    with ARTIFACT.open("rb") as handle:
        artifact = tomllib.load(handle)

    manifest_path = ROOT / artifact["source_manifest"]
    source_check_path = ROOT / artifact["source_verification_report"]
    assert sha256_file(manifest_path) == artifact["source_manifest_sha256"]
    assert sha256_file(source_check_path) == artifact["source_verification_report_sha256"]
    assert artifact["schema_version"] == "survival_hr_benchmark/v1"
    assert artifact["status"] == "local_pass"
    assert artifact["certification_effect"] == "none"
    assert artifact["effect_scale"] == "log_hr"
    assert artifact["n_studies"] == 4
    assert "not a multi-treatment survival NMA" in " ".join(artifact["limitations"])

    manifest = load_survival_hr_manifest(manifest_path)
    report = load_survival_hr_verification_report(source_check_path)
    source_bundle = validate_survival_hr_source_bundle(manifest, report)
    assert source_bundle == artifact["source_bundle"]

    recomputed = run_survival_hr_benchmark(
        manifest_path,
        verification_report_path=source_check_path,
    )
    assert recomputed["source_manifest_sha256"] == artifact["source_manifest_sha256"]
    assert recomputed["source_verification_report_sha256"] == artifact["source_verification_report_sha256"]
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
