from pathlib import Path
import tomllib

from bias_nma_adv.real_meta import run_real_meta_benchmark, sha256_file


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_benchmark.toml"


def test_sglt2_benchmark_artifact_recomputes_from_source_rows():
    with ARTIFACT.open("rb") as handle:
        artifact = tomllib.load(handle)

    dataset_path = ROOT / artifact["dataset"]
    assert sha256_file(dataset_path) == artifact["dataset_sha256"]
    source_manifest_path = ROOT / artifact["source_manifest"]
    assert source_manifest_path.is_file()
    assert sha256_file(source_manifest_path) == artifact["source_manifest_sha256"]

    model_config = artifact["model_config"]
    result = run_real_meta_benchmark(
        dataset_path,
        source_manifest_path=source_manifest_path,
        mcmc_samples=model_config["bayesian_samples"],
        reference_treatment=model_config["reference_treatment"],
        candidate_treatment=model_config["candidate_treatment"],
    )
    assert result["dataset_sha256"] == artifact["dataset_sha256"]
    assert result["source_manifest"]["manifest_sha256"] == sha256_file(source_manifest_path)
    assert result["n_studies"] == artifact["n_studies"]
    assert result["n_arms"] == artifact["n_arms"]
    assert result["effect_scale"] == artifact["effect_scale"]
    for field in (
        "reference_treatment",
        "candidate_treatment",
        "pairwise_fixed_effect_method",
        "pairwise_random_effect_method",
        "pairwise_hksj",
        "pairwise_hksj_floor",
        "pairwise_prediction_interval",
        "pairwise_prediction_interval_df",
        "level",
        "bayesian_samples",
        "bayesian_seed",
    ):
        assert result["model_config"][field] == model_config[field]

    artifact_effects = {effect["study_id"]: effect for effect in artifact["study_effects"]}
    result_effects = {effect["study_id"]: effect for effect in result["study_effects"]}
    assert set(result_effects) == set(artifact_effects)
    for study_id, expected in artifact_effects.items():
        observed = result_effects[study_id]
        for field in (
            "study_id",
            "trial",
            "nct_id",
            "pmid",
            "outcome_id",
            "outcome_label",
            "active_treatment",
            "control_treatment",
            "active_events",
            "active_n",
            "control_events",
            "control_n",
            "effect_direction",
            "effect_scale",
            "continuity_correction",
        ):
            assert observed[field] == expected[field]
        for field in ("estimate", "variance", "se"):
            assert abs(observed[field] - expected[field]) < artifact["tolerances"]["frequentist_abs"]

    tolerance = artifact["tolerances"]["frequentist_abs"]
    for field in ("estimate", "se", "ci_low", "ci_high"):
        assert abs(result["reference"][field] - artifact["reference"]["fixed_effect_log_or"][field]) < tolerance
        assert abs(result["frequentist"][field] - artifact["candidate"]["frequentist"][field]) < tolerance
        assert (
            abs(result["pairwise"]["fixed_effect"][field] - artifact["candidate"]["pairwise_fixed_effect"][field])
            < tolerance
        )

    pairwise_tol = artifact["tolerances"]["frequentist_abs"]
    for field in ("estimate", "se", "ci_low", "ci_high", "tau2", "q", "hksj_q_factor"):
        assert (
            abs(result["pairwise"]["fixed_effect"][field] - artifact["candidate"]["pairwise_fixed_effect"][field])
            < pairwise_tol
        )
    for field in ("estimate", "se", "ci_low", "ci_high", "tau2", "q", "hksj_q_factor", "pi_low", "pi_high"):
        assert (
            abs(result["pairwise"]["reml_hksj"][field] - artifact["candidate"]["pairwise_reml_hksj"][field])
            < pairwise_tol
        )
    assert result["pairwise"]["fixed_effect"]["warnings"] == artifact["candidate"]["pairwise_fixed_effect"]["warnings"]
    assert result["pairwise"]["reml_hksj"]["warnings"] == artifact["candidate"]["pairwise_reml_hksj"]["warnings"]
    assert result["pairwise"]["fixed_effect"]["method"] == artifact["candidate"]["pairwise_fixed_effect"]["method"]
    assert result["pairwise"]["reml_hksj"]["method"] == artifact["candidate"]["pairwise_reml_hksj"]["method"]
    assert result["pairwise"]["fixed_effect"]["df"] == artifact["candidate"]["pairwise_fixed_effect"]["df"]
    assert result["pairwise"]["reml_hksj"]["df"] == artifact["candidate"]["pairwise_reml_hksj"]["df"]
    assert result["pairwise"]["fixed_effect"]["hksj"] is artifact["candidate"]["pairwise_fixed_effect"]["hksj"]
    assert result["pairwise"]["reml_hksj"]["hksj"] is artifact["candidate"]["pairwise_reml_hksj"]["hksj"]
    assert len(result["pairwise"]["fixed_effect"]["weights"]) == len(artifact["candidate"]["pairwise_fixed_effect"]["weights"])
    assert len(result["pairwise"]["reml_hksj"]["weights"]) == len(artifact["candidate"]["pairwise_reml_hksj"]["weights"])
    for observed, expected in zip(
        result["pairwise"]["fixed_effect"]["weights"],
        artifact["candidate"]["pairwise_fixed_effect"]["weights"],
    ):
        assert abs(observed - expected) < pairwise_tol
    for observed, expected in zip(
        result["pairwise"]["reml_hksj"]["weights"],
        artifact["candidate"]["pairwise_reml_hksj"]["weights"],
    ):
        assert abs(observed - expected) < pairwise_tol

    bayes_tol = artifact["tolerances"]["bayesian_mean_abs"]
    assert (
        abs(
            result["bayesian"]["posterior_mean"]
            - artifact["candidate"]["bayesian_mcmc"]["posterior_mean"]
        )
        < bayes_tol
    )
    assert (
        artifact["tolerances"]["bayesian_acceptance_min"]
        <= result["bayesian"]["acceptance_rate"]
        <= artifact["tolerances"]["bayesian_acceptance_max"]
    )
