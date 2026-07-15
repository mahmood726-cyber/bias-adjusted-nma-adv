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

    result = run_real_meta_benchmark(
        dataset_path,
        mcmc_samples=artifact["model_config"]["bayesian_samples"],
    )
    assert result["dataset_sha256"] == artifact["dataset_sha256"]
    assert result["n_studies"] == artifact["n_studies"]
    assert result["n_arms"] == artifact["n_arms"]
    assert result["effect_scale"] == artifact["effect_scale"]

    tolerance = artifact["tolerances"]["frequentist_abs"]
    for field in ("estimate", "se", "ci_low", "ci_high"):
        assert abs(result["reference"][field] - artifact["reference"]["fixed_effect_log_or"][field]) < tolerance
        assert abs(result["frequentist"][field] - artifact["candidate"]["frequentist"][field]) < tolerance

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
