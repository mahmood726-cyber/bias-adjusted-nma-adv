from pathlib import Path

from bias_nma_adv.real_meta import (
    fixed_effect_log_or_reference,
    load_arm_event_rows,
    run_real_meta_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
SGLT2_EVENTS = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_events.csv"


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


def test_real_meta_benchmark_runs_frequentist_and_bayesian_models():
    result = run_real_meta_benchmark(SGLT2_EVENTS, mcmc_samples=300)

    assert result["n_studies"] == 4
    assert result["effect_scale"] == "log_or"

    ref = result["reference"]
    freq = result["frequentist"]
    assert abs(freq["estimate"] - ref["estimate"]) < 1e-10
    assert abs(freq["se"] - ref["se"]) < 1e-10
    assert freq["ci_low"] < freq["estimate"] < freq["ci_high"]
    assert freq["warnings"] == []

    bayes = result["bayesian"]
    assert bayes["credible_interval"][0] < bayes["posterior_mean"] < bayes["credible_interval"][1]
    assert bayes["posterior_sd"] > 0.0
    assert 0.05 <= bayes["acceptance_rate"] <= 0.95
    assert abs(bayes["posterior_mean"] - ref["estimate"]) < 0.25
