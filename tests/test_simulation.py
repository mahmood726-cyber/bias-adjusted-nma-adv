"""Unit tests for the simulation and benchmarking framework."""

from bias_nma_adv.simulation import generate_synthetic_nma_dataset, run_benchmark_simulation

def test_generate_synthetic_nma_dataset():
    dataset = generate_synthetic_nma_dataset(n_studies=10, n_treatments=3, seed=42)
    assert len(dataset.studies) == 10
    assert len(dataset.outcomes_ad) >= 20  # at least 2 arms per study

    # Verify that study design is rct or nrs
    for study in dataset.studies.values():
        assert study.design in {"rct", "nrs"}
        assert 0.0 < study.rob_weight <= 1.0
        assert "year" in study.covariates


def test_run_benchmark_simulation():
    # Run a tiny simulation to verify the pipeline
    results = run_benchmark_simulation(
        n_iterations=5,
        n_studies=8,
        n_treatments=3,
        seed=101
    )

    assert results["iterations_attempted"] == 5
    assert results["iterations_successful"] > 0
    assert results["n_studies"] == 8
    assert results["n_treatments"] == 3

    # Check that methods are in the summary
    summary = results["methods_summary"]
    assert "standard_nma" in summary
    assert "standard_bias_adj" in summary
    assert "hksj_bias_adj" in summary
    assert "hksj_weighted_bias_adj" in summary
    assert "full_advanced_bias_adj" in summary

    # Verify key metrics are present
    for method_name, metrics in summary.items():
        assert "bias" in metrics
        assert "mse" in metrics
        assert "coverage" in metrics
        assert "mean_se" in metrics
