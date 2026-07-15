"""Reproducibility script to execute the simulation benchmark and save metrics."""

import json
from pathlib import Path
from bias_nma_adv.simulation import run_benchmark_simulation

def main():
    print("=" * 80)
    print("RUNNING ADVANCED BIAS-ADJUSTED NMA BENCHMARK SIMULATION")
    print("=" * 80)
    
    # Run 200 iterations for the benchmark to get stable coverage/bias metrics
    n_iterations = 200
    n_studies = 30
    n_treatments = 4
    
    print(f"Parameters:\n - Iterations: {n_iterations}\n - Studies per dataset: {n_studies}\n - Treatments: {n_treatments}")
    print("Running simulation (this fits 1,000 NMA models)...")
    
    results = run_benchmark_simulation(
        n_iterations=n_iterations,
        n_studies=n_studies,
        n_treatments=n_treatments,
        seed=101
    )
    
    print("\nBenchmark completed successfully!")
    print("-" * 80)
    print(f"{'Method':<30} | {'Bias':<10} | {'RMSE':<10} | {'Coverage':<10} | {'Mean SE':<10}")
    print("-" * 80)
    
    summary = results["methods_summary"]
    for method, metrics in summary.items():
        print(f"{method:<30} | {metrics['bias']:<10.4f} | {metrics['rmse']:<10.4f} | {metrics['coverage']:<10.2%} | {metrics['mean_se']:<10.4f}")
    print("-" * 80)
    
    # Save results to output/simulation_results.json
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "simulation_results.json"
    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved complete benchmark metrics to {output_file}")
    
if __name__ == "__main__":
    main()
