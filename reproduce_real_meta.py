"""Run source-backed real meta-analysis benchmarks."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from bias_nma_adv.real_meta import run_real_meta_benchmark


def main() -> None:
    result = run_real_meta_benchmark(
        Path("validation/real_meta/sglt2_hf_primary_events.csv"),
        source_manifest_path=Path("validation/real_meta/sglt2_hf_primary_sources.toml"),
        mcmc_samples=1200,
    )
    pprint(result)


if __name__ == "__main__":
    main()
