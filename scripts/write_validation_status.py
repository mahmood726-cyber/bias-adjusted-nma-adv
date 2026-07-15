"""Emit the unified validation status report for CI/Overmind-style gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.benchmark_registry import BenchmarkRegistryError  # noqa: E402
from bias_nma_adv.certification import CertificationError  # noqa: E402
from bias_nma_adv.grand_benchmark_plan import GrandBenchmarkPlanError  # noqa: E402
from bias_nma_adv.simulation_matrix import SimulationMatrixError  # noqa: E402
from bias_nma_adv.validation_status import (  # noqa: E402
    VALIDATION_STATUS_SCHEMA_VERSION,
    build_validation_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--checked-at",
        help="Optional deterministic timestamp for reproducible reports.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        report = build_validation_status(root, checked_at=args.checked_at)
    except (
        BenchmarkRegistryError,
        CertificationError,
        GrandBenchmarkPlanError,
        SimulationMatrixError,
        OSError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        failure = {
            "schema_version": VALIDATION_STATUS_SCHEMA_VERSION,
            "status": "failed",
            "repository": root.name,
            "error": str(exc),
            "certification_effect": "none",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return 0

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    print(f"validation status written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
