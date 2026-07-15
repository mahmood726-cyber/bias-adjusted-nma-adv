"""Run the non-certifying simulation matrix and emit a JSON report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.simulation_matrix import (  # noqa: E402
    SIMULATION_MATRIX_REPORT_SCHEMA_VERSION,
    SimulationMatrixError,
    build_simulation_matrix_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--matrix", type=Path, default=Path("validation/simulation_matrix.toml"))
    parser.add_argument(
        "--grand-benchmark-plan",
        type=Path,
        default=Path("validation/grand_benchmark_plan.toml"),
    )
    parser.add_argument("--checked-at", help="Optional deterministic timestamp.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path. Prints to stdout when omitted.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    plan_path = (
        args.grand_benchmark_plan
        if args.grand_benchmark_plan.is_absolute()
        else root / args.grand_benchmark_plan
    )
    checked_at = args.checked_at or _utc_now()
    try:
        report = build_simulation_matrix_report(
            matrix_path,
            grand_benchmark_plan_path=plan_path,
            checked_at=checked_at,
        )
    except (SimulationMatrixError, OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        failure = {
            "schema_version": SIMULATION_MATRIX_REPORT_SCHEMA_VERSION,
            "status": "failed",
            "checked_at": checked_at,
            "error": str(exc),
            "uses_real_data": False,
            "certification_effect": "none",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output_path = args.output if args.output.is_absolute() else root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"simulation matrix report written: {output_path}")
    return 0 if report["status"] == "passed" else 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
