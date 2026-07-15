"""Write the source-backed real benchmark coverage atlas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.real_benchmark_atlas import write_real_benchmark_atlas  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT, help="Repository root.")
    parser.add_argument(
        "--output",
        default="validation/real_benchmark_atlas.json",
        help="Output JSON path.",
    )
    parser.add_argument("--checked-at", default=None, help="ISO timestamp to store in the atlas.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atlas = write_real_benchmark_atlas(root, output, checked_at=args.checked_at)
    print(
        "real benchmark atlas written: "
        f"{output} ({atlas['n_benchmarks']} benchmarks, "
        f"{atlas['n_benchmark_study_effects']} study effects)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
