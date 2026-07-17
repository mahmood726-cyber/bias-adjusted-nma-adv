"""Run the pinned aggregate reversal-yardstick verifier.

The arena JSON lives outside the public repository. This command verifies the
provided artifact hash against ``validation/reversal_yardstick.toml`` and checks
the aggregate metrics without certifying clinical, HTA, or tier-one superiority
claims.
"""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.reversal_runner import (  # noqa: E402
    REVERSAL_ARENA_AGGREGATE_SCHEMA_VERSION,
    ReversalArenaError,
    run_reversal_arena_aggregate_from_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yardstick",
        type=Path,
        default=ROOT / "validation" / "reversal_yardstick.toml",
        help="Path to the committed reversal yardstick TOML.",
    )
    parser.add_argument(
        "--arena-json",
        type=Path,
        required=True,
        help="Path to the external aggregate arena JSON, for example C:\\key\\arena.json.",
    )
    args = parser.parse_args(argv)

    try:
        report = run_reversal_arena_aggregate_from_paths(
            args.arena_json,
            args.yardstick,
        )
    except (ReversalArenaError, OSError, JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        failure = {
            "schema_version": REVERSAL_ARENA_AGGREGATE_SCHEMA_VERSION,
            "status": "failed",
            "error": str(exc),
            "certification_effect": "none",
            "global_goal_complete": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
