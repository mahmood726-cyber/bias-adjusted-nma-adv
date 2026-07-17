"""Verify external reversal-yardstick source artifact pins.

This script is intentionally separate from the default validation-status writer
because the pinned files live outside the public repository. When a caller
provides paths, drift fails closed; missing paths are reported distinctly and
also fail unless ``--allow-unavailable`` is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.reversal_yardstick import (  # noqa: E402
    REVERSAL_YARDSTICK_SCHEMA_VERSION,
    ReversalYardstickError,
    load_reversal_yardstick,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yardstick",
        type=Path,
        default=ROOT / "validation" / "reversal_yardstick.toml",
        help="Path to the reversal yardstick TOML.",
    )
    parser.add_argument(
        "--source-artifact",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="External source artifact to verify, for example fix_md=/path/to/FIX.md.",
    )
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Exit 0 when a requested artifact path is missing; drift still fails.",
    )
    args = parser.parse_args(argv)

    try:
        if not args.source_artifact:
            raise ReversalYardstickError("at least one --source-artifact ID=PATH is required.")
        yardstick = load_reversal_yardstick(args.yardstick)
        report = yardstick.verify_source_artifact_pins(
            _parse_source_artifacts(args.source_artifact),
        )
    except (ReversalYardstickError, OSError, tomllib.TOMLDecodeError) as exc:
        failure = {
            "schema_version": REVERSAL_YARDSTICK_SCHEMA_VERSION,
            "status": "failed",
            "error": str(exc),
            "certification_effect": "none",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    report = {
        **report,
        "certification_effect": "none",
        "yardstick": _display_path(args.yardstick),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "unavailable" and not args.allow_unavailable:
        return 1
    return 0


def _parse_source_artifacts(raw_items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ReversalYardstickError(
                "--source-artifact entries must use ID=PATH syntax."
            )
        artifact_id, path_text = raw.split("=", 1)
        artifact_id = artifact_id.strip()
        path_text = path_text.strip()
        if not artifact_id or not path_text:
            raise ReversalYardstickError(
                "--source-artifact entries must include a non-empty ID and PATH."
            )
        parsed[artifact_id] = Path(path_text)
    return parsed


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
