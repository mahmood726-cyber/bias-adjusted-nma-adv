"""Scan local portfolio repositories registered as reuse candidates."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.portfolio_reuse import (  # noqa: E402
    PORTFOLIO_REUSE_SCAN_SCHEMA_VERSION,
    PortfolioReuseError,
    build_portfolio_reuse_scan_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=ROOT / "validation" / "portfolio_reuse_sources.toml",
        type=Path,
        help="Portfolio reuse registry TOML.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        type=Path,
        help="Portfolio root directory to scan. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--checked-at",
        default=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        help="UTC timestamp to include in the report.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    try:
        report = build_portfolio_reuse_scan_report(
            registry_path=args.registry,
            roots=args.root,
            checked_at=args.checked_at,
        )
    except (PortfolioReuseError, OSError, tomllib.TOMLDecodeError) as exc:
        report = {
            "schema_version": PORTFOLIO_REUSE_SCAN_SCHEMA_VERSION,
            "checked_at": args.checked_at,
            "status": "failed",
            "certification_effect": "none",
            "error": str(exc),
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 1

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"portfolio reuse scan written: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
