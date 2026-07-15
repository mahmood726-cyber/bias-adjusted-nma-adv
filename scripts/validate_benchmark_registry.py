"""Validate local source-backed benchmark registry artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.benchmark_registry import (  # noqa: E402
    BenchmarkRegistryError,
    assert_registry_covers_source_backed_artifacts,
    validate_source_benchmark_registry,
)


DEFAULT_REGISTRY = Path("validation/benchmark_registry.toml")


def build_summary(root: Path, registry_path: Path) -> dict[str, object]:
    """Validate the benchmark registry and return a machine-readable summary."""

    registry = validate_source_benchmark_registry(registry_path, repo_root=root)
    assert_registry_covers_source_backed_artifacts(registry, repo_root=root)
    by_domain: dict[str, int] = {}
    for entry in registry.benchmarks:
        by_domain[entry.domain] = by_domain.get(entry.domain, 0) + 1
    return {
        "schema_version": "benchmark_registry_validation/v1",
        "status": "passed",
        "registry": registry_path.relative_to(root).as_posix()
        if registry_path.is_relative_to(root)
        else registry_path.as_posix(),
        "checked_at": registry.checked_at,
        "n_benchmarks": len(registry.benchmarks),
        "benchmark_ids": [entry.id for entry in registry.benchmarks],
        "domains": dict(sorted(by_domain.items())),
        "certification_effect": "none",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--json", action="store_true", help="Emit a JSON summary.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    try:
        summary = build_summary(root, registry_path.resolve())
    except (BenchmarkRegistryError, OSError, json.JSONDecodeError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": "benchmark_registry_validation/v1",
                        "status": "failed",
                        "registry": registry_path.as_posix(),
                        "error": str(exc),
                        "certification_effect": "none",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"benchmark registry validation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "benchmark registry validation passed: "
            f"{summary['n_benchmarks']} source-backed benchmarks; certification_effect=none"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
