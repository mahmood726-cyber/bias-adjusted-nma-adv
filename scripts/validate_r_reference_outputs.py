"""Validate generated R reference outputs and write passed candidate reports."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.r_reference_validation import (  # noqa: E402
    RReferenceValidationError,
    load_r_reference_output,
    validate_multiarm_netmeta_output,
    validate_pairwise_metafor_meta_output,
)
from bias_nma_adv.real_meta import sha256_file  # noqa: E402


PAIRWISE_OUTPUT = Path("validation/reference_runs/pairwise_metafor_meta_output.json")
PAIRWISE_REPORT = Path("validation/reference_runs/pairwise_metafor_meta_reference.toml")
MULTIARM_OUTPUT = Path("validation/reference_runs/multiarm_netmeta_output.json")
MULTIARM_REPORT = Path("validation/reference_runs/multiarm_netmeta_reference.toml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pairwise-output", type=Path, default=PAIRWISE_OUTPUT)
    parser.add_argument("--pairwise-report", type=Path, default=PAIRWISE_REPORT)
    parser.add_argument("--multiarm-output", type=Path, default=MULTIARM_OUTPUT)
    parser.add_argument("--multiarm-report", type=Path, default=MULTIARM_REPORT)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--checked-at",
        default=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        pairwise_output = _resolve(root, args.pairwise_output)
        multiarm_output = _resolve(root, args.multiarm_output)
        pairwise_summary = validate_pairwise_metafor_meta_output(
            pairwise_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        multiarm_summary = validate_multiarm_netmeta_output(
            multiarm_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        _write_report(
            _resolve(root, args.pairwise_report),
            _pairwise_report(
                root=root,
                output_path=pairwise_output,
                summary=pairwise_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
        _write_report(
            _resolve(root, args.multiarm_report),
            _multiarm_report(
                root=root,
                output_path=multiarm_output,
                summary=multiarm_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
    except (OSError, RReferenceValidationError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"R reference output validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"R reference output validation passed: {args.pairwise_report}, {args.multiarm_report}")
    return 0


def _pairwise_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/real_meta/sglt2_hf_primary_events.csv"),
        Path("validation/real_meta/sglt2_hf_primary_sources.toml"),
        Path("validation/real_meta/sglt2_hf_primary_benchmark.toml"),
        Path("external/r/pairwise_metafor_meta.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "pairwise_metafor_meta",
        "adapter_id": "r_metafor_meta_pairwise_output_validation",
        "reference_method": "metafor and meta",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/pairwise_metafor_meta.R",
            "--events",
            "validation/real_meta/sglt2_hf_primary_events.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": f"absolute <= {tolerance:g} for validated components",
        "skip_reason": "",
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [summary["hksj_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _multiarm_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/multiarm/netmeta_portfolio_multiarm_arms.csv"),
        Path("validation/multiarm/netmeta_portfolio_multiarm_benchmark.toml"),
        Path("external/r/multiarm_netmeta_fixture.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "multiarm_gls_netmeta_portfolio_fixture",
        "adapter_id": "r_netmeta_multiarm_output_validation",
        "reference_method": "netmeta",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/multiarm_netmeta_fixture.R",
            "--arms",
            "validation/multiarm/netmeta_portfolio_multiarm_arms.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": f"absolute <= {tolerance:g} for validated components",
        "skip_reason": "",
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [
            "This local fixture validates multi-arm covariance handling against netmeta outputs; it is not clinical evidence."
        ],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_toml_report(report), encoding="utf-8", newline="\n")


def _toml_report(report: dict[str, Any]) -> str:
    scalar_keys = [
        "schema_version",
        "target_id",
        "adapter_id",
        "reference_method",
        "status",
        "certification_effect",
        "checked_at",
        "command",
        "executable",
        "executable_found",
        "package_versions",
        "input_artifacts",
        "output_artifacts",
        "tolerance",
        "skip_reason",
        "validated_components",
        "max_abs_difference",
        "notes",
    ]
    lines = [_format_toml_key_value(key, report[key]) for key in scalar_keys]
    lines.extend(["", "[input_sha256]"])
    for key, value in report["input_sha256"].items():
        lines.append(f"{_quote(key)} = {_quote(value)}")
    lines.extend(["", "[output_sha256]"])
    for key, value in report["output_sha256"].items():
        lines.append(f"{_quote(key)} = {_quote(value)}")
    lines.append("")
    return "\n".join(lines)


def _format_toml_key_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {str(value).lower()}"
    if isinstance(value, float):
        return f"{key} = {value:.17g}"
    if isinstance(value, dict):
        inner = ", ".join(f"{_bare_key(item_key)} = {_quote(str(item_value))}" for item_key, item_value in value.items())
        return key + " = " + "{" + inner + "}"
    if isinstance(value, list):
        return f"{key} = [" + ", ".join(_quote(str(item)) for item in value) + "]"
    return f"{key} = {_quote(str(value))}"


def _bare_key(value: str) -> str:
    if value.replace("_", "").isalnum():
        return value
    return _quote(value)


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    raise SystemExit(main())
