"""Write the source-verified cross-design RCT/NRS benchmark artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.cross_design_benchmark import run_cross_design_benchmark


DEFAULT_MANIFEST = Path("validation/cross_design/sglt2_rct_nrs_cross_design.toml")
DEFAULT_SOURCE_CHECK = Path("validation/source_checks/sglt2_rct_nrs_cross_design_check.json")
DEFAULT_OUTPUT = Path("validation/cross_design/sglt2_rct_nrs_cross_design_benchmark.toml")


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(format_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key} = {format_toml_value(item)}" for key, item in sorted(value.items())
        ) + "}"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def append_table(lines: list[str], header: str, payload: dict[str, Any]) -> None:
    lines.append("")
    lines.append(f"[{header}]")
    for key, value in payload.items():
        lines.append(f"{key} = {format_toml_value(value)}")


def append_array_table(lines: list[str], header: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        lines.append("")
        lines.append(f"[[{header}]]")
        for key, value in row.items():
            lines.append(f"{key} = {format_toml_value(value)}")


def artifact_to_toml(artifact: dict[str, Any]) -> str:
    lines: list[str] = []
    top_keys = [
        "schema_version",
        "benchmark_id",
        "status",
        "certification_effect",
        "source_policy",
        "evidence_mode",
        "effect_scale",
        "cross_design_strategy",
        "source_manifest",
        "source_manifest_sha256",
        "source_verification_report",
        "source_verification_report_sha256",
        "n_studies",
        "limitations",
    ]
    for key in top_keys:
        lines.append(f"{key} = {format_toml_value(artifact[key])}")
    append_table(lines, "source_bundle", artifact["source_bundle"])
    append_table(lines, "model_config", artifact["model_config"])
    append_table(lines, "compatibility", artifact["compatibility"])
    append_array_table(lines, "study_effects", artifact["study_effects"])
    append_table(lines, "candidate.nrs_minus_rct_diagnostic", artifact["candidate"]["nrs_minus_rct_diagnostic"])
    for design, summary in artifact["candidate"]["separated_by_design"].items():
        append_table(lines, f"candidate.separated_by_design.{design}", summary)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-check", type=Path, default=DEFAULT_SOURCE_CHECK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = run_cross_design_benchmark(args.manifest, verification_report_path=args.source_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact_to_toml(artifact), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
