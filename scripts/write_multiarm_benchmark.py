"""Write the governed multi-arm NMA fixture benchmark artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.multiarm import ContrastRow, fit_multiarm_gls


DEFAULT_ARMS = Path("validation/multiarm/netmeta_portfolio_multiarm_arms.csv")
DEFAULT_OUTPUT = Path("validation/multiarm/netmeta_portfolio_multiarm_benchmark.toml")


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, list | tuple):
        return "[" + ", ".join(format_toml_value(item) for item in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_arm_rows(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"fixture_id", "study", "treatment", "events", "n"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"arm fixture missing columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            row = {
                "fixture_id": str(raw["fixture_id"]).strip(),
                "study": str(raw["study"]).strip(),
                "treatment": str(raw["treatment"]).strip(),
                "events": int(raw["events"]),
                "n": int(raw["n"]),
            }
            if not row["fixture_id"] or not row["study"] or not row["treatment"]:
                raise ValueError(f"arm fixture contains empty identifiers: {row}")
            if not 0 < row["events"] < row["n"]:
                raise ValueError(f"arm fixture row must have 0 < events < n: {row}")
            rows.append(row)
    if not rows:
        raise ValueError("arm fixture is empty.")
    return tuple(rows)


def log_odds(events: int, n: int) -> float:
    return math.log(events / (n - events))


def contrast_rows_from_arms(arms: tuple[dict[str, Any], ...]) -> tuple[ContrastRow, ...]:
    by_study: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        by_study.setdefault(str(arm["study"]), []).append(arm)

    rows: list[ContrastRow] = []
    for study, study_arms in by_study.items():
        for i in range(len(study_arms)):
            for j in range(i + 1, len(study_arms)):
                arm_1 = study_arms[i]
                arm_2 = study_arms[j]
                e1 = int(arm_1["events"])
                n1 = int(arm_1["n"])
                e2 = int(arm_2["events"])
                n2 = int(arm_2["n"])
                est = log_odds(e2, n2) - log_odds(e1, n1)
                se = math.sqrt(1 / e1 + 1 / (n1 - e1) + 1 / e2 + 1 / (n2 - e2))
                rows.append(
                    ContrastRow(
                        study=study,
                        t1=str(arm_1["treatment"]),
                        t2=str(arm_2["treatment"]),
                        est=est,
                        se=se,
                    )
                )
    return tuple(rows)


def build_artifact(arms_path: Path) -> dict[str, Any]:
    arm_rows = load_arm_rows(arms_path)
    fixture_ids = tuple(dict.fromkeys(str(row["fixture_id"]) for row in arm_rows))

    fixtures: list[dict[str, Any]] = []
    model_fits: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for fixture_id in fixture_ids:
        fixture_arms = tuple(row for row in arm_rows if row["fixture_id"] == fixture_id)
        studies = tuple(dict.fromkeys(str(row["study"]) for row in fixture_arms))
        treatments = tuple(sorted({str(row["treatment"]) for row in fixture_arms}))
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "n_arm_rows": len(fixture_arms),
                "n_studies": len(studies),
                "treatments": list(treatments),
                "description": (
                    "consistent multi-arm covariance fixture"
                    if fixture_id == "consistent"
                    else "heterogeneous multi-arm covariance fixture"
                ),
            }
        )

        contrast_rows = contrast_rows_from_arms(fixture_arms)
        models = ("fixed",) if fixture_id == "consistent" else ("fixed", "random")
        for model in models:
            fit = fit_multiarm_gls(
                contrast_rows,
                reference_treatment="A",
                model=model,
            )
            fit_row = {
                "fixture_id": fixture_id,
                "model": fit.model,
                "reference_treatment": fit.reference_treatment,
                "treatments": list(fit.treatments),
                "nonreference_treatments": list(fit.nonreference_treatments),
                "multi_arm_studies": list(fit.multi_arm_studies),
                "warnings": list(fit.warnings),
                "tau2": float(fit.tau2),
            }
            if fit.q is not None:
                fit_row["q"] = float(fit.q)
            if fit.df is not None:
                fit_row["df"] = int(fit.df)
            model_fits.append(fit_row)

            for treatment in fit.nonreference_treatments:
                estimate, se = fit.effect_vs_reference(treatment)
                effects.append(
                    {
                        "fixture_id": fixture_id,
                        "model": fit.model,
                        "treatment": treatment,
                        "reference_treatment": fit.reference_treatment,
                        "estimate": float(estimate),
                        "se": float(se),
                    }
                )

    return {
        "schema_version": "multiarm_benchmark/v1",
        "benchmark_id": "netmeta_portfolio_multiarm_fixture",
        "status": "local_fixture_recomputed",
        "certification_effect": "none",
        "source_policy": "portfolio_algorithmic_fixture_not_clinical_evidence",
        "evidence_mode": "algorithmic_fixture",
        "effect_scale": "log_or",
        "reference_method": "netmeta",
        "reference_target_id": "multiarm_gls_netmeta_portfolio_fixture",
        "reference_status": "external_netmeta_preflight_required_before_reference_matching",
        "arm_data": repo_path(arms_path),
        "arm_data_sha256": sha256_file(arms_path),
        "tolerance": "absolute <= 1e-6 for deterministic fixture replay",
        "limitations": [
            "The fixture is algorithmic and is not clinical evidence.",
            "The values are local deterministic replay outputs and do not certify netmeta parity by themselves.",
            "Reference matching requires a passed external netmeta adapter report with package versions and output hashes.",
        ],
        "fixtures": fixtures,
        "model_fits": model_fits,
        "effects": effects,
    }


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
        "reference_method",
        "reference_target_id",
        "reference_status",
        "arm_data",
        "arm_data_sha256",
        "tolerance",
        "limitations",
    ]
    for key in top_keys:
        lines.append(f"{key} = {format_toml_value(artifact[key])}")

    for fixture in artifact["fixtures"]:
        lines.append("")
        lines.append("[[fixtures]]")
        for key, value in fixture.items():
            lines.append(f"{key} = {format_toml_value(value)}")

    for fit in artifact["model_fits"]:
        lines.append("")
        lines.append("[[model_fits]]")
        for key, value in fit.items():
            lines.append(f"{key} = {format_toml_value(value)}")

    for effect in artifact["effects"]:
        lines.append("")
        lines.append("[[effects]]")
        for key, value in effect.items():
            lines.append(f"{key} = {format_toml_value(value)}")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arms_path = args.arms.resolve()
    artifact = build_artifact(arms_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact_to_toml(artifact), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
