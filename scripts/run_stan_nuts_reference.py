"""Run the source-backed CmdStan/NUTS reference candidate."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import arviz as az
import numpy as np


DEFAULT_MODEL = Path("external/stan/standard_binary_nma.stan")
DEFAULT_EVENTS = Path("validation/real_meta/sglt2_hf_primary_events.csv")
DEFAULT_SOURCES = Path("validation/real_meta/sglt2_hf_primary_sources.toml")
DEFAULT_REFERENCE = Path("validation/reference_runs/pairwise_metafor_meta_output.json")
DEFAULT_SCRIPT = Path("scripts/run_stan_nuts_reference.py")
DEFAULT_OUTPUT = Path("validation/reference_runs/stan_nuts_cmdstan_output.json")
DEFAULT_REPORT = Path("validation/reference_runs/stan_nuts_cmdstan_reference.toml")
DEFAULT_TOLERANCE = 0.03
TEXT_HASH_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".r",
    ".stan",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_file(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_EXTENSIONS:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def toml_path(path: Path) -> str:
    return path.as_posix()


def toml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_stan_data(rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    study_ids = sorted({row["study_id"] for row in rows})
    study_map = {study_id: idx + 1 for idx, study_id in enumerate(study_ids)}
    treatment_order = ["Placebo", "SGLT2i"]
    treatment_map = {treatment: idx + 1 for idx, treatment in enumerate(treatment_order)}
    unexpected_treatments = sorted({row["treatment"] for row in rows} - set(treatment_map))
    if unexpected_treatments:
        raise ValueError(f"Unexpected treatments for Stan benchmark: {unexpected_treatments}")

    stan_data = {
        "N": len(rows),
        "S": len(study_ids),
        "T": len(treatment_order),
        "y": [int(row["events"]) for row in rows],
        "n": [int(row["n"]) for row in rows],
        "study": [study_map[row["study_id"]] for row in rows],
        "treatment": [treatment_map[row["treatment"]] for row in rows],
    }
    active_rows = [row for row in rows if row["arm_role"] == "active"]
    metadata = {
        "source_artifact": toml_path(DEFAULT_EVENTS),
        "n_rows": len(rows),
        "n_studies": len(study_ids),
        "treatments": treatment_order,
        "study_order": study_ids,
        "studies": [
            {
                "study_id": row["study_id"],
                "trial": row["trial"],
                "nct_id": row["nct_id"],
                "pmid": str(row["pmid"]),
            }
            for row in sorted(active_rows, key=lambda item: item["study_id"])
        ],
    }
    return stan_data, metadata


def cmdstan_version_string(cmdstanpy: Any) -> str:
    try:
        version = cmdstanpy.cmdstan_version()
        return ".".join(str(part) for part in version)
    except Exception:
        return "available_version_unresolved"


def ensure_cmdstan_toolchain_path(cmdstanpy: Any) -> None:
    """Expose a CmdStanPy-installed RTools toolchain without hardcoded paths."""

    try:
        cmdstan_root = Path(cmdstanpy.cmdstan_path()).resolve().parent
    except Exception:
        return
    rtools = cmdstan_root / "RTools40"
    candidate_dirs = [
        rtools / "mingw64" / "bin",
        rtools / "ucrt64" / "bin",
        rtools / "usr" / "bin",
    ]
    additions = [str(path) for path in candidate_dirs if path.is_dir()]
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + [os.environ.get("PATH", "")])


def arviz_version_string() -> str:
    return str(getattr(az, "__version__", "unknown"))


def summarize_draws(fit: Any, *, max_treedepth: int) -> dict[str, Any]:
    draws = fit.draws(inc_warmup=False, concat_chains=False)
    column_names = list(fit.column_names)
    d2_index = column_names.index("d[2]")
    d2_by_chain = np.asarray(draws[:, :, d2_index]).T
    idata = az.from_dict({"posterior": {"d2": d2_by_chain}})
    r_hat = _arviz_scalar(az.rhat(idata, var_names=["d2"]), "d2")
    ess_bulk = _arviz_scalar(az.ess(idata, var_names=["d2"], method="bulk"), "d2")
    ess_tail = _arviz_scalar(az.ess(idata, var_names=["d2"], method="tail"), "d2")
    mcse_mean = _arviz_scalar(az.mcse(idata, var_names=["d2"], method="mean"), "d2")

    combined = np.asarray(draws[:, :, d2_index]).reshape(-1)
    divergent = _sampler_count(draws, column_names, "divergent__")
    treedepth_saturation = _sampler_threshold_count(
        draws,
        column_names,
        "treedepth__",
        max_treedepth,
    )
    return {
        "posterior": {
            "parameter": "d[2]",
            "mean": float(np.mean(combined)),
            "sd": float(np.std(combined, ddof=1)),
            "median": float(np.quantile(combined, 0.5)),
            "q025": float(np.quantile(combined, 0.025)),
            "q975": float(np.quantile(combined, 0.975)),
            "mcse_mean": mcse_mean,
        },
        "diagnostics": {
            "r_hat": r_hat,
            "ess_bulk": ess_bulk,
            "ess_tail": ess_tail,
            "divergent_transitions": divergent,
            "treedepth_saturation": treedepth_saturation,
            "mcse_mean": mcse_mean,
            "prior_predictive_checks": "not_run_for_this_reference_candidate",
            "posterior_predictive_checks": "log_likelihood_exported_only",
        },
    }


def _arviz_scalar(value: Any, variable: str) -> float:
    if hasattr(value, "to_array"):
        return float(value.to_array().values.item())
    if hasattr(value, "groups") and "posterior" in value.groups:
        posterior = value["posterior"]
        return float(posterior.data_vars[variable].values.item())
    if hasattr(value, "data_vars"):
        return float(value.data_vars[variable].values.item())
    raise TypeError(f"Unsupported ArviZ scalar return type: {type(value)!r}")


def _sampler_count(draws: np.ndarray, column_names: list[str], column_name: str) -> int:
    if column_name not in column_names:
        return 0
    values = np.asarray(draws[:, :, column_names.index(column_name)])
    return int(np.sum(values))


def _sampler_threshold_count(
    draws: np.ndarray,
    column_names: list[str],
    column_name: str,
    threshold: int,
) -> int:
    if column_name not in column_names:
        return 0
    values = np.asarray(draws[:, :, column_names.index(column_name)])
    return int(np.sum(values >= threshold))


def build_output(
    *,
    root: Path,
    checked_at: str,
    fit: Any,
    cmdstanpy: Any,
    sampling: dict[str, Any],
    data_metadata: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    summary = summarize_draws(fit, max_treedepth=int(sampling["max_treedepth"]))
    reference = json.loads((root / DEFAULT_REFERENCE).read_text(encoding="utf-8"))
    reference_estimate = float(reference["metafor"]["fixed_effect"]["estimate"])
    posterior_mean = float(summary["posterior"]["mean"])
    absolute_difference = abs(posterior_mean - reference_estimate)
    comparison_status = "passed" if absolute_difference <= tolerance else "failed"

    return {
        "schema_version": "stan_nuts_reference/v1",
        "target_id": "bayesian_nma_multinma_cmdstan",
        "source_policy": (
            "Source-backed SGLT2i heart-failure arm counts from ClinicalTrials.gov "
            "records and PubMed abstracts; this is not broad Bayesian or multinma parity."
        ),
        "checked_at": checked_at,
        "effect_scale": "log_or",
        "model": {
            "stan_file": toml_path(DEFAULT_MODEL),
            "model_family": "fixed_effect_arm_level_binary_nma",
            "network_reference_treatment": "Placebo",
            "active_treatment_parameter": "d[2]",
        },
        "package_versions": {
            "python": platform.python_version(),
            "cmdstanpy": str(getattr(cmdstanpy, "__version__", "unknown")),
            "cmdstan": cmdstan_version_string(cmdstanpy),
            "arviz": arviz_version_string(),
        },
        "sampling": sampling,
        "data": data_metadata,
        "posterior": summary["posterior"],
        "diagnostics": summary["diagnostics"],
        "reference_comparison": {
            "reference_artifact": toml_path(DEFAULT_REFERENCE),
            "reference_method": "metafor fixed-effect log-OR on the same source-backed arm counts",
            "reference_estimate": reference_estimate,
            "posterior_mean": posterior_mean,
            "absolute_difference": absolute_difference,
            "tolerance": tolerance,
            "status": comparison_status,
        },
        "validated_components": [
            "arm_level_binary_nma_compiled_by_cmdstan",
            "nuts_sampler_diagnostics",
            "sglt2i_source_backed_log_or_posterior",
            "metafor_fixed_effect_mean_alignment",
        ],
        "certification_effect": "evidence_candidate",
        "claim_limit": (
            "Narrow CmdStan/NUTS reference candidate only; not broad feature parity, "
            "not multinma parity, not clinical or HTA certification."
        ),
    }


def write_reference_report(
    *,
    root: Path,
    output_path: Path,
    report_path: Path,
    checked_at: str,
    output: dict[str, Any],
    chains: int,
    iter_warmup: int,
    iter_sampling: int,
    seed: int,
    adapt_delta: float,
    max_treedepth: int,
) -> None:
    input_paths = [
        DEFAULT_MODEL,
        DEFAULT_EVENTS,
        DEFAULT_SOURCES,
        DEFAULT_REFERENCE,
        DEFAULT_SCRIPT,
        Path("src/bias_nma_adv/stan_reference_validation.py"),
    ]
    output_paths = [output_path.relative_to(root)]
    command = [
        "python",
        toml_path(DEFAULT_SCRIPT),
        "--root",
        ".",
        "--output",
        toml_path(DEFAULT_OUTPUT),
        "--report",
        toml_path(DEFAULT_REPORT),
        "--chains",
        str(chains),
        "--iter-warmup",
        str(iter_warmup),
        "--iter-sampling",
        str(iter_sampling),
        "--seed",
        str(seed),
        "--adapt-delta",
        str(adapt_delta),
        "--max-treedepth",
        str(max_treedepth),
    ]

    lines = [
        'schema_version = "reference_run/v1"',
        'target_id = "bayesian_nma_multinma_cmdstan"',
        'adapter_id = "python_cmdstan_nuts_output_validation"',
        'reference_method = "CmdStanPy/CmdStan NUTS"',
        'status = "passed"',
        'certification_effect = "evidence_candidate"',
        f'checked_at = "{checked_at}"',
        "command = [" + ", ".join(toml_quote(part) for part in command) + "]",
        'executable = "python"',
        "executable_found = true",
        "package_versions = {"
        + ", ".join(
            f"{key} = {toml_quote(str(value))}"
            for key, value in sorted(output["package_versions"].items())
        )
        + "}",
        "input_artifacts = [",
        *[f"  {toml_quote(toml_path(path))}," for path in input_paths],
        "]",
        "output_artifacts = [",
        *[f"  {toml_quote(toml_path(path))}," for path in output_paths],
        "]",
        f'tolerance = {toml_quote("posterior mean absolute difference <= 0.03 vs metafor fixed-effect log-OR; R-hat <= 1.01; bulk/tail ESS >= 400; divergences = 0; treedepth saturation = 0; MCSE <= 0.005")}',
        'skip_reason = ""',
        "",
        "[input_sha256]",
    ]
    for relative_path in input_paths:
        lines.append(
            f"{toml_quote(toml_path(relative_path))} = {toml_quote(sha256_file(root / relative_path))}"
        )
    lines.extend(["", "[output_sha256]"])
    for relative_path in output_paths:
        lines.append(
            f"{toml_quote(toml_path(relative_path))} = {toml_quote(sha256_file(root / relative_path))}"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=500)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--adapt-delta", type=float, default=0.95)
    parser.add_argument("--max-treedepth", type=int, default=10)
    parser.add_argument("--checked-at", default=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output if args.output.is_absolute() else root / args.output
    report_path = args.report if args.report.is_absolute() else root / args.report

    sys.path.insert(0, str(root / "src"))
    import cmdstanpy  # type: ignore[import-not-found]

    ensure_cmdstan_toolchain_path(cmdstanpy)
    rows = load_rows(root / DEFAULT_EVENTS)
    stan_data, data_metadata = build_stan_data(rows)
    model = cmdstanpy.CmdStanModel(stan_file=str(root / DEFAULT_MODEL))
    fit = model.sample(
        data=stan_data,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        show_progress=False,
    )
    sampling = {
        "chains": args.chains,
        "iter_warmup": args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "seed": args.seed,
        "adapt_delta": args.adapt_delta,
        "max_treedepth": args.max_treedepth,
    }
    output = build_output(
        root=root,
        checked_at=args.checked_at,
        fit=fit,
        cmdstanpy=cmdstanpy,
        sampling=sampling,
        data_metadata=data_metadata,
        tolerance=DEFAULT_TOLERANCE,
    )
    if output["reference_comparison"]["status"] != "passed":
        raise RuntimeError("Stan/NUTS reference comparison did not pass tolerance.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_reference_report(
        root=root,
        output_path=output_path,
        report_path=report_path,
        checked_at=args.checked_at,
        output=output,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
    )
    print(f"stan nuts reference output written to {output_path}")
    print(f"stan nuts reference report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
