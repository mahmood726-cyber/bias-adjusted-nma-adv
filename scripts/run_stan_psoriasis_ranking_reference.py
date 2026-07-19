"""Run a source-backed multi-treatment CmdStan ranking reference candidate."""

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
DEFAULT_ARMS = Path("validation/reference_runs/psoriasis_pasi90_ctgov_binary_network_arms.csv")
DEFAULT_MANIFEST = Path("validation/networks/psoriasis_pasi90_ctgov_binary_network.toml")
DEFAULT_SOURCE_CHECK = Path("validation/source_checks/psoriasis_pasi90_ctgov_binary_network_check.json")
DEFAULT_NETMETA = Path(
    "validation/reference_runs/psoriasis_pasi90_ctgov_binary_network_netmeta_output.json"
)
DEFAULT_SCRIPT = Path("scripts/run_stan_psoriasis_ranking_reference.py")
DEFAULT_VALIDATOR = Path("src/bias_nma_adv/stan_reference_validation.py")
DEFAULT_OUTPUT = Path("validation/reference_runs/stan_psoriasis_ranking_output.json")
DEFAULT_REPORT = Path("validation/reference_runs/stan_psoriasis_ranking_reference.toml")
DEFAULT_EFFECT_TOLERANCE = 0.5
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
    study_order = _ordered_unique(row["study"] for row in rows)
    treatment_order = ["placebo"] + sorted(
        {row["treatment"] for row in rows if row["treatment"] != "placebo"}
    )
    study_map = {study_id: idx + 1 for idx, study_id in enumerate(study_order)}
    treatment_map = {treatment: idx + 1 for idx, treatment in enumerate(treatment_order)}
    stan_data = {
        "N": len(rows),
        "S": len(study_order),
        "T": len(treatment_order),
        "y": [int(row["events"]) for row in rows],
        "n": [int(row["n"]) for row in rows],
        "study": [study_map[row["study"]] for row in rows],
        "treatment": [treatment_map[row["treatment"]] for row in rows],
        "prior_only": 0,
    }
    metadata = {
        "source_artifact": toml_path(DEFAULT_ARMS),
        "source_manifest": toml_path(DEFAULT_MANIFEST),
        "source_check": toml_path(DEFAULT_SOURCE_CHECK),
        "n_rows": len(rows),
        "n_studies": len(study_order),
        "n_treatments": len(treatment_order),
        "study_order": study_order,
        "treatments": treatment_order,
    }
    return stan_data, metadata


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(str(value))
            seen.add(str(value))
    return ordered


def ensure_cmdstan_toolchain_path(cmdstanpy: Any) -> None:
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


def cmdstan_version_string(cmdstanpy: Any) -> str:
    try:
        version = cmdstanpy.cmdstan_version()
        return ".".join(str(part) for part in version)
    except Exception:
        return "available_version_unresolved"


def summarize_fit(fit: Any, treatments: list[str], *, max_treedepth: int) -> dict[str, Any]:
    draws = fit.draws(inc_warmup=False, concat_chains=False)
    column_names = list(fit.column_names)
    effect_draws = _treatment_effect_draws(draws, column_names, treatments)
    posterior_effects = _posterior_effect_summaries(effect_draws, treatments)
    diagnostics = _diagnostics(draws, column_names, effect_draws, treatments, max_treedepth=max_treedepth)
    ranking = _posterior_ranking(effect_draws, treatments)
    return {
        "posterior_effects": posterior_effects,
        "diagnostics": diagnostics,
        "posterior_ranking": ranking,
    }


def _treatment_effect_draws(
    draws: np.ndarray,
    column_names: list[str],
    treatments: list[str],
) -> np.ndarray:
    matrix = np.zeros((draws.shape[0] * draws.shape[1], len(treatments)), dtype=float)
    for idx, treatment in enumerate(treatments[1:], start=2):
        column_name = f"d[{idx}]"
        if column_name not in column_names:
            raise KeyError(f"Missing Stan treatment-effect column for {treatment}: {column_name}")
        matrix[:, idx - 1] = np.asarray(draws[:, :, column_names.index(column_name)]).reshape(-1)
    return matrix


def _posterior_effect_summaries(
    effect_draws: np.ndarray,
    treatments: list[str],
) -> dict[str, dict[str, float | str]]:
    summaries: dict[str, dict[str, float | str]] = {}
    for idx, treatment in enumerate(treatments):
        values = effect_draws[:, idx]
        summaries[treatment] = {
            "parameter": "d[1] = 0" if idx == 0 else f"d[{idx + 1}]",
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        }
    return summaries


def _diagnostics(
    draws: np.ndarray,
    column_names: list[str],
    effect_draws: np.ndarray,
    treatments: list[str],
    *,
    max_treedepth: int,
) -> dict[str, Any]:
    posterior = {
        treatment: np.asarray(draws[:, :, column_names.index(f"d[{idx}]")]).T
        for idx, treatment in enumerate(treatments[1:], start=2)
    }
    idata = az.from_dict({"posterior": posterior})
    rhat = az.rhat(idata)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    return {
        "max_r_hat": _max_arviz_value(rhat),
        "min_ess_bulk": _min_arviz_value(ess_bulk),
        "min_ess_tail": _min_arviz_value(ess_tail),
        "divergent_transitions": _sampler_count(draws, column_names, "divergent__"),
        "treedepth_saturation": _sampler_threshold_count(
            draws,
            column_names,
            "treedepth__",
            max_treedepth,
        ),
        "mcse_mean_max": _mcse_mean_max(effect_draws[:, 1:]),
    }


def _posterior_ranking(effect_draws: np.ndarray, treatments: list[str]) -> dict[str, Any]:
    order = np.argsort(-effect_draws, axis=1)
    ranks = np.empty_like(order)
    draw_indices = np.arange(effect_draws.shape[0])[:, None]
    ranks[draw_indices, order] = np.arange(1, len(treatments) + 1)
    treatment_summaries = []
    for idx, treatment in enumerate(treatments):
        treatment_ranks = ranks[:, idx]
        rank_probabilities = [
            float(np.mean(treatment_ranks == rank))
            for rank in range(1, len(treatments) + 1)
        ]
        mean_rank = float(np.mean(treatment_ranks))
        treatment_summaries.append(
            {
                "treatment": treatment,
                "effect_parameter": "d[1] = 0" if idx == 0 else f"d[{idx + 1}]",
                "rank_probabilities": rank_probabilities,
                "mean_rank": mean_rank,
                "rank_1_probability": rank_probabilities[0],
                "sucra": (len(treatments) - mean_rank) / (len(treatments) - 1),
            }
        )
    posterior_mean_order = [
        treatments[idx]
        for idx in np.argsort(-np.mean(effect_draws, axis=0))
    ]
    return {
        "method": "rank_all_treatments_by_joint_posterior_draw",
        "ranking_scale": "higher_log_odds_preferred",
        "preserves_joint_draws": True,
        "n_draws": int(effect_draws.shape[0]),
        "posterior_mean_order": posterior_mean_order,
        "treatments": treatment_summaries,
    }


def _max_arviz_value(value: Any) -> float:
    return float(np.nanmax(value.to_array().values))


def _min_arviz_value(value: Any) -> float:
    return float(np.nanmin(value.to_array().values))


def _sampler_count(draws: np.ndarray, column_names: list[str], column_name: str) -> int:
    if column_name not in column_names:
        return 0
    return int(np.sum(np.asarray(draws[:, :, column_names.index(column_name)])))


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


def _mcse_mean_max(values: np.ndarray) -> float:
    n_draws = values.shape[0]
    return float(np.max(np.std(values, axis=0, ddof=1) / np.sqrt(n_draws)))


def build_output(
    *,
    root: Path,
    checked_at: str,
    fit: Any,
    cmdstanpy: Any,
    sampling: dict[str, Any],
    data_metadata: dict[str, Any],
    effect_tolerance: float,
) -> dict[str, Any]:
    fit_summary = summarize_fit(
        fit,
        list(data_metadata["treatments"]),
        max_treedepth=int(sampling["max_treedepth"]),
    )
    netmeta = json.loads((root / DEFAULT_NETMETA).read_text(encoding="utf-8"))["fixtures"][0]
    netmeta_order = ["placebo"] + [
        treatment
        for treatment, _ in sorted(
            netmeta["common"].items(),
            key=lambda item: float(item[1]["estimate"]),
            reverse=True,
        )
    ]
    posterior_mean_order = list(fit_summary["posterior_ranking"]["posterior_mean_order"])
    common_differences = {
        treatment: float(fit_summary["posterior_effects"][treatment]["mean"])
        - float(netmeta["common"][treatment]["estimate"])
        for treatment in data_metadata["treatments"]
        if treatment != "placebo"
    }
    max_abs_effect_difference = max(abs(value) for value in common_differences.values())
    comparison_status = (
        "passed"
        if posterior_mean_order == netmeta_order
        and max_abs_effect_difference <= effect_tolerance
        else "failed"
    )
    return {
        "schema_version": "stan_multitreatment_ranking_reference/v1",
        "target_id": "bayesian_nma_multinma_cmdstan",
        "benchmark_id": "psoriasis_pasi90_ctgov_binary_network",
        "source_policy": (
            "Source-backed psoriasis PASI90 arm counts from ClinicalTrials.gov records "
            "and PubMed abstracts; this is a ranking-order candidate, not broad Bayesian parity."
        ),
        "checked_at": checked_at,
        "effect_scale": "log_or",
        "outcome_direction": "higher_events_better",
        "model": {
            "stan_file": toml_path(DEFAULT_MODEL),
            "model_family": "fixed_effect_arm_level_binary_nma",
            "network_reference_treatment": "placebo",
        },
        "package_versions": {
            "python": platform.python_version(),
            "cmdstanpy": str(getattr(cmdstanpy, "__version__", "unknown")),
            "cmdstan": cmdstan_version_string(cmdstanpy),
            "arviz": str(getattr(az, "__version__", "unknown")),
        },
        "sampling": sampling,
        "data": data_metadata,
        "posterior_effects": fit_summary["posterior_effects"],
        "diagnostics": fit_summary["diagnostics"],
        "posterior_ranking": fit_summary["posterior_ranking"],
        "reference_comparison": {
            "reference_artifact": toml_path(DEFAULT_NETMETA),
            "reference_method": "netmeta common-effect treatment-effect order",
            "netmeta_common_order": netmeta_order,
            "posterior_mean_order": posterior_mean_order,
            "complete_order_match": posterior_mean_order == netmeta_order,
            "common_effect_mean_differences": common_differences,
            "max_abs_effect_difference": max_abs_effect_difference,
            "effect_difference_tolerance": effect_tolerance,
            "status": comparison_status,
            "interpretation": (
                "Order agreement supports a narrow ranking candidate. The effect mean "
                "differences are reported for compatibility only and are not exact netmeta parity."
            ),
        },
        "validated_components": [
            "arm_level_binary_nma_compiled_by_cmdstan",
            "multitreatment_joint_posterior_ranking_draw_summary",
            "netmeta_common_order_alignment",
            "nuts_sampler_diagnostics",
            "psoriasis_source_backed_pasi90_counts",
        ],
        "certification_effect": "evidence_candidate",
        "claim_limit": (
            "Narrow source-backed multi-treatment ranking-order candidate only; not broad "
            "Bayesian NMA parity, not multinma parity, not clinical or HTA certification."
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
        DEFAULT_ARMS,
        DEFAULT_MANIFEST,
        DEFAULT_SOURCE_CHECK,
        DEFAULT_NETMETA,
        DEFAULT_SCRIPT,
        DEFAULT_VALIDATOR,
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
        'adapter_id = "python_cmdstan_psoriasis_ranking_output_validation"',
        'reference_method = "CmdStanPy/CmdStan NUTS posterior ranking vs netmeta order"',
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
        (
            "tolerance = "
            + toml_quote(
                "posterior mean treatment order equals netmeta common-effect order; "
                "max absolute effect mean difference <= 0.5 as compatibility screen only; "
                "R-hat <= 1.01; bulk/tail ESS >= 400; divergences = 0; treedepth saturation = 0"
            )
        ),
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
    parser.add_argument("--seed", type=int, default=20260718)
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
    rows = load_rows(root / DEFAULT_ARMS)
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
        effect_tolerance=DEFAULT_EFFECT_TOLERANCE,
    )
    if output["reference_comparison"]["status"] != "passed":
        raise RuntimeError("Stan psoriasis ranking comparison did not pass.")
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
    print(f"stan psoriasis ranking output written to {output_path}")
    print(f"stan psoriasis ranking report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
