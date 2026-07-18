"""Validation for CmdStan/NUTS reference-run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


STAN_NUTS_REFERENCE_SCHEMA_VERSION = "stan_nuts_reference/v1"


class StanReferenceValidationError(ValueError):
    """Raised when a Stan/NUTS reference artifact is malformed or drifting."""


def load_stan_reference_output(path: str | Path) -> dict[str, Any]:
    """Load one Stan/NUTS reference output JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_stan_nuts_reference_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 0.03,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    mcse_threshold: float = 0.005,
) -> dict[str, Any]:
    """Validate the source-backed CmdStan/NUTS output artifact.

    This is a narrow artifact check for one arm-level binary NMA model. It
    must not be interpreted as broad ``multinma`` or Bayesian feature parity.
    """

    root = Path(repo_root)
    output = load_stan_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "target_id",
            "source_policy",
            "effect_scale",
            "package_versions",
            "sampling",
            "prior_sampling",
            "data",
            "posterior",
            "diagnostics",
            "predictive_checks",
            "reference_comparison",
            "validated_components",
            "certification_effect",
            "claim_limit",
        },
        label="Stan/NUTS output",
    )
    if output["schema_version"] != STAN_NUTS_REFERENCE_SCHEMA_VERSION:
        raise StanReferenceValidationError("Stan/NUTS output schema_version mismatch.")
    if output["target_id"] != "bayesian_nma_multinma_cmdstan":
        raise StanReferenceValidationError("Stan/NUTS output target_id mismatch.")
    if output["effect_scale"] != "log_or":
        raise StanReferenceValidationError("Stan/NUTS output must use log_or scale.")
    if output["certification_effect"] != "evidence_candidate":
        raise StanReferenceValidationError("Stan/NUTS output must remain an evidence candidate.")
    if "not broad" not in str(output["claim_limit"]).lower():
        raise StanReferenceValidationError("Stan/NUTS output claim_limit must block broad parity.")
    if "source-backed" not in str(output["source_policy"]).lower():
        raise StanReferenceValidationError("Stan/NUTS output must declare source-backed policy.")

    _require_package_versions(output["package_versions"], {"cmdstanpy", "cmdstan", "arviz"})
    _validate_source_rows(output["data"], root)
    _validate_sampling(output["sampling"])
    _validate_prior_sampling(output["prior_sampling"])
    _validate_diagnostics(
        output["diagnostics"],
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        mcse_threshold=mcse_threshold,
    )
    _validate_predictive_checks(output["predictive_checks"], expected_n_rows=int(output["data"]["n_rows"]))
    max_abs_difference = _validate_reference_comparison(
        output["reference_comparison"],
        root=root,
        tolerance=tolerance,
    )

    required_components = {
        "arm_level_binary_nma_compiled_by_cmdstan",
        "nuts_sampler_diagnostics",
        "sglt2i_source_backed_log_or_posterior",
        "metafor_fixed_effect_mean_alignment",
        "stan_prior_predictive_check_declared_priors",
        "stan_posterior_predictive_check_y_rep",
    }
    components = set(str(item) for item in output["validated_components"])
    missing_components = sorted(required_components - components)
    if missing_components:
        raise StanReferenceValidationError(
            f"Stan/NUTS output missing validated components: {missing_components}"
        )

    return {
        "schema_version": "stan_reference_validation/v1",
        "target_id": "bayesian_nma_multinma_cmdstan",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "CmdStanPy/CmdStan NUTS",
        "validated_components": sorted(required_components),
        "max_abs_difference": max_abs_difference,
        "tolerance": tolerance,
        "r_hat": float(output["diagnostics"]["r_hat"]),
        "ess_bulk": float(output["diagnostics"]["ess_bulk"]),
        "ess_tail": float(output["diagnostics"]["ess_tail"]),
        "mcse_mean": float(output["posterior"]["mcse_mean"]),
        "prior_predictive_tail_area": float(
            output["predictive_checks"]["prior_predictive"]["tail_area_total_events_two_sided"]
        ),
        "posterior_predictive_tail_area": float(
            output["predictive_checks"]["posterior_predictive"]["tail_area_total_events_two_sided"]
        ),
        "claim_limit": output["claim_limit"],
    }


def _validate_source_rows(data: dict[str, Any], root: Path) -> None:
    _require_keys(
        data,
        {"source_artifact", "n_rows", "n_studies", "treatments", "studies"},
        label="Stan/NUTS data",
    )
    source_path = root / str(data["source_artifact"])
    if not source_path.is_file():
        raise StanReferenceValidationError(f"Stan/NUTS source artifact missing: {data['source_artifact']}")
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if int(data["n_rows"]) != len(rows):
        raise StanReferenceValidationError("Stan/NUTS n_rows mismatch.")
    studies = sorted({row["study_id"] for row in rows})
    if int(data["n_studies"]) != len(studies):
        raise StanReferenceValidationError("Stan/NUTS n_studies mismatch.")
    observed_treatments = list(data["treatments"])
    if observed_treatments != ["Placebo", "SGLT2i"]:
        raise StanReferenceValidationError("Stan/NUTS treatment coding changed.")
    source_ids = {
        row["study_id"]: (row["nct_id"], str(row["pmid"]))
        for row in rows
        if row["arm_role"] == "active"
    }
    output_ids = {
        str(item["study_id"]): (str(item["nct_id"]), str(item["pmid"]))
        for item in data["studies"]
    }
    if source_ids != output_ids:
        raise StanReferenceValidationError("Stan/NUTS study source identifiers mismatch.")


def _validate_sampling(sampling: dict[str, Any]) -> None:
    _require_keys(
        sampling,
        {"chains", "iter_warmup", "iter_sampling", "seed", "adapt_delta", "max_treedepth"},
        label="Stan/NUTS sampling",
    )
    if int(sampling["chains"]) < 4:
        raise StanReferenceValidationError("Stan/NUTS reference run requires at least four chains.")
    if int(sampling["iter_sampling"]) < 500:
        raise StanReferenceValidationError("Stan/NUTS reference run requires at least 500 post-warmup draws per chain.")


def _validate_prior_sampling(sampling: dict[str, Any]) -> None:
    _require_keys(
        sampling,
        {
            "chains",
            "iter_warmup",
            "iter_sampling",
            "seed",
            "adapt_delta",
            "max_treedepth",
            "prior_only",
        },
        label="Stan/NUTS prior sampling",
    )
    if sampling["prior_only"] is not True:
        raise StanReferenceValidationError("Stan/NUTS prior predictive sampling must run in prior_only mode.")
    if int(sampling["chains"]) < 4:
        raise StanReferenceValidationError("Stan/NUTS prior predictive run requires at least four chains.")
    if int(sampling["iter_sampling"]) < 250:
        raise StanReferenceValidationError("Stan/NUTS prior predictive run requires at least 250 draws per chain.")


def _validate_diagnostics(
    diagnostics: dict[str, Any],
    *,
    rhat_threshold: float,
    ess_threshold: float,
    mcse_threshold: float,
) -> None:
    _require_keys(
        diagnostics,
        {
            "r_hat",
            "ess_bulk",
            "ess_tail",
            "divergent_transitions",
            "treedepth_saturation",
        },
        label="Stan/NUTS diagnostics",
    )
    r_hat = float(diagnostics["r_hat"])
    ess_bulk = float(diagnostics["ess_bulk"])
    ess_tail = float(diagnostics["ess_tail"])
    if r_hat > rhat_threshold:
        raise StanReferenceValidationError(f"Stan/NUTS r_hat {r_hat:.4g} exceeds {rhat_threshold:.4g}.")
    if ess_bulk < ess_threshold or ess_tail < ess_threshold:
        raise StanReferenceValidationError("Stan/NUTS ESS fell below threshold.")
    if int(diagnostics["divergent_transitions"]) != 0:
        raise StanReferenceValidationError("Stan/NUTS reference run has divergent transitions.")
    if int(diagnostics["treedepth_saturation"]) != 0:
        raise StanReferenceValidationError("Stan/NUTS reference run has treedepth saturation.")
    if "mcse_mean" in diagnostics and float(diagnostics["mcse_mean"]) > mcse_threshold:
        raise StanReferenceValidationError("Stan/NUTS MCSE exceeds threshold.")
    if diagnostics.get("prior_predictive_checks") != "cmdstan_prior_only_declared_model_summary":
        raise StanReferenceValidationError("Stan/NUTS prior predictive diagnostic summary is missing.")
    if diagnostics.get("posterior_predictive_checks") != "cmdstan_posterior_y_rep_summary":
        raise StanReferenceValidationError("Stan/NUTS posterior predictive diagnostic summary is missing.")


def _validate_predictive_checks(raw: dict[str, Any], *, expected_n_rows: int) -> None:
    _require_keys(
        raw,
        {"prior_predictive", "posterior_predictive"},
        label="Stan/NUTS predictive checks",
    )
    _validate_predictive_check(
        raw["prior_predictive"],
        expected_type="prior_predictive",
        expected_method="same_stan_model_prior_only_mode_declared_priors",
        expected_n_rows=expected_n_rows,
    )
    _validate_predictive_check(
        raw["posterior_predictive"],
        expected_type="posterior_predictive",
        expected_method="same_stan_model_posterior_generated_quantities_y_rep",
        expected_n_rows=expected_n_rows,
    )


def _validate_predictive_check(
    raw: dict[str, Any],
    *,
    expected_type: str,
    expected_method: str,
    expected_n_rows: int,
) -> None:
    _require_keys(
        raw,
        {
            "check_type",
            "method",
            "n_draws",
            "n_arms",
            "observed_total_events",
            "observed_total_n",
            "replicated_total_events_mean",
            "replicated_total_events_sd",
            "replicated_total_events_q025",
            "replicated_total_events_q975",
            "observed_total_within_95_interval",
            "tail_area_total_events_two_sided",
            "row_level_95_interval_coverage_fraction",
        },
        label=f"Stan/NUTS {expected_type}",
    )
    if raw["check_type"] != expected_type:
        raise StanReferenceValidationError(f"Stan/NUTS predictive check type must be {expected_type}.")
    if raw["method"] != expected_method:
        raise StanReferenceValidationError(f"Stan/NUTS {expected_type} method changed unexpectedly.")
    if int(raw["n_arms"]) != expected_n_rows:
        raise StanReferenceValidationError(f"Stan/NUTS {expected_type} n_arms does not match source data.")
    if int(raw["n_draws"]) < 1000:
        raise StanReferenceValidationError(f"Stan/NUTS {expected_type} has too few replicated draws.")
    q025 = float(raw["replicated_total_events_q025"])
    q975 = float(raw["replicated_total_events_q975"])
    mean = float(raw["replicated_total_events_mean"])
    tail_area = float(raw["tail_area_total_events_two_sided"])
    coverage = float(raw["row_level_95_interval_coverage_fraction"])
    if q025 > mean or mean > q975:
        raise StanReferenceValidationError(f"Stan/NUTS {expected_type} predictive quantiles are inconsistent.")
    if not 0.0 <= tail_area <= 1.0:
        raise StanReferenceValidationError(f"Stan/NUTS {expected_type} tail area is outside [0, 1].")
    if not 0.0 <= coverage <= 1.0:
        raise StanReferenceValidationError(f"Stan/NUTS {expected_type} row coverage is outside [0, 1].")


def _validate_reference_comparison(
    comparison: dict[str, Any],
    *,
    root: Path,
    tolerance: float,
) -> float:
    _require_keys(
        comparison,
        {
            "reference_artifact",
            "reference_method",
            "reference_estimate",
            "posterior_mean",
            "absolute_difference",
            "tolerance",
            "status",
        },
        label="Stan/NUTS reference comparison",
    )
    if comparison["status"] != "passed":
        raise StanReferenceValidationError("Stan/NUTS reference comparison did not pass.")
    reference_path = root / str(comparison["reference_artifact"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = float(reference["metafor"]["fixed_effect"]["estimate"])
    observed_reference = float(comparison["reference_estimate"])
    if abs(observed_reference - expected) > 1e-12:
        raise StanReferenceValidationError("Stan/NUTS reference estimate drifted from metafor JSON.")
    difference = abs(float(comparison["posterior_mean"]) - expected)
    reported_difference = float(comparison["absolute_difference"])
    if abs(difference - reported_difference) > 1e-12:
        raise StanReferenceValidationError("Stan/NUTS absolute difference is inconsistent.")
    if float(comparison["tolerance"]) != tolerance:
        raise StanReferenceValidationError("Stan/NUTS tolerance changed unexpectedly.")
    if difference > tolerance:
        raise StanReferenceValidationError(
            f"Stan/NUTS posterior mean differs by {difference:.6g}, exceeding tolerance {tolerance:.6g}."
        )
    return difference


def _require_keys(raw: dict[str, Any], required: set[str], *, label: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        raise StanReferenceValidationError(f"{label} missing required keys: {missing}")


def _require_package_versions(raw: Any, required: set[str]) -> None:
    if not isinstance(raw, dict):
        raise StanReferenceValidationError("Stan/NUTS package_versions must be an object.")
    missing = sorted(required - set(raw))
    if missing:
        raise StanReferenceValidationError(f"Stan/NUTS missing package versions: {missing}")
    for package_name in required:
        if not str(raw[package_name]).strip():
            raise StanReferenceValidationError(f"Stan/NUTS package version for {package_name} must not be empty.")
