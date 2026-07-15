"""Validation for optional R reference-adapter output artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any


class RReferenceValidationError(ValueError):
    """Raised when an R reference output is malformed or drifts from the local artifact."""


def load_r_reference_output(path: str | Path) -> dict[str, Any]:
    """Load one R reference output JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_pairwise_metafor_meta_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate the pairwise ``metafor`` output against the source-backed artifact."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {"schema_version", "effect_scale", "package_versions", "study_effects", "metafor"},
        label="pairwise R output",
    )
    if output["schema_version"] != "pairwise_metafor_meta/v1":
        raise RReferenceValidationError("pairwise R output schema_version mismatch.")
    if output["effect_scale"] != "log_or":
        raise RReferenceValidationError("pairwise R output must use log_or scale.")
    _require_package_versions(output["package_versions"], {"R", "metafor", "meta"})

    benchmark = _load_toml(root / "validation" / "real_meta" / "sglt2_hf_primary_benchmark.toml")
    study_effects = {str(effect["study_id"]): effect for effect in benchmark["study_effects"]}
    r_effects = {str(effect["study_id"]): effect for effect in output["study_effects"]}
    if set(study_effects) != set(r_effects):
        raise RReferenceValidationError("pairwise R study effects do not match benchmark studies.")

    max_abs_diff = 0.0
    for study_id, expected in study_effects.items():
        observed = r_effects[study_id]
        if str(observed["nct_id"]) != expected["nct_id"]:
            raise RReferenceValidationError(f"{study_id}: R output NCT ID mismatch.")
        if str(observed["pmid"]) != str(expected["pmid"]):
            raise RReferenceValidationError(f"{study_id}: R output PMID mismatch.")
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(f"{study_id} yi", observed["yi"], expected["estimate"], tolerance),
            _assert_close(f"{study_id} vi", observed["vi"], expected["variance"], tolerance),
            _assert_close(f"{study_id} sei", observed["sei"], expected["se"], tolerance),
        )

    fixed = output["metafor"]["fixed_effect"]
    expected_fixed = benchmark["candidate"]["pairwise_fixed_effect"]
    for field in ("estimate", "se", "ci_low", "ci_high", "tau2", "q"):
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"metafor fixed_effect {field}",
                fixed[field],
                expected_fixed[field],
                tolerance,
            ),
        )
    if int(fixed["df"]) != int(expected_fixed["df"]):
        raise RReferenceValidationError("metafor fixed_effect df mismatch.")

    reml = output["metafor"]["reml_hksj"]
    expected_reml = benchmark["candidate"]["pairwise_reml_hksj"]
    for field in ("estimate", "tau2", "q"):
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"metafor reml_hksj {field}",
                reml[field],
                expected_reml[field],
                tolerance,
            ),
        )
    if int(reml["df"]) != int(expected_reml["df"]):
        raise RReferenceValidationError("metafor reml_hksj df mismatch.")
    if float(expected_reml["hksj_q_factor"]) != 1.0:
        raise RReferenceValidationError("expected Python artifact to use floored HKSJ q factor.")
    r_q_factor = (float(reml["se"]) / float(expected_fixed["se"])) ** 2
    expected_unfloored = float(reml["q"]) / float(reml["df"])
    _assert_close("metafor HKSJ q factor", r_q_factor, expected_unfloored, 2e-3)

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "pairwise_metafor_meta",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "metafor and meta",
        "validated_components": [
            "study_log_or_effects",
            "fixed_effect_log_or",
            "reml_tau2_q_df",
            "hksj_floor_difference_documented",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "hksj_note": (
            "metafor KNHA uses the unfloored q factor here; the Python artifact "
            "documents its HKSJ floor and is intentionally wider."
        ),
    }


def validate_multiarm_netmeta_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate the ``netmeta`` multi-arm output against the local fixture artifact."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {"schema_version", "effect_scale", "package_versions", "fixtures"},
        label="multiarm R output",
    )
    if output["schema_version"] != "multiarm_netmeta_fixture/v1":
        raise RReferenceValidationError("multiarm R output schema_version mismatch.")
    if output["effect_scale"] != "log_or":
        raise RReferenceValidationError("multiarm R output must use log_or scale.")
    _require_package_versions(output["package_versions"], {"R", "netmeta", "meta"})

    artifact = _load_toml(root / "validation" / "multiarm" / "netmeta_portfolio_multiarm_benchmark.toml")
    effects = {
        (item["fixture_id"], item["model"], item["treatment"]): item
        for item in artifact["effects"]
    }
    fits = {(item["fixture_id"], item["model"]): item for item in artifact["model_fits"]}

    max_abs_diff = 0.0
    seen_models: set[tuple[str, str]] = set()
    for fixture in output["fixtures"]:
        fixture_id = str(fixture["fixture_id"])
        for r_model, local_model in (("common", "fixed"), ("random", "random")):
            expected_model = local_model
            if (fixture_id, expected_model) not in fits:
                expected_model = "fixed"
            seen_models.add((fixture_id, expected_model))
            for treatment, observed in fixture[r_model].items():
                expected = effects[(fixture_id, expected_model, treatment)]
                max_abs_diff = max(
                    max_abs_diff,
                    _assert_close(
                        f"{fixture_id} {r_model} {treatment} estimate",
                        observed["estimate"],
                        expected["estimate"],
                        tolerance,
                    ),
                    _assert_close(
                        f"{fixture_id} {r_model} {treatment} se",
                        observed["se"],
                        expected["se"],
                        tolerance,
                    ),
                )
        if (fixture_id, "random") in fits:
            expected_fit = fits[(fixture_id, "random")]
            max_abs_diff = max(
                max_abs_diff,
                _assert_close(f"{fixture_id} tau2", fixture["tau2"], expected_fit["tau2"], tolerance),
                _assert_close(f"{fixture_id} q", fixture["q"], expected_fit["q"], tolerance),
            )
            if int(fixture["df"]) != int(expected_fit["df"]):
                raise RReferenceValidationError(f"{fixture_id}: netmeta df mismatch.")

    required_models = {("consistent", "fixed"), ("heterogeneous", "fixed"), ("heterogeneous", "random")}
    if not required_models <= seen_models:
        raise RReferenceValidationError("multiarm R output did not validate all required fixture models.")

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "multiarm_gls_netmeta_portfolio_fixture",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "netmeta",
        "validated_components": [
            "multiarm_fixed_effect_estimates",
            "multiarm_random_effect_estimates",
            "multiarm_standard_errors",
            "random_effect_tau2_q_df",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
    }


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _require_keys(raw: dict[str, Any], required: set[str], *, label: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        raise RReferenceValidationError(f"{label} missing required keys: {missing}")


def _require_package_versions(raw: Any, required: set[str]) -> None:
    if not isinstance(raw, dict):
        raise RReferenceValidationError("R reference output package_versions must be an object.")
    missing = sorted(required - set(raw))
    if missing:
        raise RReferenceValidationError(f"R reference output missing package versions: {missing}")
    for package_name in required:
        if not str(raw[package_name]).strip():
            raise RReferenceValidationError(f"R package version for {package_name} must not be empty.")


def _assert_close(label: str, observed: Any, expected: Any, tolerance: float) -> float:
    observed_float = float(observed)
    expected_float = float(expected)
    difference = abs(observed_float - expected_float)
    if difference > tolerance:
        raise RReferenceValidationError(
            f"{label} differs by {difference:.6g}, exceeding tolerance {tolerance:.6g}."
        )
    return difference
