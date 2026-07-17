"""Validation for optional R reference-adapter output artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.dta import fit_bivariate_dta_reml


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


def validate_dta_mada_reitsma_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    probability_tolerance: float = 1e-3,
    log_tolerance: float = 2e-3,
    variance_tolerance: float = 3e-3,
    rho_tolerance: float = 6e-3,
) -> dict[str, Any]:
    """Validate ``mada::reitsma`` output against the local DTA fixture."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "fixture_id",
            "source_policy",
            "certification_effect",
            "package_versions",
            "n_studies",
            "continuity_correction",
            "correction_control",
            "method",
            "converged",
            "summary",
            "warnings",
        },
        label="DTA mada output",
    )
    if output["schema_version"] != "dta_mada_reitsma_fixture/v1":
        raise RReferenceValidationError("DTA mada output schema_version mismatch.")
    if output["fixture_id"] != "dta_algorithmic_fixture":
        raise RReferenceValidationError("DTA mada output fixture_id mismatch.")
    if output["source_policy"] != "synthetic_algorithmic_fixture_not_clinical_evidence":
        raise RReferenceValidationError("DTA mada output must remain nonclinical.")
    if output["certification_effect"] != "none":
        raise RReferenceValidationError("DTA mada output JSON cannot certify a module.")
    _require_package_versions(output["package_versions"], {"R", "mada", "jsonlite"})
    if str(output["method"]) != "reml":
        raise RReferenceValidationError("DTA mada output must use REML.")
    if str(output["correction_control"]) != "all":
        raise RReferenceValidationError("DTA mada output must use all-cell correction.")
    if not bool(output["converged"]):
        raise RReferenceValidationError("DTA mada output did not converge.")

    fixture_path = root / "validation" / "dta" / "dta_algorithmic_fixture.csv"
    fixture_rows = _load_dta_fixture_rows(fixture_path)
    if int(output["n_studies"]) != len(fixture_rows):
        raise RReferenceValidationError("DTA mada output n_studies mismatch.")

    fit = fit_bivariate_dta_reml(
        fixture_rows,
        continuity_correction=float(output["continuity_correction"]),
    )
    summary = output["summary"]
    _require_keys(
        summary,
        {
            "logit_sensitivity",
            "logit_fpr",
            "se_logit_sensitivity",
            "se_logit_fpr",
            "pooled_sensitivity",
            "pooled_specificity",
            "tau2_sensitivity",
            "tau2_fpr",
            "cov_sensitivity_fpr",
            "rho_sensitivity_fpr",
            "log_diagnostic_odds_ratio",
            "diagnostic_odds_ratio",
            "auc",
        },
        label="DTA mada summary",
    )

    probability_diff = _assert_many_close(
        {
            "pooled_sensitivity": (
                fit.pooled_sensitivity,
                summary["pooled_sensitivity"],
            ),
            "pooled_specificity": (
                fit.pooled_specificity,
                summary["pooled_specificity"],
            ),
        },
        probability_tolerance,
    )
    log_diff = _assert_many_close(
        {
            "logit_sensitivity": (fit.logit_sensitivity, summary["logit_sensitivity"]),
            "logit_fpr": (fit.logit_fpr, summary["logit_fpr"]),
            "log_diagnostic_odds_ratio": (
                fit.log_diagnostic_odds_ratio,
                summary["log_diagnostic_odds_ratio"],
            ),
            "auc": (fit.auc_trapezoid, summary["auc"]),
        },
        log_tolerance,
    )
    variance_diff = _assert_many_close(
        {
            "tau2_sensitivity": (fit.tau2_sensitivity, summary["tau2_sensitivity"]),
            "tau2_fpr": (fit.tau2_fpr, summary["tau2_fpr"]),
            "cov_sensitivity_fpr": (
                fit.cov_sensitivity_fpr,
                summary["cov_sensitivity_fpr"],
            ),
        },
        variance_tolerance,
    )
    rho_diff = _assert_many_close(
        {
            "rho_sensitivity_fpr": (
                fit.rho_sensitivity_fpr,
                summary["rho_sensitivity_fpr"],
            ),
        },
        rho_tolerance,
    )

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "dta_bivariate_hsroc_reference",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "mada::reitsma",
        "validated_components": [
            "pooled_sensitivity_specificity",
            "logit_summary_point",
            "between_study_covariance",
            "sroc_auc",
            "nonclinical_fixture_boundary",
        ],
        "max_abs_difference": max(
            probability_diff,
            log_diff,
            variance_diff,
            rho_diff,
        ),
        "max_abs_probability_difference": probability_diff,
        "max_abs_log_difference": log_diff,
        "max_abs_variance_difference": variance_diff,
        "max_abs_rho_difference": rho_diff,
        "tolerance": {
            "probability": probability_tolerance,
            "log": log_tolerance,
            "variance": variance_tolerance,
            "rho": rho_tolerance,
        },
        "source_policy_note": (
            "This is an algorithmic DTA fixture; it is not source-backed clinical evidence."
        ),
    }


def validate_dose_response_metafor_polynomial_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate ``metafor`` polynomial dose-response output against the local artifact."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "effect_scale",
            "package_versions",
            "study_effects",
            "metafor",
            "limitations",
        },
        label="dose-response metafor output",
    )
    if output["schema_version"] != "dose_response_metafor_polynomial/v1":
        raise RReferenceValidationError("dose-response metafor output schema_version mismatch.")
    if output["benchmark_id"] != "semaglutide_obesity_dose_response":
        raise RReferenceValidationError("dose-response metafor output benchmark_id mismatch.")
    if output["source_policy"] != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
        raise RReferenceValidationError("dose-response metafor output source_policy mismatch.")
    if output["effect_scale"] != "percentage_point_change_vs_placebo":
        raise RReferenceValidationError("dose-response metafor output effect_scale mismatch.")
    _require_package_versions(output["package_versions"], {"R", "metafor", "jsonlite"})

    benchmark = _load_toml(
        root
        / "validation"
        / "dose_response"
        / "semaglutide_obesity_dose_response_benchmark.toml"
    )
    expected_effects = {
        str(effect["study_id"]): effect for effect in benchmark["study_effects"]
    }
    observed_effects = {
        str(effect["study_id"]): effect for effect in output["study_effects"]
    }
    if set(expected_effects) != set(observed_effects):
        raise RReferenceValidationError(
            "dose-response metafor study effects do not match benchmark effects."
        )

    max_abs_diff = 0.0
    for study_id, expected in expected_effects.items():
        observed = observed_effects[study_id]
        if str(observed["nct_id"]) != expected["nct_id"]:
            raise RReferenceValidationError(f"{study_id}: dose-response NCT ID mismatch.")
        if str(observed["pmid"]) != str(expected["pmid"]):
            raise RReferenceValidationError(f"{study_id}: dose-response PMID mismatch.")
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(f"{study_id} dose", observed["dose"], expected["dose"], tolerance),
            _assert_close(
                f"{study_id} estimate",
                observed["estimate"],
                expected["estimate"],
                tolerance,
            ),
            _assert_close(f"{study_id} se", observed["se"], expected["se"], tolerance),
            _assert_close(
                f"{study_id} variance",
                observed["variance"],
                expected["variance"],
                tolerance,
            ),
        )

    metafor = output["metafor"]
    max_abs_diff = max(
        max_abs_diff,
        _validate_polynomial_fit(
            observed=metafor["weighted_linear"],
            expected=benchmark["candidate"]["weighted_linear"],
            label="metafor weighted_linear",
            tolerance=tolerance,
        ),
        _validate_polynomial_fit(
            observed=metafor["weighted_quadratic"],
            expected=benchmark["candidate"]["weighted_quadratic"],
            label="metafor weighted_quadratic",
            tolerance=tolerance,
        ),
    )

    limitations = [str(item) for item in output["limitations"]]
    if not any("not MBNMAdose reference matching" in item for item in limitations):
        raise RReferenceValidationError(
            "dose-response metafor output must preserve the MBNMAdose limitation."
        )

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "dose_response_metafor_polynomial_smoke",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "metafor fixed-effect polynomial meta-regression",
        "validated_components": [
            "source_backed_dose_level_effects",
            "weighted_linear_coefficients",
            "weighted_quadratic_coefficients",
            "weighted_residual_q_df",
            "mbnmadose_limitation_preserved",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "source_policy_note": (
            "This validates a source-backed dose-response smoke artifact; it is not "
            "MBNMAdose parity or dose-response NMA certification."
        ),
    }


def validate_survival_hr_metafor_pairwise_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate ``metafor`` fixed-effect reported-HR output against a local artifact."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "effect_scale",
            "package_versions",
            "study_effects",
            "metafor",
            "limitations",
        },
        label="survival HR metafor output",
    )
    if output["schema_version"] != "survival_hr_metafor_pairwise/v1":
        raise RReferenceValidationError("survival HR metafor output schema_version mismatch.")
    if output["source_policy"] != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
        raise RReferenceValidationError("survival HR metafor output source_policy mismatch.")
    if output["effect_scale"] != "log_hr":
        raise RReferenceValidationError("survival HR metafor output must use log_hr scale.")
    _require_package_versions(output["package_versions"], {"R", "metafor", "jsonlite"})

    benchmark_id = str(output["benchmark_id"])
    benchmark_path = _survival_hr_benchmark_path(root, benchmark_id)
    benchmark = _load_toml(benchmark_path)
    expected_effects = {
        str(effect["study_id"]): effect for effect in benchmark["study_effects"]
    }
    observed_effects = {
        str(effect["study_id"]): effect for effect in output["study_effects"]
    }
    if set(expected_effects) != set(observed_effects):
        raise RReferenceValidationError(
            f"{benchmark_id}: survival HR metafor study effects do not match benchmark effects."
        )

    max_abs_diff = 0.0
    for study_id, expected in expected_effects.items():
        observed = observed_effects[study_id]
        if str(observed["nct_id"]) != expected["nct_id"]:
            raise RReferenceValidationError(f"{study_id}: survival HR NCT ID mismatch.")
        if str(observed["pmid"]) != str(expected["pmid"]):
            raise RReferenceValidationError(f"{study_id}: survival HR PMID mismatch.")
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"{study_id} estimate",
                observed["estimate"],
                expected["estimate"],
                tolerance,
            ),
            _assert_close(f"{study_id} se", observed["se"], expected["se"], tolerance),
            _assert_close(
                f"{study_id} variance",
                observed["variance"],
                expected["variance"],
                tolerance,
            ),
        )

    fixed = output["metafor"]["fixed_effect"]
    expected_fixed = benchmark["candidate"]["pairwise_fixed_effect"]
    for field in ("estimate", "se", "ci_low", "ci_high", "tau2", "q"):
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"{benchmark_id} metafor fixed_effect {field}",
                fixed[field],
                expected_fixed[field],
                tolerance,
            ),
        )
    if int(fixed["df"]) != int(expected_fixed["df"]):
        raise RReferenceValidationError(f"{benchmark_id}: survival HR fixed_effect df mismatch.")

    limitations = [str(item).lower() for item in output["limitations"]]
    if not any("not a multi-treatment survival nma" in item for item in limitations):
        raise RReferenceValidationError(
            f"{benchmark_id}: survival HR output must preserve the pairwise limitation."
        )

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "reported_hr_survival_metafor_pairwise",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "metafor fixed-effect reported-HR meta-analysis",
        "benchmark_id": benchmark_id,
        "validated_components": [
            "source_backed_reported_hr_effects",
            "fixed_effect_log_hr",
            "fixed_effect_standard_error",
            "fixed_effect_q_df",
            "survival_nma_limitation_preserved",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "source_policy_note": (
            "This validates reported-HR pairwise pooling only; it is not KM "
            "reconstruction, survival NMA parity, or clinical certification."
        ),
    }


def validate_ctgov_hr_network_netmeta_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate ``netmeta`` output against the CT.gov reported-HR star network."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "effect_scale",
            "reference_treatment",
            "package_versions",
            "study_effects",
            "netmeta",
            "limitations",
        },
        label="CT.gov HR netmeta output",
    )
    if output["schema_version"] != "ctgov_hr_network_netmeta/v1":
        raise RReferenceValidationError("CT.gov HR netmeta output schema_version mismatch.")
    if output["benchmark_id"] != "t2d_mace_ctgov_hr_network":
        raise RReferenceValidationError("CT.gov HR netmeta output benchmark_id mismatch.")
    if output["source_policy"] != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
        raise RReferenceValidationError("CT.gov HR netmeta output source_policy mismatch.")
    if output["effect_scale"] != "log_hr":
        raise RReferenceValidationError("CT.gov HR netmeta output must use log_hr scale.")
    if output["reference_treatment"] != "placebo":
        raise RReferenceValidationError("CT.gov HR netmeta output reference_treatment mismatch.")
    _require_package_versions(output["package_versions"], {"R", "netmeta", "jsonlite"})

    benchmark = _load_toml(root / "validation" / "networks" / "t2d_mace_ctgov_hr_network_benchmark.toml")
    expected_effects = {
        str(effect["study_id"]): effect for effect in benchmark["study_effects"]
    }
    observed_effects = {
        str(effect["study_id"]): effect for effect in output["study_effects"]
    }
    if set(expected_effects) != set(observed_effects):
        raise RReferenceValidationError("CT.gov HR netmeta study effects do not match benchmark effects.")

    max_abs_diff = 0.0
    for study_id, expected in expected_effects.items():
        observed = observed_effects[study_id]
        if str(observed["nct_id"]) != expected["nct_id"]:
            raise RReferenceValidationError(f"{study_id}: CT.gov HR netmeta NCT ID mismatch.")
        if str(observed["analysis_treatment"]) != expected["analysis_treatment"]:
            raise RReferenceValidationError(f"{study_id}: CT.gov HR netmeta treatment mismatch.")
        if str(observed["control_treatment"]) != expected["control_treatment"]:
            raise RReferenceValidationError(f"{study_id}: CT.gov HR netmeta control mismatch.")
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"{study_id} estimate",
                observed["estimate"],
                expected["estimate"],
                tolerance,
            ),
            _assert_close(f"{study_id} se", observed["se"], expected["se"], tolerance),
            _assert_close(
                f"{study_id} variance",
                observed["variance"],
                expected["variance"],
                tolerance,
            ),
        )

    expected_fixed = {
        str(effect["treatment"]): effect
        for effect in benchmark["candidate"]["fixed_gls"]["effects"]
    }
    observed_fixed = output["netmeta"]["common"]
    if set(expected_fixed) != set(observed_fixed):
        raise RReferenceValidationError("CT.gov HR netmeta treatment effects mismatch.")
    for treatment, expected in expected_fixed.items():
        observed = observed_fixed[treatment]
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"{treatment} netmeta common estimate",
                observed["estimate"],
                expected["estimate"],
                tolerance,
            ),
            _assert_close(
                f"{treatment} netmeta common se",
                observed["se"],
                expected["se"],
                tolerance,
            ),
        )

    expected_random = benchmark["candidate"]["random_gls"]
    max_abs_diff = max(
        max_abs_diff,
        _assert_close("CT.gov HR netmeta Q", output["netmeta"]["q"], expected_random["q"], tolerance),
    )
    if int(output["netmeta"]["df"]) != int(expected_random["df"]):
        raise RReferenceValidationError("CT.gov HR netmeta df mismatch.")

    limitations = [str(item).lower() for item in output["limitations"]]
    if not any("not broad netmeta parity" in item for item in limitations):
        raise RReferenceValidationError("CT.gov HR netmeta output must preserve broad-parity limitation.")

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "ctgov_hr_network_netmeta_star",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "netmeta fixed-effect CT.gov reported-HR star network",
        "validated_components": [
            "source_backed_ctgov_reported_hr_effects",
            "fixed_effect_class_log_hr",
            "fixed_effect_class_standard_error",
            "network_q_df",
            "star_network_limitation_preserved",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "source_policy_note": (
            "This validates a CT.gov reported-HR star network only; it is not "
            "closed-loop inconsistency validation or broad netmeta parity."
        ),
    }


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_dta_fixture_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _assert_many_close(checks: dict[str, tuple[Any, Any]], tolerance: float) -> float:
    max_abs_diff = 0.0
    for label, (observed, expected) in checks.items():
        max_abs_diff = max(max_abs_diff, _assert_close(label, observed, expected, tolerance))
    return max_abs_diff


def _validate_polynomial_fit(
    *,
    observed: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    tolerance: float,
) -> float:
    _require_keys(
        observed,
        {"coefficients", "coefficient_ses", "q", "df"},
        label=label,
    )
    max_abs_diff = 0.0
    observed_coefficients = _as_list(observed["coefficients"])
    expected_coefficients = list(expected["coefficients"])
    if len(observed_coefficients) != len(expected_coefficients):
        raise RReferenceValidationError(f"{label} coefficient count mismatch.")
    for index, (observed_value, expected_value) in enumerate(
        zip(observed_coefficients, expected_coefficients, strict=True)
    ):
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"{label} coefficient {index}",
                observed_value,
                expected_value,
                tolerance,
            ),
        )

    observed_ses = _as_list(observed["coefficient_ses"])
    expected_ses = list(expected["coefficient_ses"])
    if len(observed_ses) != len(expected_ses):
        raise RReferenceValidationError(f"{label} coefficient SE count mismatch.")
    for index, (observed_value, expected_value) in enumerate(
        zip(observed_ses, expected_ses, strict=True)
    ):
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"{label} coefficient SE {index}",
                observed_value,
                expected_value,
                tolerance,
            ),
        )

    max_abs_diff = max(
        max_abs_diff,
        _assert_close(f"{label} q", observed["q"], expected["q"], tolerance),
    )
    if int(observed["df"]) != int(expected["df"]):
        raise RReferenceValidationError(f"{label} df mismatch.")
    return max_abs_diff


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _survival_hr_benchmark_path(root: Path, benchmark_id: str) -> Path:
    paths = {
        "sglt2_hf_reported_hr": root
        / "validation"
        / "survival"
        / "sglt2_hf_reported_hr_benchmark.toml",
        "pcsk9_mace_reported_hr": root
        / "validation"
        / "survival"
        / "pcsk9_mace_reported_hr_benchmark.toml",
    }
    try:
        return paths[benchmark_id]
    except KeyError as exc:
        raise RReferenceValidationError(
            f"unsupported survival HR benchmark_id '{benchmark_id}'."
        ) from exc
