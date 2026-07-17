"""Validation for optional R reference-adapter output artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.component_nma import fit_additive_component_nma
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


def validate_multinma_sglt2_binary_nma_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    mean_tolerance: float = 0.03,
    sd_tolerance: float = 0.02,
    max_rhat: float = 1.01,
    min_neff: float = 400.0,
) -> dict[str, Any]:
    """Validate a source-backed ``multinma`` binary NMA reference candidate."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "effect_scale",
            "model",
            "package_versions",
            "study_arms",
            "relative_effect",
            "diagnostics",
            "limitations",
        },
        label="multinma SGLT2 binary output",
    )
    if output["schema_version"] != "multinma_sglt2_binary_nma/v1":
        raise RReferenceValidationError("multinma SGLT2 output schema_version mismatch.")
    if output["benchmark_id"] != "sglt2_hf_primary_log_or":
        raise RReferenceValidationError("multinma SGLT2 output benchmark_id mismatch.")
    if output["source_policy"] != "clinicaltrials_gov + pubmed_abstract only":
        raise RReferenceValidationError("multinma SGLT2 output source_policy mismatch.")
    if output["effect_scale"] != "log_or":
        raise RReferenceValidationError("multinma SGLT2 output must use log_or scale.")
    _require_package_versions(output["package_versions"], {"R", "multinma", "rstan", "jsonlite"})

    model = output["model"]
    _require_keys(
        model,
        {
            "engine",
            "likelihood",
            "link",
            "consistency",
            "trt_effects",
            "reference_treatment",
            "chains",
            "iter",
            "warmup",
            "seed",
        },
        label="multinma SGLT2 model",
    )
    expected_model = {
        "engine": "multinma",
        "likelihood": "binomial",
        "link": "logit",
        "consistency": "consistency",
        "trt_effects": "fixed",
        "reference_treatment": "Placebo",
    }
    for field, expected in expected_model.items():
        if str(model[field]) != expected:
            raise RReferenceValidationError(f"multinma SGLT2 model {field} mismatch.")
    if int(model["chains"]) < 4:
        raise RReferenceValidationError("multinma SGLT2 reference must use at least 4 chains.")

    expected_rows = _load_event_rows(root / "validation" / "real_meta" / "sglt2_hf_primary_events.csv")
    observed_rows = {
        (str(row["study_id"]), str(row["arm_role"]), str(row["treatment"])): row
        for row in output["study_arms"]
    }
    if set(observed_rows) != set(expected_rows):
        raise RReferenceValidationError("multinma SGLT2 study arms do not match source CSV rows.")
    for key, expected in expected_rows.items():
        observed = observed_rows[key]
        for field in ("trial", "nct_id", "outcome_id", "outcome_label"):
            if str(observed[field]) != str(expected[field]):
                raise RReferenceValidationError(f"{key}: multinma SGLT2 {field} mismatch.")
        if str(observed["pmid"]) != str(expected["pmid"]):
            raise RReferenceValidationError(f"{key}: multinma SGLT2 PMID mismatch.")
        for field in ("events", "n"):
            if int(observed[field]) != int(expected[field]):
                raise RReferenceValidationError(f"{key}: multinma SGLT2 {field} mismatch.")

    benchmark = _load_toml(root / "validation" / "real_meta" / "sglt2_hf_primary_benchmark.toml")
    reference = benchmark["reference"]["fixed_effect_log_or"]
    relative_effect = output["relative_effect"]
    _require_keys(
        relative_effect,
        {"mean", "sd", "ci_low", "ci_high", "bulk_ess", "tail_ess", "rhat"},
        label="multinma SGLT2 relative effect",
    )
    mean_diff = _assert_close(
        "multinma SGLT2 posterior mean vs fixed-effect log-OR",
        relative_effect["mean"],
        reference["estimate"],
        mean_tolerance,
    )
    sd_diff = _assert_close(
        "multinma SGLT2 posterior sd vs fixed-effect SE",
        relative_effect["sd"],
        reference["se"],
        sd_tolerance,
    )
    if not float(relative_effect["ci_low"]) < float(relative_effect["mean"]) < float(relative_effect["ci_high"]):
        raise RReferenceValidationError("multinma SGLT2 CrI does not contain posterior mean.")
    if float(relative_effect["bulk_ess"]) < min_neff:
        raise RReferenceValidationError("multinma SGLT2 relative-effect bulk ESS below threshold.")
    if float(relative_effect["tail_ess"]) < min_neff:
        raise RReferenceValidationError("multinma SGLT2 relative-effect tail ESS below threshold.")
    if float(relative_effect["rhat"]) > max_rhat:
        raise RReferenceValidationError("multinma SGLT2 relative-effect R-hat exceeds threshold.")

    diagnostics = output["diagnostics"]
    _require_keys(
        diagnostics,
        {"max_rhat", "min_neff", "divergent_transitions", "max_treedepth_observed"},
        label="multinma SGLT2 diagnostics",
    )
    if float(diagnostics["max_rhat"]) > max_rhat:
        raise RReferenceValidationError("multinma SGLT2 maximum R-hat exceeds threshold.")
    if float(diagnostics["min_neff"]) < min_neff:
        raise RReferenceValidationError("multinma SGLT2 minimum n_eff below threshold.")
    if int(diagnostics["divergent_transitions"]) != 0:
        raise RReferenceValidationError("multinma SGLT2 reference has divergent transitions.")
    if int(diagnostics["max_treedepth_observed"]) > 10:
        raise RReferenceValidationError("multinma SGLT2 treedepth exceeds configured limit.")

    limitations = [str(item).lower() for item in output["limitations"]]
    if not any("not broad bayesian nma parity" in item for item in limitations):
        raise RReferenceValidationError("multinma SGLT2 output must preserve broad-parity limitation.")
    if not any("not ml-nmr" in item for item in limitations):
        raise RReferenceValidationError("multinma SGLT2 output must preserve ML-NMR limitation.")

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "bayesian_nma_multinma_cmdstan",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "multinma fixed-effect binomial NMA via rstan",
        "validated_components": [
            "source_backed_binary_arm_counts",
            "multinma_fixed_effect_log_or",
            "posterior_mean_matches_fixed_effect_reference",
            "rhat_ess_and_divergence_diagnostics",
            "broad_bayesian_parity_limitation_preserved",
        ],
        "max_abs_difference": max(mean_diff, sd_diff),
        "tolerance": {
            "mean": mean_tolerance,
            "sd": sd_tolerance,
            "rhat": max_rhat,
            "neff": min_neff,
        },
        "diagnostics": diagnostics,
        "source_policy_note": (
            "This validates one source-backed two-treatment SGLT2 binomial NMA "
            "against multinma/rstan only. It is not broad Bayesian NMA, ML-NMR, "
            "ranking, inconsistency, clinical, or HTA certification."
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


def validate_dta_mada_source_table_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Validate ``mada::reitsma`` output against the source-backed DTA table."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "evidence_mode",
            "effect_scale",
            "certification_effect",
            "package_versions",
            "n_studies",
            "continuity_correction",
            "correction_control",
            "method",
            "converged",
            "study_effects",
            "summary",
            "limitations",
        },
        label="source-backed DTA mada output",
    )
    if output["schema_version"] != "dta_mada_reitsma_source_table/v1":
        raise RReferenceValidationError("source-backed DTA mada schema_version mismatch.")
    if output["benchmark_id"] != "midkine_elisa_cancer_dta":
        raise RReferenceValidationError("source-backed DTA mada benchmark_id mismatch.")
    if output["source_policy"] != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
        raise RReferenceValidationError("source-backed DTA mada source_policy mismatch.")
    if output["evidence_mode"] != "open_access_jats_table_2x2":
        raise RReferenceValidationError("source-backed DTA mada evidence_mode mismatch.")
    if output["effect_scale"] != "logit_sensitivity_and_logit_false_positive_rate":
        raise RReferenceValidationError("source-backed DTA mada effect_scale mismatch.")
    if output["certification_effect"] != "none":
        raise RReferenceValidationError("source-backed DTA mada output cannot certify a module.")
    _require_package_versions(output["package_versions"], {"R", "mada", "jsonlite"})
    if str(output["method"]) != "reml":
        raise RReferenceValidationError("source-backed DTA mada output must use REML.")
    if str(output["correction_control"]) != "all":
        raise RReferenceValidationError("source-backed DTA mada output must use all-cell correction.")
    if not bool(output["converged"]):
        raise RReferenceValidationError("source-backed DTA mada output did not converge.")

    benchmark = _load_toml(root / "validation" / "dta" / "midkine_elisa_cancer_dta_benchmark.toml")
    expected_effects = {
        str(effect["study_id"]): effect for effect in benchmark["study_effects"]
    }
    observed_effects = {
        str(effect["study_id"]): effect for effect in output["study_effects"]
    }
    if set(expected_effects) != set(observed_effects):
        raise RReferenceValidationError("source-backed DTA mada study rows do not match benchmark.")
    if int(output["n_studies"]) != len(expected_effects):
        raise RReferenceValidationError("source-backed DTA mada n_studies mismatch.")

    for study_id, expected in expected_effects.items():
        observed = observed_effects[study_id]
        for field in (
            "citation",
            "country",
            "cancer",
            "index_test",
            "reference_standard",
            "threshold",
            "source_type",
            "source_doi",
            "table_doi",
            "table_id",
            "row_label",
        ):
            if str(observed[field]) != str(expected[field]):
                raise RReferenceValidationError(f"{study_id}: source-backed DTA {field} mismatch.")
        for field in ("tp", "fp", "fn", "tn"):
            if int(observed[field]) != int(expected[field]):
                raise RReferenceValidationError(f"{study_id}: source-backed DTA {field} mismatch.")

    summary = output["summary"]
    expected_fit = benchmark["candidate"]["bivariate_logitnormal_reml"]
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
        label="source-backed DTA mada summary",
    )
    max_abs_diff = _assert_many_close(
        {
            "pooled_sensitivity": (
                summary["pooled_sensitivity"],
                expected_fit["pooled_sensitivity"],
            ),
            "pooled_specificity": (
                summary["pooled_specificity"],
                expected_fit["pooled_specificity"],
            ),
            "logit_sensitivity": (
                summary["logit_sensitivity"],
                expected_fit["logit_sensitivity"],
            ),
            "logit_fpr": (summary["logit_fpr"], expected_fit["logit_fpr"]),
            "tau2_sensitivity": (
                summary["tau2_sensitivity"],
                expected_fit["tau2_sensitivity"],
            ),
            "tau2_fpr": (summary["tau2_fpr"], expected_fit["tau2_fpr"]),
            "cov_sensitivity_fpr": (
                summary["cov_sensitivity_fpr"],
                expected_fit["cov_sensitivity_fpr"],
            ),
            "rho_sensitivity_fpr": (
                summary["rho_sensitivity_fpr"],
                expected_fit["rho_sensitivity_fpr"],
            ),
            "log_diagnostic_odds_ratio": (
                summary["log_diagnostic_odds_ratio"],
                expected_fit["log_diagnostic_odds_ratio"],
            ),
            "diagnostic_odds_ratio": (
                summary["diagnostic_odds_ratio"],
                expected_fit["diagnostic_odds_ratio"],
            ),
        },
        tolerance,
    )
    auc = float(summary["auc"])
    if auc <= 0.0 or auc >= 1.0:
        raise RReferenceValidationError("source-backed DTA mada AUC must be in (0, 1).")
    limitations = [str(item).lower() for item in output["limitations"]]
    if not any("not clinical diagnostic accuracy guidance" in item for item in limitations):
        raise RReferenceValidationError("source-backed DTA mada limitations must block clinical guidance.")

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "dta_source_table_mada_reitsma_smoke",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "mada::reitsma source-backed DTA table",
        "benchmark_id": "midkine_elisa_cancer_dta",
        "validated_components": [
            "source_backed_tp_fp_fn_tn_rows",
            "all_cell_correction_convention",
            "pooled_sensitivity_specificity",
            "between_study_covariance",
            "clinical_guidance_limitation_preserved",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "auc_note": (
            "mada AUC is exported but not treated as local parity because the "
            "prototype uses a separate conditional SROC integration formula."
        ),
        "source_policy_note": (
            "This validates one open-access source-backed DTA table only; it is "
            "not broad HSROC parity or diagnostic-accuracy certification."
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


def validate_ctgov_binary_network_netmeta_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate ``netmeta`` output against the source-backed CT.gov binary network."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {"schema_version", "effect_scale", "package_versions", "fixtures"},
        label="CT.gov binary netmeta output",
    )
    if output["schema_version"] != "multiarm_netmeta_fixture/v1":
        raise RReferenceValidationError("CT.gov binary netmeta output schema_version mismatch.")
    if output["effect_scale"] != "log_or":
        raise RReferenceValidationError("CT.gov binary netmeta output must use log_or scale.")
    _require_package_versions(output["package_versions"], {"R", "netmeta", "meta"})

    artifact = _load_toml(
        root / "validation" / "networks" / "psoriasis_pasi90_ctgov_binary_network_benchmark.toml"
    )
    fixture = _single_fixture(output["fixtures"], "psoriasis_pasi90_ctgov_binary_network")
    if fixture["reference_treatment"] != artifact["reference_treatment"]:
        raise RReferenceValidationError("CT.gov binary netmeta reference_treatment mismatch.")

    _validate_ctgov_binary_reference_input_csv(
        root / "validation" / "reference_runs" / "psoriasis_pasi90_ctgov_binary_network_arms.csv",
        artifact,
    )

    max_abs_diff = 0.0
    for r_model, local_model in (("common", "fixed_effect"), ("random", "random_effect")):
        expected_effects = {
            str(effect["treatment"]): effect
            for effect in artifact["candidate"][local_model]["effects"]
        }
        observed_effects = fixture[r_model]
        if set(expected_effects) != set(observed_effects):
            raise RReferenceValidationError(
                f"CT.gov binary netmeta {r_model} treatment effects mismatch."
            )
        for treatment, expected in expected_effects.items():
            observed = observed_effects[treatment]
            max_abs_diff = max(
                max_abs_diff,
                _assert_close(
                    f"{treatment} CT.gov binary netmeta {r_model} estimate",
                    observed["estimate"],
                    expected["estimate"],
                    tolerance,
                ),
                _assert_close(
                    f"{treatment} CT.gov binary netmeta {r_model} se",
                    observed["se"],
                    expected["se"],
                    tolerance,
                ),
            )

    expected_random = artifact["candidate"]["random_effect"]
    max_abs_diff = max(
        max_abs_diff,
        _assert_close("CT.gov binary netmeta tau2", fixture["tau2"], expected_random["tau2"], tolerance),
        _assert_close("CT.gov binary netmeta Q", fixture["q"], expected_random["q"], tolerance),
    )
    if int(fixture["df"]) != int(expected_random["df"]):
        raise RReferenceValidationError("CT.gov binary netmeta df mismatch.")

    limitations = [str(item).lower() for item in artifact["limitations"]]
    if not any("not broad inconsistency performance" in item for item in limitations):
        raise RReferenceValidationError("CT.gov binary artifact must preserve broad-inconsistency limitation.")
    if int(artifact["closed_loop_cycle_rank"]) <= 0:
        raise RReferenceValidationError("CT.gov binary artifact is not a closed-loop network.")

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "ctgov_binary_network_netmeta_closed_loop",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "netmeta CT.gov arm-count closed-loop binary network",
        "benchmark_id": artifact["benchmark_id"],
        "validated_components": [
            "source_backed_arm_count_rows",
            "netmeta_common_effect_estimates",
            "netmeta_common_effect_standard_errors",
            "netmeta_random_tau2_q_df",
            "closed_loop_source_backed_network",
            "broad_inconsistency_limitation_preserved",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "source_policy_note": (
            "This validates one CT.gov arm-count closed-loop psoriasis network "
            "against netmeta. It is not broad netmeta parity, node-splitting "
            "parity, clinical guidance, or HTA certification."
        ),
    }


def validate_component_netmeta_cnma_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate ``netmeta::discomb`` component output against the local CNMA core."""

    root = Path(repo_root)
    output = load_r_reference_output(output_path)
    _require_keys(
        output,
        {
            "schema_version",
            "fixture_id",
            "source_policy",
            "effect_scale",
            "inactive_treatment",
            "package_versions",
            "n_studies",
            "n_contrasts",
            "n_components",
            "study_effects",
            "component_effects",
            "additive",
            "limitations",
        },
        label="component netmeta output",
    )
    if output["schema_version"] != "component_netmeta_cnma_fixture/v1":
        raise RReferenceValidationError("component netmeta output schema_version mismatch.")
    if output["fixture_id"] != "netmeta_component_fixture":
        raise RReferenceValidationError("component netmeta output fixture_id mismatch.")
    if output["source_policy"] != "algorithmic_fixture_not_clinical_evidence":
        raise RReferenceValidationError("component netmeta output must remain nonclinical.")
    if output["effect_scale"] != "mean_difference":
        raise RReferenceValidationError("component netmeta output effect_scale mismatch.")
    if output["inactive_treatment"] != "Placebo":
        raise RReferenceValidationError("component netmeta output inactive_treatment mismatch.")
    _require_package_versions(output["package_versions"], {"R", "netmeta", "jsonlite"})

    fixture_path = root / "validation" / "component" / "netmeta_component_fixture_effects.csv"
    fixture_rows = _load_component_fixture_rows(fixture_path)
    observed_rows = {str(item["study_id"]): item for item in output["study_effects"]}
    expected_rows = {str(item["study_id"]): item for item in fixture_rows}
    if set(observed_rows) != set(expected_rows):
        raise RReferenceValidationError("component netmeta study rows do not match fixture.")
    for study_id, expected in expected_rows.items():
        observed = observed_rows[study_id]
        for field in ("treat1", "treat2"):
            if str(observed[field]) != str(expected[field]):
                raise RReferenceValidationError(f"{study_id}: component {field} mismatch.")
        _assert_close(f"{study_id} estimate", observed["estimate"], expected["estimate"], tolerance)
        _assert_close(f"{study_id} se", observed["se"], expected["se"], tolerance)

    if int(output["n_studies"]) != len(fixture_rows):
        raise RReferenceValidationError("component netmeta n_studies mismatch.")
    if int(output["n_contrasts"]) != len(fixture_rows):
        raise RReferenceValidationError("component netmeta n_contrasts mismatch.")

    local_fit = fit_additive_component_nma(fixture_rows)
    if int(output["n_components"]) != len(local_fit.components):
        raise RReferenceValidationError("component netmeta n_components mismatch.")

    max_abs_diff = 0.0
    observed_components = output["component_effects"]
    for expected in local_fit.component_effects:
        observed = observed_components.get(expected.name)
        if not isinstance(observed, dict):
            raise RReferenceValidationError(f"missing netmeta component effect {expected.name!r}.")
        max_abs_diff = max(
            max_abs_diff,
            _assert_close(
                f"component {expected.name} estimate",
                observed["estimate"],
                expected.estimate,
                tolerance,
            ),
            _assert_close(
                f"component {expected.name} se",
                observed["se"],
                expected.se,
                tolerance,
            ),
        )

    additive = output["additive"]
    max_abs_diff = max(
        max_abs_diff,
        _assert_close("component additive Q", additive["q"], local_fit.q, tolerance),
    )
    if int(additive["df"]) != local_fit.df:
        raise RReferenceValidationError("component additive df mismatch.")

    limitations = [str(item).lower() for item in output["limitations"]]
    if not any("not source-backed" in item for item in limitations):
        raise RReferenceValidationError("component output must preserve source-backed limitation.")
    if not any("not broad netmeta cnma" in item for item in limitations):
        raise RReferenceValidationError("component output must preserve broad-parity limitation.")

    return {
        "schema_version": "r_reference_validation/v1",
        "target_id": "component_nma_netmeta_cnma",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "reference_method": "netmeta::discomb additive CNMA",
        "validated_components": [
            "algorithmic_component_contrast_rows",
            "additive_component_effects",
            "component_standard_errors",
            "additive_q_df",
            "source_backed_limitation_preserved",
        ],
        "max_abs_difference": max_abs_diff,
        "tolerance": tolerance,
        "source_policy_note": (
            "This validates an algorithmic component-NMA fixture only; it is not "
            "source-backed CNMA validation or broad netmeta CNMA parity."
        ),
    }


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_dta_fixture_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_component_fixture_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "study_id": row["study_id"],
            "treat1": row["treat1"],
            "treat2": row["treat2"],
            "estimate": float(row["estimate"]),
            "se": float(row["se"]),
        }
        for row in rows
    ]


def _load_event_rows(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (str(row["study_id"]), str(row["arm_role"]), str(row["treatment"])): row
        for row in rows
    }


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


def _single_fixture(fixtures: Any, fixture_id: str) -> dict[str, Any]:
    if not isinstance(fixtures, list):
        raise RReferenceValidationError("R reference output fixtures must be a list.")
    matches = [item for item in fixtures if str(item.get("fixture_id", "")) == fixture_id]
    if len(matches) != 1:
        raise RReferenceValidationError(f"expected exactly one fixture named {fixture_id}.")
    return matches[0]


def _validate_ctgov_binary_reference_input_csv(path: Path, artifact: dict[str, Any]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (str(arm["study_id"]), str(arm["treatment"])): {
            "events": int(arm["events"]),
            "n": int(arm["n"]),
        }
        for arm in artifact["arm_counts"]
    }
    observed = {
        (str(row["study"]), str(row["treatment"])): {
            "events": int(row["events"]),
            "n": int(row["n"]),
        }
        for row in rows
    }
    if set(observed) != set(expected):
        raise RReferenceValidationError("CT.gov binary netmeta input CSV arm set mismatch.")
    for key, expected_counts in expected.items():
        observed_counts = observed[key]
        if observed_counts != expected_counts:
            raise RReferenceValidationError(f"CT.gov binary netmeta input CSV count mismatch for {key}.")


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
        "sglt2_ckd_reported_hr": root
        / "validation"
        / "survival"
        / "sglt2_ckd_reported_hr_benchmark.toml",
    }
    try:
        return paths[benchmark_id]
    except KeyError as exc:
        raise RReferenceValidationError(
            f"unsupported survival HR benchmark_id '{benchmark_id}'."
        ) from exc
