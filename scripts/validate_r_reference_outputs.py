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
    validate_component_netmeta_cnma_output,
    validate_crossnma_sglt2_compatibility_output,
    validate_ctgov_hr_network_netmeta_output,
    validate_dose_response_metafor_polynomial_output,
    validate_dta_mada_reitsma_output,
    validate_dta_mada_source_table_output,
    validate_mbnmadose_semaglutide_polynomial_output,
    validate_multinma_sglt2_binary_nma_output,
    validate_multiarm_netmeta_output,
    validate_pairwise_metafor_meta_output,
    validate_survival_hr_metafor_pairwise_output,
)
from bias_nma_adv.real_meta import sha256_file  # noqa: E402


PAIRWISE_OUTPUT = Path("validation/reference_runs/pairwise_metafor_meta_output.json")
PAIRWISE_REPORT = Path("validation/reference_runs/pairwise_metafor_meta_reference.toml")
MULTINMA_OUTPUT = Path("validation/reference_runs/multinma_sglt2_binary_nma_output.json")
MULTINMA_REPORT = Path("validation/reference_runs/multinma_sglt2_binary_nma_reference.toml")
MULTIARM_OUTPUT = Path("validation/reference_runs/multiarm_netmeta_output.json")
MULTIARM_REPORT = Path("validation/reference_runs/multiarm_netmeta_reference.toml")
DTA_OUTPUT = Path("validation/reference_runs/dta_mada_reitsma_output.json")
DTA_REPORT = Path("validation/reference_runs/dta_mada_reitsma_reference.toml")
DTA_SOURCE_OUTPUT = Path("validation/reference_runs/dta_mada_reitsma_midkine_source_output.json")
DTA_SOURCE_REPORT = Path("validation/reference_runs/dta_mada_reitsma_midkine_source_reference.toml")
DOSE_RESPONSE_OUTPUT = Path("validation/reference_runs/dose_response_metafor_polynomial_output.json")
DOSE_RESPONSE_REPORT = Path("validation/reference_runs/dose_response_metafor_polynomial_reference.toml")
MBNMADOSE_OUTPUT = Path("validation/reference_runs/mbnmadose_semaglutide_polynomial_output.json")
MBNMADOSE_REPORT = Path("validation/reference_runs/mbnmadose_semaglutide_polynomial_reference.toml")
SGLT2_SURVIVAL_OUTPUT = Path("validation/reference_runs/sglt2_survival_hr_metafor_output.json")
SGLT2_SURVIVAL_REPORT = Path("validation/reference_runs/sglt2_survival_hr_metafor_reference.toml")
PCSK9_SURVIVAL_OUTPUT = Path("validation/reference_runs/pcsk9_survival_hr_metafor_output.json")
PCSK9_SURVIVAL_REPORT = Path("validation/reference_runs/pcsk9_survival_hr_metafor_reference.toml")
CTGOV_HR_NETWORK_OUTPUT = Path("validation/reference_runs/t2d_ctgov_hr_network_netmeta_output.json")
CTGOV_HR_NETWORK_REPORT = Path("validation/reference_runs/t2d_ctgov_hr_network_netmeta_reference.toml")
COMPONENT_OUTPUT = Path("validation/reference_runs/component_netmeta_cnma_output.json")
COMPONENT_REPORT = Path("validation/reference_runs/component_netmeta_cnma_reference.toml")
CROSSNMA_COMPAT_OUTPUT = Path("validation/reference_runs/crossnma_sglt2_compatibility_output.json")
CROSSNMA_COMPAT_REPORT = Path("validation/reference_runs/crossnma_sglt2_compatibility_preflight.toml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pairwise-output", type=Path, default=PAIRWISE_OUTPUT)
    parser.add_argument("--pairwise-report", type=Path, default=PAIRWISE_REPORT)
    parser.add_argument("--multinma-output", type=Path, default=MULTINMA_OUTPUT)
    parser.add_argument("--multinma-report", type=Path, default=MULTINMA_REPORT)
    parser.add_argument("--multiarm-output", type=Path, default=MULTIARM_OUTPUT)
    parser.add_argument("--multiarm-report", type=Path, default=MULTIARM_REPORT)
    parser.add_argument("--dta-output", type=Path, default=DTA_OUTPUT)
    parser.add_argument("--dta-report", type=Path, default=DTA_REPORT)
    parser.add_argument("--dta-source-output", type=Path, default=DTA_SOURCE_OUTPUT)
    parser.add_argument("--dta-source-report", type=Path, default=DTA_SOURCE_REPORT)
    parser.add_argument("--dose-response-output", type=Path, default=DOSE_RESPONSE_OUTPUT)
    parser.add_argument("--dose-response-report", type=Path, default=DOSE_RESPONSE_REPORT)
    parser.add_argument("--mbnmadose-output", type=Path, default=MBNMADOSE_OUTPUT)
    parser.add_argument("--mbnmadose-report", type=Path, default=MBNMADOSE_REPORT)
    parser.add_argument("--sglt2-survival-output", type=Path, default=SGLT2_SURVIVAL_OUTPUT)
    parser.add_argument("--sglt2-survival-report", type=Path, default=SGLT2_SURVIVAL_REPORT)
    parser.add_argument("--pcsk9-survival-output", type=Path, default=PCSK9_SURVIVAL_OUTPUT)
    parser.add_argument("--pcsk9-survival-report", type=Path, default=PCSK9_SURVIVAL_REPORT)
    parser.add_argument("--ctgov-hr-network-output", type=Path, default=CTGOV_HR_NETWORK_OUTPUT)
    parser.add_argument("--ctgov-hr-network-report", type=Path, default=CTGOV_HR_NETWORK_REPORT)
    parser.add_argument("--component-output", type=Path, default=COMPONENT_OUTPUT)
    parser.add_argument("--component-report", type=Path, default=COMPONENT_REPORT)
    parser.add_argument("--crossnma-compat-output", type=Path, default=CROSSNMA_COMPAT_OUTPUT)
    parser.add_argument("--crossnma-compat-report", type=Path, default=CROSSNMA_COMPAT_REPORT)
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
        multinma_output = _resolve(root, args.multinma_output)
        multinma_summary = validate_multinma_sglt2_binary_nma_output(
            multinma_output,
            repo_root=root,
        )
        multiarm_summary = validate_multiarm_netmeta_output(
            multiarm_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        dta_output = _resolve(root, args.dta_output)
        dta_summary = validate_dta_mada_reitsma_output(
            dta_output,
            repo_root=root,
        )
        dta_source_output = _resolve(root, args.dta_source_output)
        dta_source_summary = validate_dta_mada_source_table_output(
            dta_source_output,
            repo_root=root,
        )
        dose_response_output = _resolve(root, args.dose_response_output)
        dose_response_summary = validate_dose_response_metafor_polynomial_output(
            dose_response_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        mbnmadose_output = _resolve(root, args.mbnmadose_output)
        mbnmadose_summary = validate_mbnmadose_semaglutide_polynomial_output(
            mbnmadose_output,
            repo_root=root,
        )
        sglt2_survival_output = _resolve(root, args.sglt2_survival_output)
        sglt2_survival_summary = validate_survival_hr_metafor_pairwise_output(
            sglt2_survival_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        pcsk9_survival_output = _resolve(root, args.pcsk9_survival_output)
        pcsk9_survival_summary = validate_survival_hr_metafor_pairwise_output(
            pcsk9_survival_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        ctgov_hr_network_output = _resolve(root, args.ctgov_hr_network_output)
        ctgov_hr_network_summary = validate_ctgov_hr_network_netmeta_output(
            ctgov_hr_network_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        component_output = _resolve(root, args.component_output)
        component_summary = validate_component_netmeta_cnma_output(
            component_output,
            repo_root=root,
            tolerance=args.tolerance,
        )
        crossnma_compat_output = _resolve(root, args.crossnma_compat_output)
        crossnma_compat_summary = validate_crossnma_sglt2_compatibility_output(
            crossnma_compat_output,
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
            _resolve(root, args.multinma_report),
            _multinma_report(
                root=root,
                output_path=multinma_output,
                summary=multinma_summary,
                checked_at=args.checked_at,
            ),
        )
        _write_report(
            _resolve(root, args.dta_report),
            _dta_report(
                root=root,
                output_path=dta_output,
                summary=dta_summary,
                checked_at=args.checked_at,
            ),
        )
        _write_report(
            _resolve(root, args.dta_source_report),
            _dta_source_report(
                root=root,
                output_path=dta_source_output,
                summary=dta_source_summary,
                checked_at=args.checked_at,
                tolerance=1e-5,
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
        _write_report(
            _resolve(root, args.dose_response_report),
            _dose_response_report(
                root=root,
                output_path=dose_response_output,
                summary=dose_response_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
        _write_report(
            _resolve(root, args.mbnmadose_report),
            _mbnmadose_report(
                root=root,
                output_path=mbnmadose_output,
                summary=mbnmadose_summary,
                checked_at=args.checked_at,
            ),
        )
        _write_report(
            _resolve(root, args.sglt2_survival_report),
            _survival_hr_report(
                root=root,
                output_path=sglt2_survival_output,
                summary=sglt2_survival_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
        _write_report(
            _resolve(root, args.pcsk9_survival_report),
            _survival_hr_report(
                root=root,
                output_path=pcsk9_survival_output,
                summary=pcsk9_survival_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
        _write_report(
            _resolve(root, args.ctgov_hr_network_report),
            _ctgov_hr_network_report(
                root=root,
                output_path=ctgov_hr_network_output,
                summary=ctgov_hr_network_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
        _write_report(
            _resolve(root, args.component_report),
            _component_report(
                root=root,
                output_path=component_output,
                summary=component_summary,
                checked_at=args.checked_at,
                tolerance=args.tolerance,
            ),
        )
        _write_report(
            _resolve(root, args.crossnma_compat_report),
            _crossnma_compat_report(
                root=root,
                output_path=crossnma_compat_output,
                summary=crossnma_compat_summary,
                checked_at=args.checked_at,
            ),
        )
    except (OSError, RReferenceValidationError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        print(f"R reference output validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        "R reference output validation passed: "
        f"{args.pairwise_report}, {args.multinma_report}, {args.multiarm_report}, {args.dta_report}, "
        f"{args.dta_source_report}, {args.dose_response_report}, {args.mbnmadose_report}, "
        f"{args.sglt2_survival_report}, "
        f"{args.pcsk9_survival_report}, {args.ctgov_hr_network_report}, {args.component_report}, "
        f"{args.crossnma_compat_report}"
    )
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


def _multinma_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/real_meta/sglt2_hf_primary_events.csv"),
        Path("validation/real_meta/sglt2_hf_primary_sources.toml"),
        Path("validation/real_meta/sglt2_hf_primary_benchmark.toml"),
        Path("external/r/multinma_sglt2_binary_nma.R"),
    ]
    tolerances = summary["tolerance"]
    tolerance_label = (
        f"posterior mean abs <= {tolerances['mean']:g}; "
        f"posterior sd abs <= {tolerances['sd']:g}; "
        f"R-hat <= {tolerances['rhat']:g}; "
        f"n_eff >= {tolerances['neff']:g}; divergences = 0; treedepth <= 10"
    )
    model = output["model"]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "bayesian_nma_multinma_cmdstan",
        "adapter_id": "r_multinma_sglt2_binary_nma_output_validation",
        "reference_method": "multinma fixed-effect binomial NMA via rstan",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/multinma_sglt2_binary_nma.R",
            "--events",
            "validation/real_meta/sglt2_hf_primary_events.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
            "--chains",
            str(model["chains"]),
            "--iter",
            str(model["iter"]),
            "--warmup",
            str(model["warmup"]),
            "--seed",
            str(model["seed"]),
            "--adapt-delta",
            str(model["adapt_delta"]),
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": tolerance_label,
        "skip_reason": "",
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [summary["source_policy_note"]],
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


def _dta_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/dta/dta_algorithmic_fixture.csv"),
        Path("validation/dta/dta_algorithmic_fixture.toml"),
        Path("external/r/dta_mada_reitsma_fixture.R"),
    ]
    tolerances = summary["tolerance"]
    tolerance_label = (
        f"probability <= {tolerances['probability']:g}; "
        f"log/auc <= {tolerances['log']:g}; "
        f"variance <= {tolerances['variance']:g}; "
        f"rho <= {tolerances['rho']:g}"
    )
    return {
        "schema_version": "reference_run/v1",
        "target_id": "dta_bivariate_hsroc_reference",
        "adapter_id": "r_mada_dta_reitsma_output_validation",
        "reference_method": "mada::reitsma",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/dta_mada_reitsma_fixture.R",
            "--input",
            "validation/dta/dta_algorithmic_fixture.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": tolerance_label,
        "skip_reason": "",
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [summary["source_policy_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _dta_source_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/dta/midkine_elisa_cancer_dta_2x2.csv"),
        Path("validation/dta/midkine_elisa_cancer_dta.toml"),
        Path("validation/dta/midkine_elisa_cancer_dta_benchmark.toml"),
        Path("validation/source_checks/midkine_elisa_cancer_dta_check.json"),
        Path("external/r/dta_mada_reitsma_source_table.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "dta_source_table_mada_reitsma_smoke",
        "adapter_id": "r_mada_dta_midkine_source_output_validation",
        "reference_method": "mada::reitsma source-backed DTA table",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/dta_mada_reitsma_source_table.R",
            "--input",
            "validation/dta/midkine_elisa_cancer_dta_2x2.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
            "--benchmark-id",
            "midkine_elisa_cancer_dta",
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": f"absolute <= {tolerance:g} for main bivariate parameters; AUC exported only",
        "skip_reason": "",
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [summary["source_policy_note"], summary["auc_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _dose_response_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/dose_response/semaglutide_obesity_dose_response_effects.csv"),
        Path("validation/dose_response/semaglutide_obesity_dose_response.toml"),
        Path("validation/dose_response/semaglutide_obesity_dose_response_benchmark.toml"),
        Path("validation/source_checks/semaglutide_obesity_dose_response_check.json"),
        Path("external/r/dose_response_metafor_polynomial.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "dose_response_metafor_polynomial_smoke",
        "adapter_id": "r_metafor_dose_response_polynomial_output_validation",
        "reference_method": "metafor fixed-effect polynomial meta-regression",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/dose_response_metafor_polynomial.R",
            "--effects",
            "validation/dose_response/semaglutide_obesity_dose_response_effects.csv",
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
        "notes": [summary["source_policy_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _mbnmadose_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/dose_response/semaglutide_obesity_dose_response_arms.csv"),
        Path("validation/dose_response/semaglutide_obesity_dose_response.toml"),
        Path("validation/dose_response/semaglutide_obesity_dose_response_benchmark.toml"),
        Path("validation/source_checks/semaglutide_obesity_dose_response_check.json"),
        Path("external/r/mbnmadose_semaglutide_polynomial.R"),
    ]
    tolerances = summary["tolerance"]
    tolerance_label = (
        f"posterior mean abs <= {tolerances['mean']:g}; "
        f"posterior sd abs <= {tolerances['sd']:g}; "
        f"R-hat <= {tolerances['rhat']:g}; "
        f"n_eff >= {tolerances['neff']:g}"
    )
    model = output["model"]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "dose_response_mbnmadose",
        "adapter_id": "r_mbnmadose_semaglutide_polynomial_output_validation",
        "reference_method": "MBNMAdose common-effect linear polynomial dose-response smoke",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/mbnmadose_semaglutide_polynomial.R",
            "--arms",
            "validation/dose_response/semaglutide_obesity_dose_response_arms.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
            "--chains",
            str(model["chains"]),
            "--iter",
            str(model["iter"]),
            "--burnin",
            str(model["burnin"]),
            "--thin",
            str(model["thin"]),
            "--seed",
            str(model["seed"]),
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": tolerance_label,
        "skip_reason": "",
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [summary["source_policy_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _survival_hr_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    benchmark_id = str(summary["benchmark_id"])
    if benchmark_id == "sglt2_hf_reported_hr":
        input_artifacts = [
            Path("validation/survival/sglt2_hf_reported_hr_effects.csv"),
            Path("validation/survival/sglt2_hf_reported_hrs.toml"),
            Path("validation/survival/sglt2_hf_reported_hr_benchmark.toml"),
            Path("validation/source_checks/sglt2_hf_reported_hr_tokens.json"),
            Path("validation/source_checks/sglt2_hf_reported_hr_source_check.json"),
            Path("external/r/survival_hr_metafor_pairwise.R"),
        ]
        adapter_id = "r_metafor_sglt2_survival_hr_output_validation"
    elif benchmark_id == "pcsk9_mace_reported_hr":
        input_artifacts = [
            Path("validation/survival/pcsk9_mace_reported_hr_effects.csv"),
            Path("validation/survival/pcsk9_mace_reported_hrs.toml"),
            Path("validation/survival/pcsk9_mace_reported_hr_benchmark.toml"),
            Path("validation/source_checks/pcsk9_mace_reported_hr_tokens.json"),
            Path("validation/source_checks/pcsk9_mace_reported_hr_source_check.json"),
            Path("external/r/survival_hr_metafor_pairwise.R"),
        ]
        adapter_id = "r_metafor_pcsk9_survival_hr_output_validation"
    else:
        raise ValueError(f"unsupported survival HR benchmark_id {benchmark_id!r}")

    return {
        "schema_version": "reference_run/v1",
        "target_id": "reported_hr_survival_metafor_pairwise",
        "adapter_id": adapter_id,
        "reference_method": "metafor fixed-effect reported-HR meta-analysis",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/survival_hr_metafor_pairwise.R",
            "--benchmark-id",
            benchmark_id,
            "--effects",
            input_artifacts[0].as_posix(),
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
        "notes": [summary["source_policy_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _ctgov_hr_network_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/networks/t2d_mace_ctgov_hr_network_effects.csv"),
        Path("validation/networks/t2d_mace_ctgov_hrs.toml"),
        Path("validation/networks/t2d_mace_ctgov_hr_network_benchmark.toml"),
        Path("validation/source_checks/t2d_mace_ctgov_hr_network_check.json"),
        Path("external/r/ctgov_hr_network_netmeta.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "ctgov_hr_network_netmeta_star",
        "adapter_id": "r_netmeta_t2d_ctgov_hr_network_output_validation",
        "reference_method": "netmeta fixed-effect CT.gov reported-HR star network",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/ctgov_hr_network_netmeta.R",
            "--effects",
            "validation/networks/t2d_mace_ctgov_hr_network_effects.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
            "--reference",
            "placebo",
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
        "notes": [summary["source_policy_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _component_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
    tolerance: float,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/component/netmeta_component_fixture_effects.csv"),
        Path("validation/component/netmeta_component_fixture_benchmark.toml"),
        Path("external/r/component_netmeta_cnma_fixture.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "component_nma_netmeta_cnma",
        "adapter_id": "r_netmeta_component_cnma_output_validation",
        "reference_method": "netmeta::discomb additive CNMA",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/component_netmeta_cnma_fixture.R",
            "--effects",
            "validation/component/netmeta_component_fixture_effects.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
            "--inactive",
            "Placebo",
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
        "notes": [summary["source_policy_note"]],
        "input_sha256": {
            path.as_posix(): sha256_file(root / path)
            for path in input_artifacts
        },
        "output_sha256": {
            output_path.relative_to(root).as_posix(): sha256_file(output_path),
        },
    }


def _crossnma_compat_report(
    *,
    root: Path,
    output_path: Path,
    summary: dict[str, Any],
    checked_at: str,
) -> dict[str, Any]:
    output = load_r_reference_output(output_path)
    input_artifacts = [
        Path("validation/cross_design/sglt2_rct_nrs_cross_design_effects.csv"),
        Path("validation/cross_design/sglt2_rct_nrs_cross_design.toml"),
        Path("validation/cross_design/sglt2_rct_nrs_cross_design_benchmark.toml"),
        Path("validation/source_checks/sglt2_rct_nrs_cross_design_check.json"),
        Path("external/r/crossnma_sglt2_compatibility_preflight.R"),
    ]
    return {
        "schema_version": "reference_run/v1",
        "target_id": "cross_design_crossnma",
        "adapter_id": "r_crossnma_sglt2_compatibility_preflight",
        "reference_method": summary["reference_method"],
        "status": "failed",
        "certification_effect": "none",
        "checked_at": checked_at,
        "command": [
            "Rscript",
            "--vanilla",
            "external/r/crossnma_sglt2_compatibility_preflight.R",
            "--input",
            "validation/cross_design/sglt2_rct_nrs_cross_design_effects.csv",
            "--output",
            output_path.relative_to(root).as_posix(),
        ],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {str(key): str(value) for key, value in output["package_versions"].items()},
        "input_artifacts": [path.as_posix() for path in input_artifacts],
        "output_artifacts": [output_path.relative_to(root).as_posix()],
        "tolerance": "",
        "skip_reason": summary["skip_reason"],
        "validated_components": summary["validated_components"],
        "max_abs_difference": float(summary["max_abs_difference"]),
        "notes": [
            "Expected fail-closed compatibility preflight: no crossnma model is run on incompatible log-HR estimands."
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
