"""Large-scale validation evidence gate.

This module distinguishes current source-backed evidence from the much larger
benchmark surface required before broad superiority or production claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping

from bias_nma_adv.certification import ReferenceRunReport
from bias_nma_adv.simulation_matrix import SimulationMatrix


LARGE_SCALE_VALIDATION_SCHEMA_VERSION = "large_scale_validation/v1"


class LargeScaleValidationError(ValueError):
    """Raised when large-scale validation metadata is malformed."""


@dataclass(frozen=True)
class LargeScaleThresholds:
    """Minimum evidence thresholds for broad validation claims."""

    minimum_source_backed_benchmarks: int
    minimum_benchmark_study_effects: int
    minimum_unique_nct_ids: int
    minimum_unique_pmids: int
    minimum_passed_reference_reports: int
    minimum_tau2_positive_benchmarks: int
    minimum_simulation_jobs: int
    minimum_simulation_iterations: int
    required_real_domains: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "LargeScaleThresholds":
        required = {
            "minimum_source_backed_benchmarks",
            "minimum_benchmark_study_effects",
            "minimum_unique_nct_ids",
            "minimum_unique_pmids",
            "minimum_passed_reference_reports",
            "minimum_tau2_positive_benchmarks",
            "minimum_simulation_jobs",
            "minimum_simulation_iterations",
            "required_real_domains",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise LargeScaleValidationError(f"large-scale thresholds missing keys: {missing}")
        thresholds = cls(
            minimum_source_backed_benchmarks=int(raw["minimum_source_backed_benchmarks"]),
            minimum_benchmark_study_effects=int(raw["minimum_benchmark_study_effects"]),
            minimum_unique_nct_ids=int(raw["minimum_unique_nct_ids"]),
            minimum_unique_pmids=int(raw["minimum_unique_pmids"]),
            minimum_passed_reference_reports=int(raw["minimum_passed_reference_reports"]),
            minimum_tau2_positive_benchmarks=int(raw["minimum_tau2_positive_benchmarks"]),
            minimum_simulation_jobs=int(raw["minimum_simulation_jobs"]),
            minimum_simulation_iterations=int(raw["minimum_simulation_iterations"]),
            required_real_domains=tuple(str(item) for item in raw["required_real_domains"]),
        )
        thresholds.validate()
        return thresholds

    def validate(self) -> None:
        values = {
            "minimum_source_backed_benchmarks": self.minimum_source_backed_benchmarks,
            "minimum_benchmark_study_effects": self.minimum_benchmark_study_effects,
            "minimum_unique_nct_ids": self.minimum_unique_nct_ids,
            "minimum_unique_pmids": self.minimum_unique_pmids,
            "minimum_passed_reference_reports": self.minimum_passed_reference_reports,
            "minimum_tau2_positive_benchmarks": self.minimum_tau2_positive_benchmarks,
            "minimum_simulation_jobs": self.minimum_simulation_jobs,
            "minimum_simulation_iterations": self.minimum_simulation_iterations,
        }
        for key, value in values.items():
            if value <= 0:
                raise LargeScaleValidationError(f"{key} must be positive.")
        if len(self.required_real_domains) < 4:
            raise LargeScaleValidationError("required_real_domains is too narrow.")


@dataclass(frozen=True)
class LargeScaleValidationGate:
    """Validated large-scale validation gate."""

    checked_at: str
    purpose: str
    source_boundary: str
    certification_effect: str
    global_large_scale_validation_complete: bool
    claim_limit: str
    thresholds: LargeScaleThresholds

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "LargeScaleValidationGate":
        required = {
            "schema_version",
            "checked_at",
            "purpose",
            "source_boundary",
            "certification_effect",
            "global_large_scale_validation_complete",
            "claim_limit",
            "thresholds",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise LargeScaleValidationError(f"large-scale validation gate missing keys: {missing}")
        if raw["schema_version"] != LARGE_SCALE_VALIDATION_SCHEMA_VERSION:
            raise LargeScaleValidationError(
                f"schema_version must be {LARGE_SCALE_VALIDATION_SCHEMA_VERSION}."
            )
        gate = cls(
            checked_at=str(raw["checked_at"]),
            purpose=str(raw["purpose"]),
            source_boundary=str(raw["source_boundary"]),
            certification_effect=str(raw["certification_effect"]),
            global_large_scale_validation_complete=bool(
                raw["global_large_scale_validation_complete"]
            ),
            claim_limit=str(raw["claim_limit"]),
            thresholds=LargeScaleThresholds.from_mapping(raw["thresholds"]),
        )
        gate.validate()
        return gate

    def validate(self) -> None:
        if self.certification_effect != "none":
            raise LargeScaleValidationError("large-scale gate cannot certify methods.")
        if self.global_large_scale_validation_complete:
            raise LargeScaleValidationError(
                "global_large_scale_validation_complete must be set by dynamic evidence checks, not static TOML."
            )
        if "ClinicalTrials.gov" not in self.source_boundary or "PubMed" not in self.source_boundary:
            raise LargeScaleValidationError("source_boundary must preserve public-source limits.")
        claim_limit = self.claim_limit.lower()
        if "not" not in claim_limit or "superiority" not in claim_limit:
            raise LargeScaleValidationError(
                "claim_limit must block superiority claims until thresholds pass."
            )


def load_large_scale_validation_gate(path: str | Path) -> LargeScaleValidationGate:
    """Load and validate the large-scale validation gate."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return LargeScaleValidationGate.from_mapping(payload)


def summarize_large_scale_validation(
    gate: LargeScaleValidationGate,
    *,
    real_benchmark_atlas: dict[str, Any],
    simulation_matrix: SimulationMatrix,
    reference_reports: list[ReferenceRunReport],
    simulation_report: dict[str, Any] | None = None,
    formal_required_domain_exclusions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize current dynamic evidence against large-scale thresholds."""

    thresholds = gate.thresholds
    requested_domain_exclusions = {
        str(domain): str(reason)
        for domain, reason in (formal_required_domain_exclusions or {}).items()
    }
    validation_simulation_jobs = 0
    current_sim_iterations = 0
    simulation_evidence_status = "absent"
    if simulation_report is not None:
        simulation_evidence_status = str(simulation_report["status"])
        for raw_job in simulation_report.get("jobs", []):
            if raw_job.get("status") == "passed" and raw_job.get("execution_mode") == "full":
                validation_simulation_jobs += 1
                current_sim_iterations += int(raw_job.get("iterations_successful", 0))
    passed_reference_reports = [
        report for report in reference_reports if report.status == "passed"
    ]
    current_domains = set(real_benchmark_atlas["domain_counts"])
    required_domains = set(thresholds.required_real_domains)
    eligible_domain_exclusions = {
        domain: reason
        for domain, reason in requested_domain_exclusions.items()
        if domain in required_domains and domain not in current_domains
    }
    missing_domains = sorted(
        required_domains - current_domains - set(eligible_domain_exclusions)
    )
    checks = {
        "source_backed_benchmarks": (
            int(real_benchmark_atlas["n_benchmarks"]),
            thresholds.minimum_source_backed_benchmarks,
        ),
        "benchmark_study_effects": (
            int(real_benchmark_atlas["n_benchmark_study_effects"]),
            thresholds.minimum_benchmark_study_effects,
        ),
        "unique_nct_ids": (
            int(real_benchmark_atlas["n_unique_nct_ids"]),
            thresholds.minimum_unique_nct_ids,
        ),
        "unique_pmids": (
            int(real_benchmark_atlas["n_unique_pmids"]),
            thresholds.minimum_unique_pmids,
        ),
        "passed_reference_reports": (
            len(passed_reference_reports),
            thresholds.minimum_passed_reference_reports,
        ),
        "tau2_positive_benchmarks": (
            int(real_benchmark_atlas.get("n_tau2_positive_benchmarks", 0)),
            thresholds.minimum_tau2_positive_benchmarks,
        ),
        "simulation_jobs": (validation_simulation_jobs, thresholds.minimum_simulation_jobs),
        "simulation_iterations": (
            current_sim_iterations,
            thresholds.minimum_simulation_iterations,
        ),
    }
    failed_checks = [
        name for name, (observed, required) in checks.items() if observed < required
    ]
    if missing_domains:
        failed_checks.append("required_real_domains")
    if eligible_domain_exclusions:
        failed_checks.append("formally_excluded_required_real_domains")
    status = "large_scale_validated" if not failed_checks else "partial_not_large_scale"
    return {
        "schema_version": LARGE_SCALE_VALIDATION_SCHEMA_VERSION,
        "checked_at": gate.checked_at,
        "status": status,
        "global_large_scale_validation_complete": status == "large_scale_validated",
        "dynamic_counts": {
            name: {"observed": observed, "required": required}
            for name, (observed, required) in checks.items()
        },
        "missing_required_real_domains": missing_domains,
        "formally_excluded_required_real_domains": dict(
            sorted(eligible_domain_exclusions.items())
        ),
        "failed_checks": failed_checks,
        "simulation_counting_rule": (
            "Only passed full jobs from a validated simulation report count toward "
            "large-scale validation; smoke, planned, and unexecuted matrix jobs are "
            "CI checks or plans, not validation evidence."
        ),
        "simulation_evidence_status": simulation_evidence_status,
        "certification_effect": gate.certification_effect,
        "claim_limit": gate.claim_limit,
    }
