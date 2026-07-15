"""Executable simulation-matrix validation and smoke-run reporting."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import time
import tomllib
from typing import Any

from bias_nma_adv.grand_benchmark_plan import GrandBenchmarkPlan
from bias_nma_adv.grand_benchmark_plan import load_grand_benchmark_plan
from bias_nma_adv.simulation import run_benchmark_simulation


SIMULATION_MATRIX_SCHEMA_VERSION = "simulation_matrix/v1"
SIMULATION_MATRIX_REPORT_SCHEMA_VERSION = "simulation_matrix_report/v1"
SUPPORTED_SCENARIO_IDS = {"binary_sparse_multiarm_bias_grid"}
ALLOWED_JOB_STATUSES = {"active", "planned"}
ALLOWED_EXECUTION_MODES = {"smoke", "full"}
REQUIRED_METRICS = {"bias", "rmse", "coverage", "mean_se"}


class SimulationMatrixError(ValueError):
    """Raised when simulation matrix metadata or outputs are invalid."""


@dataclass(frozen=True)
class SimulationMatrixJob:
    """One executable simulation job."""

    id: str
    scenario_id: str
    status: str
    execution_mode: str
    uses_real_data: bool
    certification_effect: str
    n_iterations: int
    n_studies: int
    n_treatments: int
    design_ratio: float
    true_heterogeneity: float
    true_bias: float
    true_bias_interaction: float
    covariate_effect: float
    seed: int
    min_successful_iterations: int
    required_methods: tuple[str, ...]
    required_metrics: tuple[str, ...]
    claim_limit: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SimulationMatrixJob":
        required = {
            "id",
            "scenario_id",
            "status",
            "execution_mode",
            "uses_real_data",
            "certification_effect",
            "n_iterations",
            "n_studies",
            "n_treatments",
            "design_ratio",
            "true_heterogeneity",
            "true_bias",
            "true_bias_interaction",
            "covariate_effect",
            "seed",
            "min_successful_iterations",
            "required_methods",
            "required_metrics",
            "claim_limit",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SimulationMatrixError(f"simulation matrix job missing required keys: {missing}")
        job = cls(
            id=str(raw["id"]),
            scenario_id=str(raw["scenario_id"]),
            status=str(raw["status"]),
            execution_mode=str(raw["execution_mode"]),
            uses_real_data=bool(raw["uses_real_data"]),
            certification_effect=str(raw["certification_effect"]),
            n_iterations=int(raw["n_iterations"]),
            n_studies=int(raw["n_studies"]),
            n_treatments=int(raw["n_treatments"]),
            design_ratio=float(raw["design_ratio"]),
            true_heterogeneity=float(raw["true_heterogeneity"]),
            true_bias=float(raw["true_bias"]),
            true_bias_interaction=float(raw["true_bias_interaction"]),
            covariate_effect=float(raw["covariate_effect"]),
            seed=int(raw["seed"]),
            min_successful_iterations=int(raw["min_successful_iterations"]),
            required_methods=tuple(str(item) for item in raw["required_methods"]),
            required_metrics=tuple(str(item) for item in raw["required_metrics"]),
            claim_limit=str(raw["claim_limit"]),
        )
        job.validate()
        return job

    def validate(self) -> None:
        if not self.id.strip():
            raise SimulationMatrixError("simulation matrix job id must not be empty.")
        if self.scenario_id not in SUPPORTED_SCENARIO_IDS:
            raise SimulationMatrixError(f"{self.id}: unsupported executable scenario_id {self.scenario_id!r}.")
        if self.status not in ALLOWED_JOB_STATUSES:
            raise SimulationMatrixError(f"{self.id}: unsupported status {self.status!r}.")
        if self.execution_mode not in ALLOWED_EXECUTION_MODES:
            raise SimulationMatrixError(f"{self.id}: unsupported execution_mode {self.execution_mode!r}.")
        if self.uses_real_data:
            raise SimulationMatrixError(f"{self.id}: simulation jobs must not use real data.")
        if self.certification_effect != "none":
            raise SimulationMatrixError(f"{self.id}: simulation jobs cannot certify methods.")
        if self.n_iterations <= 0:
            raise SimulationMatrixError(f"{self.id}: n_iterations must be positive.")
        if self.n_studies < 4:
            raise SimulationMatrixError(f"{self.id}: n_studies must be at least 4.")
        if self.n_treatments < 3:
            raise SimulationMatrixError(f"{self.id}: n_treatments must be at least 3.")
        if not 0.0 <= self.design_ratio <= 1.0:
            raise SimulationMatrixError(f"{self.id}: design_ratio must be in [0, 1].")
        if self.true_heterogeneity < 0.0:
            raise SimulationMatrixError(f"{self.id}: true_heterogeneity must be non-negative.")
        if self.seed < 0:
            raise SimulationMatrixError(f"{self.id}: seed must be non-negative.")
        if not 1 <= self.min_successful_iterations <= self.n_iterations:
            raise SimulationMatrixError(
                f"{self.id}: min_successful_iterations must be between 1 and n_iterations."
            )
        if not self.required_methods:
            raise SimulationMatrixError(f"{self.id}: required_methods must not be empty.")
        missing_metrics = sorted(REQUIRED_METRICS - set(self.required_metrics))
        if missing_metrics:
            raise SimulationMatrixError(f"{self.id}: required_metrics missing {missing_metrics}.")
        boundary = self.claim_limit.lower()
        if "simulation" not in boundary or "not real clinical validation" not in boundary:
            raise SimulationMatrixError(f"{self.id}: claim_limit must state the simulation-only boundary.")


@dataclass(frozen=True)
class SimulationMatrix:
    """A validated executable simulation matrix."""

    checked_at: str
    purpose: str
    grand_benchmark_plan: str
    source_policy: str
    uses_real_data: bool
    certification_effect: str
    jobs: tuple[SimulationMatrixJob, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SimulationMatrix":
        required = {
            "schema_version",
            "checked_at",
            "purpose",
            "grand_benchmark_plan",
            "source_policy",
            "uses_real_data",
            "certification_effect",
            "jobs",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SimulationMatrixError(f"simulation matrix missing required keys: {missing}")
        if raw["schema_version"] != SIMULATION_MATRIX_SCHEMA_VERSION:
            raise SimulationMatrixError(
                f"simulation matrix schema_version must be {SIMULATION_MATRIX_SCHEMA_VERSION}."
            )
        matrix = cls(
            checked_at=str(raw["checked_at"]),
            purpose=str(raw["purpose"]),
            grand_benchmark_plan=str(raw["grand_benchmark_plan"]),
            source_policy=str(raw["source_policy"]),
            uses_real_data=bool(raw["uses_real_data"]),
            certification_effect=str(raw["certification_effect"]),
            jobs=tuple(SimulationMatrixJob.from_mapping(item) for item in raw["jobs"]),
        )
        matrix.validate()
        return matrix

    def validate(self) -> None:
        if self.source_policy != "simulation_only_no_real_data":
            raise SimulationMatrixError("simulation matrix source_policy must be simulation_only_no_real_data.")
        if self.uses_real_data:
            raise SimulationMatrixError("simulation matrix must not be labelled as real data.")
        if self.certification_effect != "none":
            raise SimulationMatrixError("simulation matrix cannot certify methods.")
        if not self.jobs:
            raise SimulationMatrixError("simulation matrix must define at least one job.")
        ids = [job.id for job in self.jobs]
        duplicates = sorted({job_id for job_id in ids if ids.count(job_id) > 1})
        if duplicates:
            raise SimulationMatrixError(f"Duplicate simulation matrix job ids: {duplicates}")


def load_simulation_matrix(path: str | Path) -> SimulationMatrix:
    """Load and validate a simulation matrix TOML file."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return SimulationMatrix.from_mapping(payload)


def validate_simulation_matrix(
    matrix_path: str | Path,
    *,
    grand_benchmark_plan_path: str | Path,
) -> SimulationMatrix:
    """Validate the matrix against the grand-benchmark simulation scenario list."""

    matrix = load_simulation_matrix(matrix_path)
    plan = load_grand_benchmark_plan(grand_benchmark_plan_path)
    _validate_matrix_plan_link(matrix, plan)
    return matrix


def summarize_simulation_matrix(matrix: SimulationMatrix) -> dict[str, Any]:
    """Return CI-friendly metadata counts for a validated simulation matrix."""

    status_counts: dict[str, int] = {}
    execution_mode_counts: dict[str, int] = {}
    for job in matrix.jobs:
        status_counts[job.status] = status_counts.get(job.status, 0) + 1
        execution_mode_counts[job.execution_mode] = execution_mode_counts.get(job.execution_mode, 0) + 1
    return {
        "checked_at": matrix.checked_at,
        "n_jobs": len(matrix.jobs),
        "job_status_counts": dict(sorted(status_counts.items())),
        "execution_mode_counts": dict(sorted(execution_mode_counts.items())),
        "certification_effect": matrix.certification_effect,
    }


def build_simulation_matrix_report(
    matrix_path: str | Path,
    *,
    grand_benchmark_plan_path: str | Path,
    checked_at: str,
) -> dict[str, Any]:
    """Run active simulation jobs and return a non-certifying report."""

    matrix = validate_simulation_matrix(
        matrix_path,
        grand_benchmark_plan_path=grand_benchmark_plan_path,
    )
    job_reports = [_run_job(job) for job in matrix.jobs if job.status == "active"]
    status = "passed" if all(report["status"] == "passed" for report in job_reports) else "failed"
    return {
        "schema_version": SIMULATION_MATRIX_REPORT_SCHEMA_VERSION,
        "status": status,
        "checked_at": checked_at,
        "matrix_checked_at": matrix.checked_at,
        "source_policy": matrix.source_policy,
        "uses_real_data": False,
        "certification_effect": "none",
        "n_jobs": len(job_reports),
        "jobs": job_reports,
        "limitations": [
            "Simulation reports test operating characteristics only.",
            "Simulation reports are not real clinical validation, tier-one parity, or superiority evidence.",
            "Simulation reports cannot enable clinical, regulatory, or HTA outputs.",
        ],
    }


def _validate_matrix_plan_link(matrix: SimulationMatrix, plan: GrandBenchmarkPlan) -> None:
    plan_ids = {scenario.id for scenario in plan.simulation_scenarios}
    missing = sorted({job.scenario_id for job in matrix.jobs} - plan_ids)
    if missing:
        raise SimulationMatrixError(
            f"simulation matrix references scenarios absent from the grand benchmark plan: {missing}"
        )
    if matrix.grand_benchmark_plan != "validation/grand_benchmark_plan.toml":
        raise SimulationMatrixError("simulation matrix must point at validation/grand_benchmark_plan.toml.")


def _run_job(job: SimulationMatrixJob) -> dict[str, Any]:
    start = time.perf_counter()
    result = run_benchmark_simulation(
        n_iterations=job.n_iterations,
        n_studies=job.n_studies,
        n_treatments=job.n_treatments,
        design_ratio=job.design_ratio,
        true_heterogeneity=job.true_heterogeneity,
        true_bias=job.true_bias,
        true_bias_interaction=job.true_bias_interaction,
        covariate_effect=job.covariate_effect,
        seed=job.seed,
    )
    runtime_seconds = time.perf_counter() - start
    _validate_job_output(job, result)
    return {
        "id": job.id,
        "scenario_id": job.scenario_id,
        "status": "passed",
        "execution_mode": job.execution_mode,
        "uses_real_data": False,
        "certification_effect": "none",
        "runtime_seconds": runtime_seconds,
        "inputs": _job_inputs(job),
        "iterations_successful": result["iterations_successful"],
        "methods_summary": result["methods_summary"],
        "claim_limit": job.claim_limit,
    }


def _validate_job_output(job: SimulationMatrixJob, result: dict[str, Any]) -> None:
    if int(result["iterations_successful"]) < job.min_successful_iterations:
        raise SimulationMatrixError(
            f"{job.id}: only {result['iterations_successful']} successful iterations; "
            f"required {job.min_successful_iterations}."
        )
    summary = result.get("methods_summary", {})
    missing_methods = sorted(set(job.required_methods) - set(summary))
    if missing_methods:
        raise SimulationMatrixError(f"{job.id}: missing required method summaries {missing_methods}.")
    for method in job.required_methods:
        metrics = summary[method]
        missing_metrics = sorted(set(job.required_metrics) - set(metrics))
        if missing_metrics:
            raise SimulationMatrixError(
                f"{job.id}: {method} missing required metrics {missing_metrics}."
            )


def _job_inputs(job: SimulationMatrixJob) -> dict[str, Any]:
    payload = asdict(job)
    for key in ("required_methods", "required_metrics"):
        payload[key] = list(payload[key])
    return payload
