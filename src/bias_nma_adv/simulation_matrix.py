"""Executable simulation-matrix validation and smoke-run reporting."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
import tomllib
from typing import Any, Mapping

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
    execution_modes: tuple[str, ...] = ("smoke",),
) -> dict[str, Any]:
    """Run active simulation jobs and return a non-certifying report."""

    matrix = validate_simulation_matrix(
        matrix_path,
        grand_benchmark_plan_path=grand_benchmark_plan_path,
    )
    selected_modes = _normalise_execution_modes(execution_modes)
    job_reports = [
        _run_job(job)
        for job in matrix.jobs
        if job.status == "active" and job.execution_mode in selected_modes
    ]
    status = (
        "passed"
        if job_reports and all(report["status"] == "passed" for report in job_reports)
        else "failed"
    )
    return {
        "schema_version": SIMULATION_MATRIX_REPORT_SCHEMA_VERSION,
        "status": status,
        "checked_at": checked_at,
        "matrix_checked_at": matrix.checked_at,
        "source_policy": matrix.source_policy,
        "execution_modes_requested": sorted(selected_modes),
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


def load_simulation_matrix_report(path: str | Path) -> dict[str, Any]:
    """Load a simulation-matrix report JSON artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SimulationMatrixError("simulation matrix report must be a JSON object.")
    return payload


def validate_simulation_matrix_report(
    report: Mapping[str, Any],
    matrix: SimulationMatrix,
) -> dict[str, Any]:
    """Validate an executed simulation report against the prespecified matrix."""

    required = {
        "schema_version",
        "status",
        "source_policy",
        "execution_modes_requested",
        "uses_real_data",
        "certification_effect",
        "n_jobs",
        "jobs",
    }
    missing = sorted(required - set(report))
    if missing:
        raise SimulationMatrixError(f"simulation matrix report missing required keys: {missing}")
    if report["schema_version"] != SIMULATION_MATRIX_REPORT_SCHEMA_VERSION:
        raise SimulationMatrixError(
            f"simulation matrix report schema_version must be {SIMULATION_MATRIX_REPORT_SCHEMA_VERSION}."
        )
    if report["status"] != "passed":
        raise SimulationMatrixError("simulation matrix report must have status='passed'.")
    if report["source_policy"] != matrix.source_policy:
        raise SimulationMatrixError("simulation matrix report source_policy does not match matrix.")
    if bool(report["uses_real_data"]):
        raise SimulationMatrixError("simulation matrix report must not use real data.")
    if report["certification_effect"] != "none":
        raise SimulationMatrixError("simulation matrix report cannot certify methods.")
    jobs = report["jobs"]
    if not isinstance(jobs, list):
        raise SimulationMatrixError("simulation matrix report jobs must be a list.")
    if int(report["n_jobs"]) != len(jobs):
        raise SimulationMatrixError("simulation matrix report n_jobs does not match jobs length.")

    matrix_jobs = {job.id: job for job in matrix.jobs}
    seen: set[str] = set()
    for raw_job in jobs:
        if not isinstance(raw_job, Mapping):
            raise SimulationMatrixError("simulation matrix report job entries must be objects.")
        job_id = str(raw_job.get("id", ""))
        if not job_id:
            raise SimulationMatrixError("simulation matrix report job id must not be empty.")
        if job_id in seen:
            raise SimulationMatrixError(f"duplicate simulation matrix report job id: {job_id}.")
        seen.add(job_id)
        matrix_job = matrix_jobs.get(job_id)
        if matrix_job is None:
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} is absent from matrix.")
        if matrix_job.status != "active":
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} is not active in matrix.")
        if raw_job.get("status") != "passed":
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} did not pass.")
        if raw_job.get("execution_mode") != matrix_job.execution_mode:
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} execution_mode drift.")
        if raw_job.get("uses_real_data") is not False:
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} must not use real data.")
        if raw_job.get("certification_effect") != "none":
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} cannot certify methods.")
        if raw_job.get("inputs") != _job_inputs(matrix_job):
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} input drift.")
        if int(raw_job.get("iterations_successful", 0)) < matrix_job.min_successful_iterations:
            raise SimulationMatrixError(
                f"simulation matrix report job {job_id!r} has fewer successful iterations than required."
            )
        summary = raw_job.get("methods_summary", {})
        if not isinstance(summary, Mapping):
            raise SimulationMatrixError(f"simulation matrix report job {job_id!r} methods_summary must be an object.")
        missing_methods = sorted(set(matrix_job.required_methods) - set(summary))
        if missing_methods:
            raise SimulationMatrixError(
                f"simulation matrix report job {job_id!r} missing methods {missing_methods}."
            )
        for method in matrix_job.required_methods:
            metrics = summary[method]
            missing_metrics = sorted(set(matrix_job.required_metrics) - set(metrics))
            if missing_metrics:
                raise SimulationMatrixError(
                    f"simulation matrix report job {job_id!r} method {method!r} missing metrics {missing_metrics}."
                )

    return dict(report)


def summarize_simulation_matrix_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize passed simulation evidence from a validated report."""

    status_counts: dict[str, int] = {}
    execution_mode_counts: dict[str, int] = {}
    full_jobs = 0
    full_iterations_successful = 0
    for raw_job in report.get("jobs", []):
        status = str(raw_job.get("status", ""))
        mode = str(raw_job.get("execution_mode", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        execution_mode_counts[mode] = execution_mode_counts.get(mode, 0) + 1
        if status == "passed" and mode == "full":
            full_jobs += 1
            full_iterations_successful += int(raw_job.get("iterations_successful", 0))
    return {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "n_jobs": int(report["n_jobs"]),
        "job_status_counts": dict(sorted(status_counts.items())),
        "execution_mode_counts": dict(sorted(execution_mode_counts.items())),
        "full_validation_jobs": full_jobs,
        "full_validation_iterations_successful": full_iterations_successful,
        "uses_real_data": bool(report["uses_real_data"]),
        "certification_effect": str(report["certification_effect"]),
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
        "iterations_attempted": result["iterations_attempted"],
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


def _normalise_execution_modes(execution_modes: tuple[str, ...]) -> frozenset[str]:
    if not execution_modes:
        raise SimulationMatrixError("at least one execution mode must be requested.")
    raw_modes = tuple(mode.strip().lower() for mode in execution_modes)
    if "all" in raw_modes:
        return frozenset(ALLOWED_EXECUTION_MODES)
    unknown = sorted(set(raw_modes) - ALLOWED_EXECUTION_MODES)
    if unknown:
        raise SimulationMatrixError(f"unsupported execution modes requested: {unknown}")
    return frozenset(raw_modes)
