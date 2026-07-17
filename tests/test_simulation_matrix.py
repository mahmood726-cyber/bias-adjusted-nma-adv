import copy
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.simulation_matrix import (
    SIMULATION_MATRIX_REPORT_SCHEMA_VERSION,
    SimulationMatrix,
    SimulationMatrixError,
    SimulationMatrixJob,
    build_simulation_matrix_report,
    summarize_simulation_matrix_report,
    validate_simulation_matrix_report,
    summarize_simulation_matrix,
    validate_simulation_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "validation" / "simulation_matrix.toml"
PLAN = ROOT / "validation" / "grand_benchmark_plan.toml"
SCRIPT = ROOT / "scripts" / "run_simulation_matrix.py"


def test_simulation_matrix_validates_against_grand_benchmark_plan():
    matrix = validate_simulation_matrix(MATRIX, grand_benchmark_plan_path=PLAN)

    assert matrix.source_policy == "simulation_only_no_real_data"
    assert matrix.uses_real_data is False
    assert matrix.certification_effect == "none"
    assert len(matrix.jobs) == 26

    job = matrix.jobs[0]
    assert job.id == "binary_sparse_multiarm_bias_grid_smoke"
    assert job.scenario_id == "binary_sparse_multiarm_bias_grid"
    assert job.status == "active"
    assert job.execution_mode == "smoke"
    assert job.uses_real_data is False
    assert job.certification_effect == "none"

    assert summarize_simulation_matrix(matrix) == {
        "checked_at": "2026-07-15",
        "n_jobs": 26,
        "job_status_counts": {"active": 26},
        "execution_mode_counts": {"full": 25, "smoke": 1},
        "certification_effect": "none",
    }


def test_simulation_matrix_report_runs_deterministic_smoke_job():
    report = build_simulation_matrix_report(
        MATRIX,
        grand_benchmark_plan_path=PLAN,
        checked_at="2026-07-15T00:00:00Z",
    )

    assert report["schema_version"] == SIMULATION_MATRIX_REPORT_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["uses_real_data"] is False
    assert report["certification_effect"] == "none"
    assert report["execution_modes_requested"] == ["smoke"]
    assert report["n_jobs"] == 1

    job = report["jobs"][0]
    assert job["status"] == "passed"
    assert job["execution_mode"] == "smoke"
    assert job["inputs"]["n_iterations"] == 3
    assert job["iterations_attempted"] == 3
    assert job["iterations_successful"] >= 1
    assert "not real clinical validation" in job["claim_limit"]
    for method in job["inputs"]["required_methods"]:
        metrics = job["methods_summary"][method]
        for metric in job["inputs"]["required_metrics"]:
            assert metric in metrics


def test_simulation_matrix_full_report_validates_against_matrix(tmp_path):
    payload = tomllib.loads(MATRIX.read_text(encoding="utf-8"))
    payload["jobs"] = [payload["jobs"][0]]
    payload["jobs"][0]["id"] = "binary_sparse_multiarm_bias_grid_full_test"
    payload["jobs"][0]["execution_mode"] = "full"
    payload["jobs"][0]["n_iterations"] = 1
    payload["jobs"][0]["min_successful_iterations"] = 1
    matrix_path = tmp_path / "simulation_matrix.toml"
    matrix_path.write_text(_to_toml_like(payload), encoding="utf-8")

    matrix = validate_simulation_matrix(matrix_path, grand_benchmark_plan_path=PLAN)
    report = build_simulation_matrix_report(
        matrix_path,
        grand_benchmark_plan_path=PLAN,
        checked_at="2026-07-15T00:00:00Z",
        execution_modes=("full",),
    )
    validated = validate_simulation_matrix_report(report, matrix)
    summary = summarize_simulation_matrix_report(validated)

    assert report["execution_modes_requested"] == ["full"]
    assert summary["full_validation_jobs"] == 1
    assert summary["full_validation_iterations_successful"] == 1

    drifted = dict(report)
    drifted["jobs"] = [dict(report["jobs"][0])]
    drifted["jobs"][0]["inputs"] = dict(drifted["jobs"][0]["inputs"])
    drifted["jobs"][0]["inputs"]["n_iterations"] = 2
    with pytest.raises(SimulationMatrixError, match="input drift"):
        validate_simulation_matrix_report(drifted, matrix)


def test_run_simulation_matrix_script_writes_json(tmp_path):
    output = tmp_path / "simulation_matrix_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(ROOT),
            "--checked-at",
            "2026-07-15T00:00:00Z",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "simulation matrix report written" in completed.stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SIMULATION_MATRIX_REPORT_SCHEMA_VERSION
    assert payload["status"] == "passed"
    assert payload["execution_modes_requested"] == ["smoke"]
    assert payload["uses_real_data"] is False
    assert payload["certification_effect"] == "none"


def test_simulation_matrix_rejects_real_data_and_certification_effect():
    payload = tomllib.loads(MATRIX.read_text(encoding="utf-8"))
    raw_job = copy.deepcopy(payload["jobs"][0])
    raw_job["uses_real_data"] = True
    with pytest.raises(SimulationMatrixError, match="must not use real data"):
        SimulationMatrixJob.from_mapping(raw_job)

    raw_job = copy.deepcopy(payload["jobs"][0])
    raw_job["certification_effect"] = "evidence_candidate"
    with pytest.raises(SimulationMatrixError, match="cannot certify methods"):
        SimulationMatrixJob.from_mapping(raw_job)

    payload["certification_effect"] = "simulation_validated"
    with pytest.raises(SimulationMatrixError, match="cannot certify methods"):
        SimulationMatrix.from_mapping(payload)


def test_simulation_matrix_rejects_unknown_plan_scenario(tmp_path):
    bad_matrix = MATRIX.read_text(encoding="utf-8").replace(
        'scenario_id = "binary_sparse_multiarm_bias_grid"',
        'scenario_id = "unknown_scenario"',
        1,
    )
    path = tmp_path / "bad_simulation_matrix.toml"
    path.write_text(bad_matrix, encoding="utf-8")

    with pytest.raises(SimulationMatrixError, match="unsupported executable scenario_id"):
        validate_simulation_matrix(path, grand_benchmark_plan_path=PLAN)


def _to_toml_like(payload: dict) -> str:
    lines = [
        f'schema_version = "{payload["schema_version"]}"',
        f'checked_at = "{payload["checked_at"]}"',
        f'purpose = "{payload["purpose"]}"',
        f'grand_benchmark_plan = "{payload["grand_benchmark_plan"]}"',
        f'source_policy = "{payload["source_policy"]}"',
        f'uses_real_data = {str(payload["uses_real_data"]).lower()}',
        f'certification_effect = "{payload["certification_effect"]}"',
        "",
    ]
    for job in payload["jobs"]:
        lines.extend(
            [
                "[[jobs]]",
                f'id = "{job["id"]}"',
                f'scenario_id = "{job["scenario_id"]}"',
                f'status = "{job["status"]}"',
                f'execution_mode = "{job["execution_mode"]}"',
                f'uses_real_data = {str(job["uses_real_data"]).lower()}',
                f'certification_effect = "{job["certification_effect"]}"',
                f'n_iterations = {job["n_iterations"]}',
                f'n_studies = {job["n_studies"]}',
                f'n_treatments = {job["n_treatments"]}',
                f'design_ratio = {job["design_ratio"]}',
                f'true_heterogeneity = {job["true_heterogeneity"]}',
                f'true_bias = {job["true_bias"]}',
                f'true_bias_interaction = {job["true_bias_interaction"]}',
                f'covariate_effect = {job["covariate_effect"]}',
                f'seed = {job["seed"]}',
                f'min_successful_iterations = {job["min_successful_iterations"]}',
                f"required_methods = {list(job['required_methods'])!r}",
                f"required_metrics = {list(job['required_metrics'])!r}",
                f'claim_limit = "{job["claim_limit"]}"',
                "",
            ]
        )
    return "\n".join(lines).replace("'", '"')
