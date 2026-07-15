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
    assert len(matrix.jobs) == 1

    job = matrix.jobs[0]
    assert job.id == "binary_sparse_multiarm_bias_grid_smoke"
    assert job.scenario_id == "binary_sparse_multiarm_bias_grid"
    assert job.status == "active"
    assert job.execution_mode == "smoke"
    assert job.uses_real_data is False
    assert job.certification_effect == "none"

    assert summarize_simulation_matrix(matrix) == {
        "checked_at": "2026-07-15",
        "n_jobs": 1,
        "job_status_counts": {"active": 1},
        "execution_mode_counts": {"smoke": 1},
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
    assert report["n_jobs"] == 1

    job = report["jobs"][0]
    assert job["status"] == "passed"
    assert job["inputs"]["n_iterations"] == 3
    assert job["iterations_successful"] >= 1
    assert "not real clinical validation" in job["claim_limit"]
    for method in job["inputs"]["required_methods"]:
        metrics = job["methods_summary"][method]
        for metric in job["inputs"]["required_metrics"]:
            assert metric in metrics


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
