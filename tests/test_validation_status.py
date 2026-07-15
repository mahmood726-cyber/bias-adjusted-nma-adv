import json
from pathlib import Path
import subprocess
import sys

from bias_nma_adv.validation_status import (
    NO_PRODUCTION_CERTIFIED_MESSAGE,
    VALIDATION_STATUS_SCHEMA_VERSION,
    build_validation_status,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_validation_status.py"


def test_validation_status_composes_all_current_gates():
    report = build_validation_status(ROOT, checked_at="2026-07-15T00:00:00Z")

    assert report["schema_version"] == VALIDATION_STATUS_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["checked_at"] == "2026-07-15T00:00:00Z"
    assert report["allowed_evidence_sources"] == [
        "clinicaltrials_gov",
        "pubmed_abstract",
        "open_access_paper",
    ]
    assert report["certification_effect"] == "none"

    assert report["clinical_hta_reporting_enabled"] is False
    assert report["clinical_hta_reporting_reason"] == NO_PRODUCTION_CERTIFIED_MESSAGE
    assert report["production_certified_modules"] == []

    source_registry = report["source_benchmark_registry"]
    assert source_registry["registry"] == "validation/benchmark_registry.toml"
    assert source_registry["n_benchmarks"] == 4
    assert source_registry["certification_effect"] == "none"
    assert set(source_registry["benchmark_ids"]) == {
        "sglt2_hf_primary_log_or",
        "sglt2_hf_reported_hr",
        "pcsk9_mace_reported_hr",
        "t2d_mace_ctgov_hr_network",
    }

    reference_targets = report["reference_targets"]
    assert reference_targets["registry"] == "validation/reference_targets.toml"
    assert reference_targets["status_counts"] == {
        "planned": reference_targets["n_targets"],
    }
    assert reference_targets["production_certified_target_ids"] == []

    reference_runs = report["reference_runs"]
    assert reference_runs["directory"] == "validation/reference_runs"
    assert reference_runs["n_reports"] == 2
    assert reference_runs["status_counts"] == {"unavailable": 2}
    assert reference_runs["certification_candidate_artifacts"] == []
    assert {item["certification_effect"] for item in reference_runs["reports"]} == {"none"}


def test_write_validation_status_script_outputs_machine_readable_json(tmp_path):
    output = tmp_path / "validation_status.json"
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
    assert "validation status written" in completed.stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == VALIDATION_STATUS_SCHEMA_VERSION
    assert payload["status"] == "passed"
    assert payload["checked_at"] == "2026-07-15T00:00:00Z"
    assert payload["clinical_hta_reporting_enabled"] is False
    assert payload["source_benchmark_registry"]["n_benchmarks"] == 4
