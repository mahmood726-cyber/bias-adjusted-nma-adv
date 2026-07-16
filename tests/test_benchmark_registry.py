import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.benchmark_registry import (
    BenchmarkRegistryError,
    SourceBenchmarkEntry,
    SourceBenchmarkRegistry,
    assert_registry_covers_source_backed_artifacts,
    discover_source_backed_benchmark_artifacts,
    load_source_benchmark_registry,
    validate_source_benchmark_entry,
    validate_source_benchmark_registry,
)
from bias_nma_adv.real_meta import sha256_file


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "validation" / "benchmark_registry.toml"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_benchmark_registry.py"


def test_source_benchmark_registry_validates_all_registered_artifacts():
    registry = validate_source_benchmark_registry(REGISTRY, repo_root=ROOT)

    assert registry.checked_at == "2026-07-16"
    assert set(registry.allowed_evidence_sources) == {
        "clinicaltrials_gov",
        "pubmed_abstract",
        "open_access_paper",
    }
    assert {entry.id for entry in registry.benchmarks} == {
        "sglt2_hf_primary_log_or",
        "sglt2_hf_reported_hr",
        "pcsk9_mace_reported_hr",
        "t2d_mace_ctgov_hr_network",
        "semaglutide_obesity_dose_response",
    }
    for entry in registry.benchmarks:
        assert entry.certification_effect == "none"
        assert entry.certification_scope == "local_source_verified_only"
        assert entry.source_checks
        assert entry.required_limitations


def test_source_benchmark_registry_covers_every_source_backed_benchmark_artifact():
    registry = load_source_benchmark_registry(REGISTRY)

    assert discover_source_backed_benchmark_artifacts(ROOT) == (
        "validation/real_meta/sglt2_hf_primary_benchmark.toml",
        "validation/survival/pcsk9_mace_reported_hr_benchmark.toml",
        "validation/survival/sglt2_hf_reported_hr_benchmark.toml",
        "validation/networks/t2d_mace_ctgov_hr_network_benchmark.toml",
        "validation/dose_response/semaglutide_obesity_dose_response_benchmark.toml",
    )
    assert_registry_covers_source_backed_artifacts(registry, repo_root=ROOT)


def test_source_benchmark_registry_rejects_certification_effect():
    payload = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    raw = copy.deepcopy(payload["benchmarks"][0])
    raw["certification_effect"] = "evidence_candidate"

    with pytest.raises(BenchmarkRegistryError, match="cannot certify model performance"):
        SourceBenchmarkEntry.from_mapping(raw)


def test_source_benchmark_registry_rejects_hash_drift():
    registry = load_source_benchmark_registry(REGISTRY)
    entry = registry.benchmarks[0]
    raw = copy.deepcopy(entry.__dict__)
    raw["artifact_sha256"] = "a" * 64
    drifted = SourceBenchmarkEntry.from_mapping(raw)

    with pytest.raises(BenchmarkRegistryError, match="SHA-256 mismatch"):
        validate_source_benchmark_entry(drifted, repo_root=ROOT)


def test_source_benchmark_registry_rejects_unregistered_artifact():
    payload = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["benchmarks"] = payload["benchmarks"][:-1]
    registry = SourceBenchmarkRegistry.from_mapping(payload)

    with pytest.raises(BenchmarkRegistryError, match="Unregistered source-backed benchmark artifacts"):
        assert_registry_covers_source_backed_artifacts(registry, repo_root=ROOT)


def test_source_benchmark_registry_rejects_missing_required_limitation():
    registry = load_source_benchmark_registry(REGISTRY)
    entry = registry.benchmarks[-1]
    raw = copy.deepcopy(entry.__dict__)
    raw["required_limitations"] = ("this term is absent",)
    bad = SourceBenchmarkEntry.from_mapping(raw)

    with pytest.raises(BenchmarkRegistryError, match="limitations missing"):
        validate_source_benchmark_entry(bad, repo_root=ROOT)


def test_validate_benchmark_registry_script_emits_machine_readable_summary():
    completed = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--root", str(ROOT), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "passed"
    assert payload["certification_effect"] == "none"
    assert payload["registry"] == "validation/benchmark_registry.toml"
    assert payload["n_benchmarks"] == 5
    assert set(payload["benchmark_ids"]) == {
        "sglt2_hf_primary_log_or",
        "sglt2_hf_reported_hr",
        "pcsk9_mace_reported_hr",
        "t2d_mace_ctgov_hr_network",
        "semaglutide_obesity_dose_response",
    }


def test_source_benchmark_registry_rejects_malformed_source_check_even_when_hash_matches(tmp_path):
    registry = load_source_benchmark_registry(REGISTRY)
    entry = next(item for item in registry.benchmarks if item.id == "sglt2_hf_primary_log_or")
    _copy_entry_files(tmp_path, entry)

    report_relpath = "validation/source_checks/sglt2_hf_primary_event_counts.json"
    report_path = tmp_path / report_relpath
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["records"][0]["active_count_token_found"] = False
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    raw = copy.deepcopy(entry.__dict__)
    raw["source_check_sha256"][report_relpath] = sha256_file(report_path)
    drifted = SourceBenchmarkEntry.from_mapping(raw)

    with pytest.raises(BenchmarkRegistryError, match="specialized source-check validation failed"):
        validate_source_benchmark_entry(drifted, repo_root=tmp_path)


def test_source_benchmark_registry_rejects_source_check_manifest_hash_drift(tmp_path):
    registry = load_source_benchmark_registry(REGISTRY)
    entry = next(item for item in registry.benchmarks if item.id == "sglt2_hf_primary_log_or")
    _copy_entry_files(tmp_path, entry)

    report_relpath = "validation/source_checks/sglt2_hf_primary_source_check.json"
    report_path = tmp_path / report_relpath
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["source_manifest_sha256"] = "a" * 64
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    raw = copy.deepcopy(entry.__dict__)
    raw["source_check_sha256"][report_relpath] = sha256_file(report_path)
    drifted = SourceBenchmarkEntry.from_mapping(raw)

    with pytest.raises(BenchmarkRegistryError, match="source verification SHA-256 mismatch"):
        validate_source_benchmark_entry(drifted, repo_root=tmp_path)


def _copy_entry_files(root: Path, entry: SourceBenchmarkEntry) -> None:
    for relpath in (entry.artifact_path, *entry.source_manifests, *entry.source_checks):
        src = ROOT / relpath
        dst = root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
