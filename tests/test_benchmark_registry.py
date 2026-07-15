import copy
from pathlib import Path
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


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "validation" / "benchmark_registry.toml"


def test_source_benchmark_registry_validates_all_registered_artifacts():
    registry = validate_source_benchmark_registry(REGISTRY, repo_root=ROOT)

    assert registry.checked_at == "2026-07-15"
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
