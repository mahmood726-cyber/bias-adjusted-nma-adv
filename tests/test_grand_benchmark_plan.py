import copy
from pathlib import Path
import tomllib

import pytest

from bias_nma_adv.benchmark_registry import validate_source_benchmark_registry
from bias_nma_adv.grand_benchmark_plan import (
    GrandBenchmarkPlan,
    GrandBenchmarkPlanError,
    SimulationScenario,
    load_grand_benchmark_plan,
    summarize_grand_benchmark_plan,
    validate_grand_benchmark_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "validation" / "grand_benchmark_plan.toml"
REGISTRY = ROOT / "validation" / "benchmark_registry.toml"


def test_grand_benchmark_plan_validates_against_source_registry():
    registry = validate_source_benchmark_registry(REGISTRY, repo_root=ROOT)
    plan = validate_grand_benchmark_plan(PLAN, source_registry=registry)

    assert plan.certification_effect == "none"
    assert set(plan.allowed_evidence_sources) == {
        "clinicaltrials_gov",
        "pubmed_abstract",
        "open_access_paper",
    }
    assert {lane.id for lane in plan.real_data_lanes} == {
        "source_backed_binary_pairwise_meta",
        "source_backed_reported_survival_hr_pairwise",
        "source_backed_ctgov_reported_hr_network",
    }
    assert {scenario.evidence_class for scenario in plan.simulation_scenarios} == {"simulation"}
    assert {scenario.uses_real_data for scenario in plan.simulation_scenarios} == {False}

    summary = summarize_grand_benchmark_plan(plan)
    assert summary == {
        "checked_at": "2026-07-15",
        "n_real_data_lanes": 3,
        "real_data_lane_status_counts": {"active": 3},
        "n_simulation_scenarios": 3,
        "simulation_scenario_status_counts": {"planned": 3},
        "certification_effect": "none",
    }


def test_grand_benchmark_plan_rejects_unknown_registry_id(tmp_path):
    registry = validate_source_benchmark_registry(REGISTRY, repo_root=ROOT)
    bad_plan = PLAN.read_text(encoding="utf-8").replace(
        'benchmark_registry_ids = ["sglt2_hf_primary_log_or"]',
        'benchmark_registry_ids = ["missing_benchmark"]',
        1,
    )
    tmp_plan = tmp_path / "bad_grand_benchmark_plan.toml"
    tmp_plan.write_text(bad_plan, encoding="utf-8")

    with pytest.raises(GrandBenchmarkPlanError, match="unknown benchmark registry ids"):
        validate_grand_benchmark_plan(tmp_plan, source_registry=registry)


def test_grand_benchmark_plan_rejects_real_data_certification_effect():
    payload = tomllib.loads(PLAN.read_text(encoding="utf-8"))
    raw = copy.deepcopy(payload["real_data_lanes"][0])
    raw["certification_effect"] = "evidence_candidate"

    with pytest.raises(GrandBenchmarkPlanError, match="cannot certify methods"):
        GrandBenchmarkPlan.from_mapping({**payload, "real_data_lanes": [raw]})


def test_simulation_scenario_rejects_real_data_label_and_missing_metrics():
    payload = tomllib.loads(PLAN.read_text(encoding="utf-8"))
    raw = copy.deepcopy(payload["simulation_scenarios"][0])
    raw["uses_real_data"] = True

    with pytest.raises(GrandBenchmarkPlanError, match="must not be labelled as real data"):
        SimulationScenario.from_mapping(raw)

    raw = copy.deepcopy(payload["simulation_scenarios"][0])
    raw["metrics"] = ["bias", "rmse"]
    with pytest.raises(GrandBenchmarkPlanError, match="missing required metrics"):
        SimulationScenario.from_mapping(raw)


def test_grand_benchmark_plan_loader_rejects_source_policy_drift(tmp_path):
    # Minimal TOML writer for this drift test avoids adding a runtime dependency.
    bad_plan = tmp_path / "bad_plan.toml"
    bad_plan.write_text(
        PLAN.read_text(encoding="utf-8").replace(
            'allowed_evidence_sources = ["clinicaltrials_gov", "pubmed_abstract", "open_access_paper"]',
            'allowed_evidence_sources = ["pubmed_abstract"]',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrandBenchmarkPlanError, match="allowed_evidence_sources drifted"):
        load_grand_benchmark_plan(bad_plan)
