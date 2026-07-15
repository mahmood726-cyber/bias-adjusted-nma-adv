"""Validation for the prespecified grand-benchmark plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.benchmark_registry import SourceBenchmarkRegistry


GRAND_BENCHMARK_PLAN_SCHEMA_VERSION = "grand_benchmark_plan/v1"
ALLOWED_EVIDENCE_SOURCES = {
    "clinicaltrials_gov",
    "pubmed_abstract",
    "open_access_paper",
}
ALLOWED_STATUSES = {"active", "planned"}
ALLOWED_EVIDENCE_CLASSES = {"real_source_backed", "simulation"}
ALLOWED_SOURCE_POLICIES = {
    "clinicaltrials_gov + pubmed_abstract only",
    "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
}
REQUIRED_SIMULATION_METRICS = {
    "bias",
    "rmse",
    "interval_coverage",
    "calibration",
    "convergence_failure",
    "runtime",
}


class GrandBenchmarkPlanError(ValueError):
    """Raised when the grand-benchmark plan is malformed or overclaims."""


@dataclass(frozen=True)
class RealDataLane:
    """One real source-backed benchmark lane."""

    id: str
    status: str
    evidence_class: str
    source_policy: str
    benchmark_registry_ids: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    claim_limit: str
    certification_effect: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "RealDataLane":
        missing = _missing_keys(
            raw,
            {
                "id",
                "status",
                "evidence_class",
                "source_policy",
                "benchmark_registry_ids",
                "required_artifacts",
                "claim_limit",
                "certification_effect",
            },
        )
        if missing:
            raise GrandBenchmarkPlanError(f"real_data_lane missing required keys: {missing}")
        lane = cls(
            id=str(raw["id"]),
            status=str(raw["status"]),
            evidence_class=str(raw["evidence_class"]),
            source_policy=str(raw["source_policy"]),
            benchmark_registry_ids=tuple(str(item) for item in raw["benchmark_registry_ids"]),
            required_artifacts=tuple(str(item) for item in raw["required_artifacts"]),
            claim_limit=str(raw["claim_limit"]),
            certification_effect=str(raw["certification_effect"]),
        )
        lane.validate()
        return lane

    def validate(self) -> None:
        if not self.id.strip():
            raise GrandBenchmarkPlanError("real_data_lane id must not be empty.")
        if self.status not in ALLOWED_STATUSES:
            raise GrandBenchmarkPlanError(f"{self.id}: unsupported lane status {self.status!r}.")
        if self.evidence_class != "real_source_backed":
            raise GrandBenchmarkPlanError(f"{self.id}: real-data lane must be real_source_backed.")
        if self.source_policy not in ALLOWED_SOURCE_POLICIES:
            raise GrandBenchmarkPlanError(f"{self.id}: unsupported source_policy.")
        if self.certification_effect != "none":
            raise GrandBenchmarkPlanError(f"{self.id}: real-data lanes cannot certify methods.")
        if not self.benchmark_registry_ids:
            raise GrandBenchmarkPlanError(f"{self.id}: benchmark_registry_ids must not be empty.")
        if not self.required_artifacts:
            raise GrandBenchmarkPlanError(f"{self.id}: required_artifacts must not be empty.")
        if "certification" not in self.claim_limit.lower() and "superiority" not in self.claim_limit.lower():
            raise GrandBenchmarkPlanError(f"{self.id}: claim_limit must state the claim boundary.")


@dataclass(frozen=True)
class SimulationScenario:
    """One simulation-only operating-characteristic scenario family."""

    id: str
    status: str
    evidence_class: str
    uses_real_data: bool
    outcome_families: tuple[str, ...]
    network_features: tuple[str, ...]
    design_features: tuple[str, ...]
    methods_under_test: tuple[str, ...]
    metrics: tuple[str, ...]
    claim_limit: str
    certification_effect: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SimulationScenario":
        missing = _missing_keys(
            raw,
            {
                "id",
                "status",
                "evidence_class",
                "uses_real_data",
                "outcome_families",
                "network_features",
                "design_features",
                "methods_under_test",
                "metrics",
                "claim_limit",
                "certification_effect",
            },
        )
        if missing:
            raise GrandBenchmarkPlanError(f"simulation_scenario missing required keys: {missing}")
        scenario = cls(
            id=str(raw["id"]),
            status=str(raw["status"]),
            evidence_class=str(raw["evidence_class"]),
            uses_real_data=bool(raw["uses_real_data"]),
            outcome_families=tuple(str(item) for item in raw["outcome_families"]),
            network_features=tuple(str(item) for item in raw["network_features"]),
            design_features=tuple(str(item) for item in raw["design_features"]),
            methods_under_test=tuple(str(item) for item in raw["methods_under_test"]),
            metrics=tuple(str(item) for item in raw["metrics"]),
            claim_limit=str(raw["claim_limit"]),
            certification_effect=str(raw["certification_effect"]),
        )
        scenario.validate()
        return scenario

    def validate(self) -> None:
        if not self.id.strip():
            raise GrandBenchmarkPlanError("simulation_scenario id must not be empty.")
        if self.status not in ALLOWED_STATUSES:
            raise GrandBenchmarkPlanError(f"{self.id}: unsupported scenario status {self.status!r}.")
        if self.evidence_class != "simulation":
            raise GrandBenchmarkPlanError(f"{self.id}: simulation scenario must use evidence_class=simulation.")
        if self.uses_real_data:
            raise GrandBenchmarkPlanError(f"{self.id}: simulation scenarios must not be labelled as real data.")
        if self.certification_effect != "none":
            raise GrandBenchmarkPlanError(f"{self.id}: simulations cannot certify methods.")
        if not self.outcome_families or not self.network_features or not self.design_features:
            raise GrandBenchmarkPlanError(f"{self.id}: scenario feature lists must not be empty.")
        if "tier_one_reference_targets" not in self.methods_under_test:
            raise GrandBenchmarkPlanError(f"{self.id}: simulations must include tier-one reference targets.")
        missing_metrics = sorted(REQUIRED_SIMULATION_METRICS - set(self.metrics))
        if missing_metrics:
            raise GrandBenchmarkPlanError(f"{self.id}: missing required metrics {missing_metrics}.")
        if "simulation" not in self.claim_limit.lower():
            raise GrandBenchmarkPlanError(f"{self.id}: claim_limit must state the simulation boundary.")


@dataclass(frozen=True)
class GrandBenchmarkPlan:
    """A prespecified plan separating real-data validation from simulation."""

    checked_at: str
    purpose: str
    allowed_evidence_sources: tuple[str, ...]
    certification_effect: str
    superiority_claim_policy: str
    real_data_lanes: tuple[RealDataLane, ...]
    simulation_scenarios: tuple[SimulationScenario, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "GrandBenchmarkPlan":
        missing = _missing_keys(
            raw,
            {
                "schema_version",
                "checked_at",
                "purpose",
                "allowed_evidence_sources",
                "certification_effect",
                "superiority_claim_policy",
                "real_data_lanes",
                "simulation_scenarios",
            },
        )
        if missing:
            raise GrandBenchmarkPlanError(f"grand benchmark plan missing required keys: {missing}")
        if raw["schema_version"] != GRAND_BENCHMARK_PLAN_SCHEMA_VERSION:
            raise GrandBenchmarkPlanError(
                f"schema_version must be {GRAND_BENCHMARK_PLAN_SCHEMA_VERSION}."
            )
        plan = cls(
            checked_at=str(raw["checked_at"]),
            purpose=str(raw["purpose"]),
            allowed_evidence_sources=tuple(str(item) for item in raw["allowed_evidence_sources"]),
            certification_effect=str(raw["certification_effect"]),
            superiority_claim_policy=str(raw["superiority_claim_policy"]),
            real_data_lanes=tuple(
                RealDataLane.from_mapping(item) for item in raw["real_data_lanes"]
            ),
            simulation_scenarios=tuple(
                SimulationScenario.from_mapping(item) for item in raw["simulation_scenarios"]
            ),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if set(self.allowed_evidence_sources) != ALLOWED_EVIDENCE_SOURCES:
            raise GrandBenchmarkPlanError("allowed_evidence_sources drifted.")
        if self.certification_effect != "none":
            raise GrandBenchmarkPlanError("grand benchmark plan cannot certify methods.")
        policy = self.superiority_claim_policy.lower()
        if "reference matching" not in policy or "machine-verifiable" not in policy:
            raise GrandBenchmarkPlanError("superiority_claim_policy must require reference matching and machine-verifiable artifacts.")
        _assert_unique("real_data_lanes", [lane.id for lane in self.real_data_lanes])
        _assert_unique("simulation_scenarios", [scenario.id for scenario in self.simulation_scenarios])
        if not self.real_data_lanes:
            raise GrandBenchmarkPlanError("grand benchmark plan must define real-data lanes.")
        if not self.simulation_scenarios:
            raise GrandBenchmarkPlanError("grand benchmark plan must define simulation scenarios.")


def load_grand_benchmark_plan(path: str | Path) -> GrandBenchmarkPlan:
    """Load and validate the grand-benchmark plan metadata."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return GrandBenchmarkPlan.from_mapping(payload)


def validate_grand_benchmark_plan(
    path: str | Path,
    *,
    source_registry: SourceBenchmarkRegistry,
) -> GrandBenchmarkPlan:
    """Validate plan metadata against the source-backed benchmark registry."""

    plan = load_grand_benchmark_plan(path)
    registered_ids = {entry.id for entry in source_registry.benchmarks}
    referenced_ids = {
        registry_id
        for lane in plan.real_data_lanes
        for registry_id in lane.benchmark_registry_ids
    }
    missing = sorted(referenced_ids - registered_ids)
    if missing:
        raise GrandBenchmarkPlanError(
            f"grand benchmark plan references unknown benchmark registry ids: {missing}"
        )
    return plan


def summarize_grand_benchmark_plan(plan: GrandBenchmarkPlan) -> dict[str, Any]:
    """Return CI-friendly counts from a validated grand-benchmark plan."""

    real_status_counts: dict[str, int] = {}
    scenario_status_counts: dict[str, int] = {}
    for lane in plan.real_data_lanes:
        real_status_counts[lane.status] = real_status_counts.get(lane.status, 0) + 1
    for scenario in plan.simulation_scenarios:
        scenario_status_counts[scenario.status] = scenario_status_counts.get(scenario.status, 0) + 1
    return {
        "checked_at": plan.checked_at,
        "n_real_data_lanes": len(plan.real_data_lanes),
        "real_data_lane_status_counts": dict(sorted(real_status_counts.items())),
        "n_simulation_scenarios": len(plan.simulation_scenarios),
        "simulation_scenario_status_counts": dict(sorted(scenario_status_counts.items())),
        "certification_effect": plan.certification_effect,
    }


def _missing_keys(raw: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - set(raw))


def _assert_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise GrandBenchmarkPlanError(f"Duplicate {label}: {duplicates}")
