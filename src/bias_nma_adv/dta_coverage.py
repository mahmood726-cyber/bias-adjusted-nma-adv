"""Diagnostic test accuracy source/model coverage gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES, PROTOCOL_ONLY_SOURCE_TYPES


DTA_SOURCE_COVERAGE_SCHEMA_VERSION = "dta_source_coverage/v1"


class DTACoverageError(ValueError):
    """Raised when the DTA coverage artifact is malformed or overclaims."""


@dataclass(frozen=True)
class DTASourceCoverage:
    """Machine-readable status of DTA data and model readiness."""

    checked_at: str
    status: str
    model_status: str
    certification_effect: str
    purpose: str
    allowed_evidence_sources: tuple[str, ...]
    protocol_only_sources: tuple[str, ...]
    protocol_registry_rule: str
    registered_benchmark_ids: tuple[str, ...]
    registered_source_counts: dict[str, int]
    required_source_fields: tuple[str, ...]
    required_model_families: tuple[str, ...]
    source_search_summary: tuple[str, ...]
    required_next_artifacts: tuple[str, ...]
    claim_limit: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DTASourceCoverage":
        required = {
            "schema_version",
            "checked_at",
            "status",
            "model_status",
            "certification_effect",
            "purpose",
            "allowed_evidence_sources",
            "protocol_only_sources",
            "protocol_registry_rule",
            "registered_benchmark_ids",
            "registered_source_counts",
            "required_source_fields",
            "required_model_families",
            "source_search_summary",
            "required_next_artifacts",
            "claim_limit",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise DTACoverageError(f"DTA source coverage missing required keys: {missing}")
        if raw["schema_version"] != DTA_SOURCE_COVERAGE_SCHEMA_VERSION:
            raise DTACoverageError(
                f"unsupported DTA source coverage schema: {raw['schema_version']}"
            )
        coverage = cls(
            checked_at=str(raw["checked_at"]),
            status=str(raw["status"]),
            model_status=str(raw["model_status"]),
            certification_effect=str(raw["certification_effect"]),
            purpose=str(raw["purpose"]),
            allowed_evidence_sources=tuple(str(item) for item in raw["allowed_evidence_sources"]),
            protocol_only_sources=tuple(str(item) for item in raw["protocol_only_sources"]),
            protocol_registry_rule=str(raw["protocol_registry_rule"]),
            registered_benchmark_ids=tuple(str(item) for item in raw["registered_benchmark_ids"]),
            registered_source_counts={
                str(key): int(value) for key, value in raw["registered_source_counts"].items()
            },
            required_source_fields=tuple(str(item) for item in raw["required_source_fields"]),
            required_model_families=tuple(str(item) for item in raw["required_model_families"]),
            source_search_summary=tuple(str(item) for item in raw["source_search_summary"]),
            required_next_artifacts=tuple(str(item) for item in raw["required_next_artifacts"]),
            claim_limit=str(raw["claim_limit"]),
        )
        coverage.validate()
        return coverage

    def validate(self) -> None:
        if tuple(sorted(self.allowed_evidence_sources)) != tuple(sorted(EFFECT_EVIDENCE_SOURCE_TYPES)):
            raise DTACoverageError("DTA allowed_evidence_sources drifted from the project boundary.")
        if tuple(sorted(self.protocol_only_sources)) != tuple(sorted(PROTOCOL_ONLY_SOURCE_TYPES)):
            raise DTACoverageError("DTA protocol_only_sources drifted from the project boundary.")
        if self.certification_effect != "none":
            raise DTACoverageError("DTA coverage cannot certify a module.")
        if self.status not in {"missing_source_backed_dta_data", "active_source_backed_dta_data"}:
            raise DTACoverageError(f"unsupported DTA source coverage status '{self.status}'.")
        if self.model_status not in {
            "not_implemented",
            "prototype_not_source_backed",
            "source_backed_not_reference_matched",
        }:
            raise DTACoverageError(f"unsupported DTA model_status '{self.model_status}'.")
        if set(self.registered_source_counts) != EFFECT_EVIDENCE_SOURCE_TYPES:
            raise DTACoverageError("DTA registered_source_counts must contain exactly the effect-source keys.")
        if any(value < 0 for value in self.registered_source_counts.values()):
            raise DTACoverageError("DTA registered_source_counts cannot be negative.")
        if self.status == "missing_source_backed_dta_data":
            if self.registered_benchmark_ids:
                raise DTACoverageError("missing DTA coverage cannot list benchmark IDs.")
            if any(value != 0 for value in self.registered_source_counts.values()):
                raise DTACoverageError("missing DTA coverage must have zero registered source counts.")
        if self.status == "active_source_backed_dta_data" and not self.registered_benchmark_ids:
            raise DTACoverageError("active DTA coverage requires benchmark IDs.")
        required_fields = {"tp", "fp", "fn", "tn", "index_test", "reference_standard"}
        if not required_fields <= set(self.required_source_fields):
            raise DTACoverageError("DTA required_source_fields must include the 2x2 table and test labels.")
        required_models = {"bivariate_random_effects_glmm", "hsroc"}
        if not required_models <= set(self.required_model_families):
            raise DTACoverageError("DTA required_model_families must include bivariate GLMM and HSROC.")
        if "cannot supply" not in self.protocol_registry_rule:
            raise DTACoverageError("DTA protocol registry rule must block model-ready effects.")
        if "No DTA" not in self.claim_limit:
            raise DTACoverageError("DTA claim_limit must block unsupported DTA claims.")


def load_dta_source_coverage(path: str | Path) -> DTASourceCoverage:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return DTASourceCoverage.from_mapping(raw)


def summarize_dta_source_coverage(coverage: DTASourceCoverage) -> dict[str, Any]:
    return {
        "schema_version": DTA_SOURCE_COVERAGE_SCHEMA_VERSION,
        "checked_at": coverage.checked_at,
        "status": coverage.status,
        "model_status": coverage.model_status,
        "registered_benchmark_ids": list(coverage.registered_benchmark_ids),
        "registered_source_counts": dict(sorted(coverage.registered_source_counts.items())),
        "has_source_backed_dta_data": bool(coverage.registered_benchmark_ids),
        "required_model_families": list(coverage.required_model_families),
        "required_next_artifacts": list(coverage.required_next_artifacts),
        "certification_effect": coverage.certification_effect,
    }
