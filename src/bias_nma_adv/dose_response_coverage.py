"""Dose-response source-coverage gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES, PROTOCOL_ONLY_SOURCE_TYPES


DOSE_RESPONSE_COVERAGE_SCHEMA_VERSION = "dose_response_source_coverage/v1"


class DoseResponseCoverageError(ValueError):
    """Raised when the dose-response source coverage artifact is malformed."""


@dataclass(frozen=True)
class DoseResponseSourceCoverage:
    """Machine-readable status of source-backed dose-response data coverage."""

    checked_at: str
    status: str
    certification_effect: str
    purpose: str
    allowed_evidence_sources: tuple[str, ...]
    protocol_only_sources: tuple[str, ...]
    protocol_registry_rule: str
    registered_benchmark_ids: tuple[str, ...]
    registered_source_counts: dict[str, int]
    source_search_summary: tuple[str, ...]
    required_next_artifacts: tuple[str, ...]
    claim_limit: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DoseResponseSourceCoverage":
        required = {
            "schema_version",
            "checked_at",
            "status",
            "certification_effect",
            "purpose",
            "allowed_evidence_sources",
            "protocol_only_sources",
            "protocol_registry_rule",
            "registered_benchmark_ids",
            "registered_source_counts",
            "source_search_summary",
            "required_next_artifacts",
            "claim_limit",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise DoseResponseCoverageError(
                f"dose-response source coverage missing required keys: {missing}"
            )
        if raw["schema_version"] != DOSE_RESPONSE_COVERAGE_SCHEMA_VERSION:
            raise DoseResponseCoverageError(
                f"unsupported dose-response source coverage schema: {raw['schema_version']}"
            )

        coverage = cls(
            checked_at=str(raw["checked_at"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            purpose=str(raw["purpose"]),
            allowed_evidence_sources=tuple(
                str(item) for item in raw["allowed_evidence_sources"]
            ),
            protocol_only_sources=tuple(str(item) for item in raw["protocol_only_sources"]),
            protocol_registry_rule=str(raw["protocol_registry_rule"]),
            registered_benchmark_ids=tuple(str(item) for item in raw["registered_benchmark_ids"]),
            registered_source_counts={
                str(key): int(value) for key, value in raw["registered_source_counts"].items()
            },
            source_search_summary=tuple(str(item) for item in raw["source_search_summary"]),
            required_next_artifacts=tuple(str(item) for item in raw["required_next_artifacts"]),
            claim_limit=str(raw["claim_limit"]),
        )
        coverage.validate()
        return coverage

    def validate(self) -> None:
        if tuple(sorted(self.allowed_evidence_sources)) != tuple(
            sorted(EFFECT_EVIDENCE_SOURCE_TYPES)
        ):
            raise DoseResponseCoverageError(
                "dose-response coverage allowed_evidence_sources must match the project effect-source boundary."
            )
        if tuple(sorted(self.protocol_only_sources)) != tuple(
            sorted(PROTOCOL_ONLY_SOURCE_TYPES)
        ):
            raise DoseResponseCoverageError(
                "dose-response coverage protocol_only_sources must match the project protocol-only boundary."
            )
        if self.certification_effect != "none":
            raise DoseResponseCoverageError("dose-response coverage cannot certify a module.")
        if self.status not in {
            "missing_source_backed_dose_response_data",
            "active_source_backed_dose_response_data",
        }:
            raise DoseResponseCoverageError(
                f"unsupported dose-response coverage status '{self.status}'."
            )
        expected_source_keys = set(EFFECT_EVIDENCE_SOURCE_TYPES)
        if set(self.registered_source_counts) != expected_source_keys:
            raise DoseResponseCoverageError(
                "registered_source_counts must contain exactly the allowed effect-source keys."
            )
        if any(value < 0 for value in self.registered_source_counts.values()):
            raise DoseResponseCoverageError("registered_source_counts cannot be negative.")
        if self.status == "missing_source_backed_dose_response_data":
            if self.registered_benchmark_ids:
                raise DoseResponseCoverageError(
                    "missing dose-response coverage cannot list registered benchmark IDs."
                )
            if any(value != 0 for value in self.registered_source_counts.values()):
                raise DoseResponseCoverageError(
                    "missing dose-response coverage must have zero registered source counts."
                )
        if self.status == "active_source_backed_dose_response_data":
            if not self.registered_benchmark_ids:
                raise DoseResponseCoverageError(
                    "active dose-response coverage requires registered benchmark IDs."
                )
            if not any(value > 0 for value in self.registered_source_counts.values()):
                raise DoseResponseCoverageError(
                    "active dose-response coverage requires at least one source count."
                )
        if "cannot supply" not in self.protocol_registry_rule:
            raise DoseResponseCoverageError(
                "protocol_registry_rule must state that protocol-only registries cannot supply effects."
            )
        if "MBNMAdose_reference_run_before_certification" not in self.required_next_artifacts:
            raise DoseResponseCoverageError(
                "dose-response coverage must require MBNMAdose reference matching before certification."
            )
        if "No dose-response" not in self.claim_limit:
            raise DoseResponseCoverageError(
                "claim_limit must block unsupported dose-response validation claims."
            )


def load_dose_response_source_coverage(path: str | Path) -> DoseResponseSourceCoverage:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return DoseResponseSourceCoverage.from_mapping(raw)


def summarize_dose_response_source_coverage(
    coverage: DoseResponseSourceCoverage,
) -> dict[str, Any]:
    return {
        "schema_version": DOSE_RESPONSE_COVERAGE_SCHEMA_VERSION,
        "checked_at": coverage.checked_at,
        "status": coverage.status,
        "registered_benchmark_ids": list(coverage.registered_benchmark_ids),
        "registered_source_counts": dict(sorted(coverage.registered_source_counts.items())),
        "has_source_backed_dose_response_data": bool(coverage.registered_benchmark_ids),
        "required_next_artifacts": list(coverage.required_next_artifacts),
        "certification_effect": coverage.certification_effect,
    }
