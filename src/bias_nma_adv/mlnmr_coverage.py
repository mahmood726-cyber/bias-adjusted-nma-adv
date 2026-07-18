"""ML-NMR source/model coverage gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES, PROTOCOL_ONLY_SOURCE_TYPES


MLNMR_SOURCE_COVERAGE_SCHEMA_VERSION = "mlnmr_source_coverage/v1"


class MLNMRCoverageError(ValueError):
    """Raised when the ML-NMR source coverage artifact is malformed or overclaims."""


@dataclass(frozen=True)
class MLNMRSourceCoverage:
    """Machine-readable status of public-source ML-NMR readiness."""

    checked_at: str
    status: str
    model_status: str
    certification_effect: str
    purpose: str
    allowed_evidence_sources: tuple[str, ...]
    protocol_only_sources: tuple[str, ...]
    protocol_registry_rule: str
    formal_source_boundary_decision: str
    formal_source_boundary_reason: str
    formal_decision_artifacts: tuple[str, ...]
    registered_benchmark_ids: tuple[str, ...]
    registered_source_counts: dict[str, int]
    required_source_components: tuple[str, ...]
    excluded_source_patterns: tuple[str, ...]
    source_search_summary: tuple[str, ...]
    required_next_artifacts: tuple[str, ...]
    claim_limit: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "MLNMRSourceCoverage":
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
            "formal_source_boundary_decision",
            "formal_source_boundary_reason",
            "formal_decision_artifacts",
            "registered_benchmark_ids",
            "registered_source_counts",
            "required_source_components",
            "excluded_source_patterns",
            "source_search_summary",
            "required_next_artifacts",
            "claim_limit",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise MLNMRCoverageError(f"ML-NMR source coverage missing required keys: {missing}")
        if raw["schema_version"] != MLNMR_SOURCE_COVERAGE_SCHEMA_VERSION:
            raise MLNMRCoverageError(
                f"unsupported ML-NMR source coverage schema: {raw['schema_version']}"
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
            formal_source_boundary_decision=str(raw["formal_source_boundary_decision"]),
            formal_source_boundary_reason=str(raw["formal_source_boundary_reason"]),
            formal_decision_artifacts=tuple(
                str(item) for item in raw["formal_decision_artifacts"]
            ),
            registered_benchmark_ids=tuple(str(item) for item in raw["registered_benchmark_ids"]),
            registered_source_counts={
                str(key): int(value) for key, value in raw["registered_source_counts"].items()
            },
            required_source_components=tuple(
                str(item) for item in raw["required_source_components"]
            ),
            excluded_source_patterns=tuple(str(item) for item in raw["excluded_source_patterns"]),
            source_search_summary=tuple(str(item) for item in raw["source_search_summary"]),
            required_next_artifacts=tuple(str(item) for item in raw["required_next_artifacts"]),
            claim_limit=str(raw["claim_limit"]),
        )
        coverage.validate()
        return coverage

    def validate(self) -> None:
        if tuple(sorted(self.allowed_evidence_sources)) != tuple(sorted(EFFECT_EVIDENCE_SOURCE_TYPES)):
            raise MLNMRCoverageError("ML-NMR allowed_evidence_sources drifted from the project boundary.")
        if tuple(sorted(self.protocol_only_sources)) != tuple(sorted(PROTOCOL_ONLY_SOURCE_TYPES)):
            raise MLNMRCoverageError("ML-NMR protocol_only_sources drifted from the project boundary.")
        if self.certification_effect != "none":
            raise MLNMRCoverageError("ML-NMR coverage cannot certify a module.")
        if self.status not in {
            "missing_source_backed_mlnmr_data",
            "active_source_backed_mlnmr_data",
        }:
            raise MLNMRCoverageError(f"unsupported ML-NMR source coverage status '{self.status}'.")
        if self.model_status not in {
            "not_implemented",
            "simulated_reference_only",
            "source_backed_not_reference_matched",
        }:
            raise MLNMRCoverageError(f"unsupported ML-NMR model_status '{self.model_status}'.")
        if set(self.registered_source_counts) != EFFECT_EVIDENCE_SOURCE_TYPES:
            raise MLNMRCoverageError(
                "ML-NMR registered_source_counts must contain exactly the effect-source keys."
            )
        if any(value < 0 for value in self.registered_source_counts.values()):
            raise MLNMRCoverageError("ML-NMR registered_source_counts cannot be negative.")
        if self.status == "missing_source_backed_mlnmr_data":
            if self.registered_benchmark_ids:
                raise MLNMRCoverageError("missing ML-NMR coverage cannot list benchmark IDs.")
            if any(value != 0 for value in self.registered_source_counts.values()):
                raise MLNMRCoverageError(
                    "missing ML-NMR coverage must have zero registered source counts."
                )
        if self.status == "active_source_backed_mlnmr_data" and not self.registered_benchmark_ids:
            raise MLNMRCoverageError("active ML-NMR coverage requires benchmark IDs.")
        required_components = {
            "public_trial_ipd_rows",
            "source_bound_aggregate_covariate_distribution",
            "shared_estimand_treatments_and_outcome",
            "covariate_overlap_diagnostics",
            "multinma_mlnmr_reference_run_before_certification",
        }
        if not required_components <= set(self.required_source_components):
            raise MLNMRCoverageError(
                "ML-NMR required_source_components must include public IPD, aggregate covariates, overlap diagnostics, and multinma reference matching."
            )
        excluded = " ".join(self.excluded_source_patterns).lower()
        for term in ("simulated", "synthetic", "pseudo-ipd", "survey"):
            if term not in excluded:
                raise MLNMRCoverageError(
                    "ML-NMR excluded_source_patterns must explicitly reject simulated, synthetic, pseudo-IPD, and survey-only shortcuts."
                )
        if "cannot supply" not in self.protocol_registry_rule:
            raise MLNMRCoverageError("ML-NMR protocol registry rule must block model-ready effects.")
        allowed_formal_decisions = {
            "no_formal_out_of_scope_decision",
            "real_mlnmr_domain_formally_out_of_scope_under_current_public_source_boundary",
        }
        if self.formal_source_boundary_decision not in allowed_formal_decisions:
            raise MLNMRCoverageError(
                "ML-NMR formal_source_boundary_decision is unsupported."
            )
        formal_text = (
            f"{self.formal_source_boundary_decision} "
            f"{self.formal_source_boundary_reason}"
        ).lower()
        if (
            self.formal_source_boundary_decision
            == "real_mlnmr_domain_formally_out_of_scope_under_current_public_source_boundary"
        ):
            for term in ("public", "ipd", "does not certify", "feature parity"):
                if term not in formal_text:
                    raise MLNMRCoverageError(
                        "formal ML-NMR source-boundary decisions must preserve public-IPD, non-certification, and feature-parity limits."
                    )
            if not self.formal_decision_artifacts:
                raise MLNMRCoverageError(
                    "formal ML-NMR source-boundary decisions require supporting artifacts."
                )
        elif self.formal_decision_artifacts:
            raise MLNMRCoverageError(
                "formal_decision_artifacts require a formal source-boundary decision."
            )
        if "No ML-NMR" not in self.claim_limit:
            raise MLNMRCoverageError("ML-NMR claim_limit must block unsupported ML-NMR claims.")

    def large_scale_domain_exclusions(self) -> dict[str, str]:
        """Return non-certifying real-domain exclusions for large-scale summaries."""

        if (
            self.formal_source_boundary_decision
            == "real_mlnmr_domain_formally_out_of_scope_under_current_public_source_boundary"
        ):
            return {"mlnmr": self.formal_source_boundary_reason}
        return {}


def load_mlnmr_source_coverage(path: str | Path) -> MLNMRSourceCoverage:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return MLNMRSourceCoverage.from_mapping(raw)


def summarize_mlnmr_source_coverage(coverage: MLNMRSourceCoverage) -> dict[str, Any]:
    return {
        "schema_version": MLNMR_SOURCE_COVERAGE_SCHEMA_VERSION,
        "checked_at": coverage.checked_at,
        "status": coverage.status,
        "model_status": coverage.model_status,
        "registered_benchmark_ids": list(coverage.registered_benchmark_ids),
        "registered_source_counts": dict(sorted(coverage.registered_source_counts.items())),
        "has_source_backed_mlnmr_data": bool(coverage.registered_benchmark_ids),
        "formal_source_boundary_decision": coverage.formal_source_boundary_decision,
        "formal_source_boundary_reason": coverage.formal_source_boundary_reason,
        "formal_decision_artifacts": list(coverage.formal_decision_artifacts),
        "has_formal_source_boundary_exclusion": bool(
            coverage.large_scale_domain_exclusions()
        ),
        "required_source_components": list(coverage.required_source_components),
        "excluded_source_patterns": list(coverage.excluded_source_patterns),
        "required_next_artifacts": list(coverage.required_next_artifacts),
        "certification_effect": coverage.certification_effect,
    }
