"""Feature-parity matrix for tier-one evidence-synthesis comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


FEATURE_PARITY_MATRIX_SCHEMA_VERSION = "feature_parity_matrix/v1"

REQUIRED_FEATURE_IDS = {
    "pairwise_metafor_meta",
    "multiarm_netmeta_gls",
    "node_splitting_inconsistency",
    "netheat_contribution_visualization",
    "publication_bias_adjustments",
    "stan_nuts_multinma_bayesian_nma",
    "mlnmr_multinma",
    "dose_response_mbnmadose",
    "cross_design_crossnma",
    "component_nma_netmeta",
    "dta_bivariate_hsroc",
    "large_scale_validation",
}

ALLOWED_PARITY_STATUSES = {
    "planned",
    "local_implemented",
    "reference_candidate",
    "reference_matched",
    "blocking",
}


class FeatureParityMatrixError(ValueError):
    """Raised when feature-parity metadata is malformed or overclaims."""


@dataclass(frozen=True)
class FeatureParityItem:
    """One method feature mapped to mature reference tooling."""

    id: str
    domain: str
    reference_methods: tuple[str, ...]
    status: str
    evidence_artifacts: tuple[str, ...]
    required_next_artifacts: tuple[str, ...]
    claim_limit: str
    certification_effect: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "FeatureParityItem":
        required = {
            "id",
            "domain",
            "reference_methods",
            "status",
            "evidence_artifacts",
            "required_next_artifacts",
            "claim_limit",
            "certification_effect",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise FeatureParityMatrixError(f"feature-parity item missing keys: {missing}")
        item = cls(
            id=str(raw["id"]),
            domain=str(raw["domain"]),
            reference_methods=tuple(str(item) for item in raw["reference_methods"]),
            status=str(raw["status"]),
            evidence_artifacts=tuple(str(item) for item in raw["evidence_artifacts"]),
            required_next_artifacts=tuple(
                str(item) for item in raw["required_next_artifacts"]
            ),
            claim_limit=str(raw["claim_limit"]),
            certification_effect=str(raw["certification_effect"]),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.id not in REQUIRED_FEATURE_IDS:
            raise FeatureParityMatrixError(f"unsupported feature-parity id: {self.id}")
        if not self.domain.strip():
            raise FeatureParityMatrixError(f"{self.id}: domain must not be empty.")
        if not self.reference_methods:
            raise FeatureParityMatrixError(f"{self.id}: reference_methods must not be empty.")
        if self.status not in ALLOWED_PARITY_STATUSES:
            raise FeatureParityMatrixError(
                f"{self.id}: unsupported status {self.status!r}."
            )
        if self.certification_effect != "none":
            raise FeatureParityMatrixError(f"{self.id}: feature matrix cannot certify.")
        if self.status == "reference_matched" and not self.evidence_artifacts:
            raise FeatureParityMatrixError(
                f"{self.id}: reference_matched status requires evidence_artifacts."
            )
        if self.status != "reference_matched" and not self.required_next_artifacts:
            raise FeatureParityMatrixError(
                f"{self.id}: unresolved parity items require next artifacts."
            )
        claim_limit = self.claim_limit.lower()
        if "not" not in claim_limit and "block" not in claim_limit:
            raise FeatureParityMatrixError(
                f"{self.id}: claim_limit must state a non-claim or blocking rule."
            )


@dataclass(frozen=True)
class FeatureParityMatrix:
    """Validated feature-parity matrix."""

    checked_at: str
    purpose: str
    certification_effect: str
    global_feature_parity_complete: bool
    source_boundary: str
    items: tuple[FeatureParityItem, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "FeatureParityMatrix":
        required = {
            "schema_version",
            "checked_at",
            "purpose",
            "certification_effect",
            "global_feature_parity_complete",
            "source_boundary",
            "features",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise FeatureParityMatrixError(f"feature-parity matrix missing keys: {missing}")
        if raw["schema_version"] != FEATURE_PARITY_MATRIX_SCHEMA_VERSION:
            raise FeatureParityMatrixError(
                f"schema_version must be {FEATURE_PARITY_MATRIX_SCHEMA_VERSION}."
            )
        matrix = cls(
            checked_at=str(raw["checked_at"]),
            purpose=str(raw["purpose"]),
            certification_effect=str(raw["certification_effect"]),
            global_feature_parity_complete=bool(raw["global_feature_parity_complete"]),
            source_boundary=str(raw["source_boundary"]),
            items=tuple(FeatureParityItem.from_mapping(item) for item in raw["features"]),
        )
        matrix.validate()
        return matrix

    def validate(self) -> None:
        if self.certification_effect != "none":
            raise FeatureParityMatrixError("feature-parity matrix cannot certify methods.")
        if "ClinicalTrials.gov" not in self.source_boundary or "PubMed" not in self.source_boundary:
            raise FeatureParityMatrixError("source_boundary must preserve public-source limits.")
        item_ids = [item.id for item in self.items]
        missing = sorted(REQUIRED_FEATURE_IDS - set(item_ids))
        if missing:
            raise FeatureParityMatrixError(f"feature-parity matrix missing ids: {missing}")
        duplicates = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
        if duplicates:
            raise FeatureParityMatrixError(f"duplicate feature-parity ids: {duplicates}")
        if self.global_feature_parity_complete and any(
            item.status != "reference_matched" for item in self.items
        ):
            raise FeatureParityMatrixError(
                "global_feature_parity_complete requires every item to be reference_matched."
            )


def load_feature_parity_matrix(path: str | Path) -> FeatureParityMatrix:
    """Load and validate a feature-parity matrix TOML."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return FeatureParityMatrix.from_mapping(payload)


def summarize_feature_parity_matrix(matrix: FeatureParityMatrix) -> dict[str, Any]:
    """Return validation-status-friendly feature parity counts."""

    status_counts: dict[str, int] = {}
    for item in matrix.items:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    return {
        "schema_version": FEATURE_PARITY_MATRIX_SCHEMA_VERSION,
        "checked_at": matrix.checked_at,
        "n_features": len(matrix.items),
        "status_counts": dict(sorted(status_counts.items())),
        "reference_matched_ids": [
            item.id for item in matrix.items if item.status == "reference_matched"
        ],
        "blocking_ids": [
            item.id
            for item in matrix.items
            if item.status in {"planned", "blocking", "local_implemented", "reference_candidate"}
        ],
        "global_feature_parity_complete": matrix.global_feature_parity_complete,
        "certification_effect": matrix.certification_effect,
    }
