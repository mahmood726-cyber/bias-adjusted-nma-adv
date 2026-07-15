"""Machine-readable tier-one gap blocker registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


TIER1_GAP_REGISTER_SCHEMA_VERSION = "tier1_gap_register/v1"
REQUIRED_GAP_IDS = {
    "feature_completeness",
    "numerical_stability",
    "bayesian_ecosystem_integration",
}
REQUIRED_BLOCKED_CLAIMS = {
    "tier_one_parity",
    "tier_one_superiority",
    "production_certification",
    "clinical_reporting",
    "hta_reporting",
}
ALLOWED_GAP_STATUSES = {"blocking", "resolved"}


class Tier1GapRegisterError(ValueError):
    """Raised when tier-one gap governance is malformed or too permissive."""


@dataclass(frozen=True)
class Tier1Gap:
    """One blocker preventing tier-one parity or superiority claims."""

    id: str
    status: str
    review_source: str
    summary: str
    tier_one_references: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    required_evidence_artifacts: tuple[str, ...]
    claim_limit: str
    certification_effect: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Tier1Gap":
        required = {
            "id",
            "status",
            "review_source",
            "summary",
            "tier_one_references",
            "missing_capabilities",
            "required_evidence_artifacts",
            "claim_limit",
            "certification_effect",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise Tier1GapRegisterError(f"tier-one gap missing required keys: {missing}")
        gap = cls(
            id=str(raw["id"]),
            status=str(raw["status"]),
            review_source=str(raw["review_source"]),
            summary=str(raw["summary"]),
            tier_one_references=tuple(str(item) for item in raw["tier_one_references"]),
            missing_capabilities=tuple(str(item) for item in raw["missing_capabilities"]),
            required_evidence_artifacts=tuple(
                str(item) for item in raw["required_evidence_artifacts"]
            ),
            claim_limit=str(raw["claim_limit"]),
            certification_effect=str(raw["certification_effect"]),
        )
        gap.validate()
        return gap

    def validate(self) -> None:
        if self.id not in REQUIRED_GAP_IDS:
            raise Tier1GapRegisterError(f"unsupported tier-one gap id: {self.id}")
        if self.status not in ALLOWED_GAP_STATUSES:
            raise Tier1GapRegisterError(f"{self.id}: unsupported gap status {self.status}.")
        if self.status != "blocking":
            raise Tier1GapRegisterError(
                f"{self.id}: resolved gaps require separate promotion evidence before this register changes."
            )
        if self.certification_effect != "none":
            raise Tier1GapRegisterError(f"{self.id}: gaps cannot certify model performance.")
        if not self.review_source.strip():
            raise Tier1GapRegisterError(f"{self.id}: review_source must not be empty.")
        if not self.summary.strip():
            raise Tier1GapRegisterError(f"{self.id}: summary must not be empty.")
        if len(self.tier_one_references) < 2:
            raise Tier1GapRegisterError(f"{self.id}: at least two tier-one references are required.")
        if len(self.missing_capabilities) < 3:
            raise Tier1GapRegisterError(f"{self.id}: missing_capabilities is too thin.")
        if len(self.required_evidence_artifacts) < 3:
            raise Tier1GapRegisterError(f"{self.id}: required_evidence_artifacts is too thin.")
        claim_limit = self.claim_limit.lower()
        if "block" not in claim_limit and "cannot" not in claim_limit:
            raise Tier1GapRegisterError(f"{self.id}: claim_limit must state the blocking rule.")


@dataclass(frozen=True)
class Tier1GapRegister:
    """Complete tier-one gap governance register."""

    checked_at: str
    certification_effect: str
    purpose: str
    source_boundary: str
    superiority_claim_rule: str
    blocked_claims: tuple[str, ...]
    gaps: tuple[Tier1Gap, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Tier1GapRegister":
        required = {
            "schema_version",
            "checked_at",
            "certification_effect",
            "purpose",
            "source_boundary",
            "superiority_claim_rule",
            "blocked_claims",
            "gaps",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise Tier1GapRegisterError(f"tier-one gap register missing required keys: {missing}")
        if raw["schema_version"] != TIER1_GAP_REGISTER_SCHEMA_VERSION:
            raise Tier1GapRegisterError(
                f"schema_version must be {TIER1_GAP_REGISTER_SCHEMA_VERSION}."
            )
        register = cls(
            checked_at=str(raw["checked_at"]),
            certification_effect=str(raw["certification_effect"]),
            purpose=str(raw["purpose"]),
            source_boundary=str(raw["source_boundary"]),
            superiority_claim_rule=str(raw["superiority_claim_rule"]),
            blocked_claims=tuple(str(item) for item in raw["blocked_claims"]),
            gaps=tuple(Tier1Gap.from_mapping(item) for item in raw["gaps"]),
        )
        register.validate()
        return register

    def validate(self) -> None:
        if not self.checked_at.strip():
            raise Tier1GapRegisterError("tier-one gap register checked_at must not be empty.")
        if self.certification_effect != "none":
            raise Tier1GapRegisterError("tier-one gap register cannot certify model performance.")
        if "ClinicalTrials.gov" not in self.source_boundary or "PubMed" not in self.source_boundary:
            raise Tier1GapRegisterError("source_boundary must preserve public-source constraints.")
        if "cannot supply model-ready effects" not in self.source_boundary:
            raise Tier1GapRegisterError("source_boundary must preserve protocol-only registry limits.")
        if "while any gap remains blocking" not in self.superiority_claim_rule:
            raise Tier1GapRegisterError("superiority_claim_rule must fail closed while gaps block.")
        missing_claims = sorted(REQUIRED_BLOCKED_CLAIMS - set(self.blocked_claims))
        if missing_claims:
            raise Tier1GapRegisterError(f"blocked_claims missing required entries: {missing_claims}")
        gap_ids = [gap.id for gap in self.gaps]
        missing_gaps = sorted(REQUIRED_GAP_IDS - set(gap_ids))
        if missing_gaps:
            raise Tier1GapRegisterError(f"tier-one gap register missing required gaps: {missing_gaps}")
        duplicates = sorted({gap_id for gap_id in gap_ids if gap_ids.count(gap_id) > 1})
        if duplicates:
            raise Tier1GapRegisterError(f"duplicate tier-one gaps: {duplicates}")


def load_tier1_gap_register(path: str | Path) -> Tier1GapRegister:
    """Load and validate the tier-one gap register."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return Tier1GapRegister.from_mapping(payload)


def summarize_tier1_gap_register(register: Tier1GapRegister) -> dict[str, Any]:
    """Return compact counts for validation-status reports."""

    status_counts: dict[str, int] = {}
    for gap in register.gaps:
        status_counts[gap.status] = status_counts.get(gap.status, 0) + 1
    return {
        "schema_version": TIER1_GAP_REGISTER_SCHEMA_VERSION,
        "checked_at": register.checked_at,
        "n_gaps": len(register.gaps),
        "gap_ids": [gap.id for gap in register.gaps],
        "status_counts": dict(sorted(status_counts.items())),
        "blocked_claims": list(register.blocked_claims),
        "certification_effect": register.certification_effect,
    }
