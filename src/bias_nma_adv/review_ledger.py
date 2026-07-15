"""Machine-readable review ledger for evidence-synthesis hardening work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


MULTIPERSON_REVIEW_SCHEMA_VERSION = "multiperson_review/v1"
REQUIRED_REVIEW_ROUNDS = {
    "source_boundary_review",
    "statistical_methods_review",
    "implementation_contract_review",
    "claims_governance_review",
}
ALLOWED_REVIEW_STATUSES = {"actioned", "tracked_next_gate", "no_findings"}


class ReviewLedgerError(ValueError):
    """Raised when the multiperson review ledger is malformed."""


@dataclass(frozen=True)
class ReviewRound:
    """One bounded review pass and its disposition."""

    id: str
    reviewer: str
    status: str
    scope: tuple[str, ...]
    findings: tuple[str, ...]
    actions: tuple[str, ...]
    next_gate: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ReviewRound":
        required = {"id", "reviewer", "status", "scope", "findings", "actions", "next_gate"}
        missing = sorted(required - set(raw))
        if missing:
            raise ReviewLedgerError(f"review round missing required keys: {missing}")
        review = cls(
            id=str(raw["id"]),
            reviewer=str(raw["reviewer"]),
            status=str(raw["status"]),
            scope=tuple(str(item) for item in raw["scope"]),
            findings=tuple(str(item) for item in raw["findings"]),
            actions=tuple(str(item) for item in raw["actions"]),
            next_gate=str(raw["next_gate"]),
        )
        review.validate()
        return review

    def validate(self) -> None:
        if self.id not in REQUIRED_REVIEW_ROUNDS:
            raise ReviewLedgerError(f"unsupported review round id: {self.id}")
        if not self.reviewer.strip():
            raise ReviewLedgerError(f"{self.id}: reviewer must not be empty.")
        if self.status not in ALLOWED_REVIEW_STATUSES:
            raise ReviewLedgerError(f"{self.id}: unsupported status {self.status}.")
        if not self.scope:
            raise ReviewLedgerError(f"{self.id}: scope must not be empty.")
        if not self.findings:
            raise ReviewLedgerError(f"{self.id}: findings must not be empty.")
        if self.status == "actioned" and not self.actions:
            raise ReviewLedgerError(f"{self.id}: actioned reviews must record actions.")
        if self.status == "tracked_next_gate" and not self.next_gate.strip():
            raise ReviewLedgerError(f"{self.id}: tracked reviews must record a next_gate.")
        if self.status == "no_findings" and not self.next_gate.strip():
            raise ReviewLedgerError(f"{self.id}: no_findings reviews must still record the next gate.")


@dataclass(frozen=True)
class ReviewLedger:
    """A complete review ledger for one hardening milestone."""

    schema_version: str
    checked_at: str
    certification_effect: str
    source_policy: str
    required_review_rounds: tuple[str, ...]
    thread_limit_note: str
    rounds: tuple[ReviewRound, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ReviewLedger":
        required = {
            "schema_version",
            "checked_at",
            "certification_effect",
            "source_policy",
            "required_review_rounds",
            "thread_limit_note",
            "rounds",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ReviewLedgerError(f"review ledger missing required keys: {missing}")
        ledger = cls(
            schema_version=str(raw["schema_version"]),
            checked_at=str(raw["checked_at"]),
            certification_effect=str(raw["certification_effect"]),
            source_policy=str(raw["source_policy"]),
            required_review_rounds=tuple(str(item) for item in raw["required_review_rounds"]),
            thread_limit_note=str(raw["thread_limit_note"]),
            rounds=tuple(ReviewRound.from_mapping(item) for item in raw["rounds"]),
        )
        ledger.validate()
        return ledger

    def validate(self) -> None:
        if self.schema_version != MULTIPERSON_REVIEW_SCHEMA_VERSION:
            raise ReviewLedgerError(
                f"review ledger schema_version must be {MULTIPERSON_REVIEW_SCHEMA_VERSION}."
            )
        if not self.checked_at.strip():
            raise ReviewLedgerError("review ledger checked_at must not be empty.")
        if self.certification_effect != "none":
            raise ReviewLedgerError("review ledgers cannot certify model performance.")
        if "clinicaltrials.gov" not in self.source_policy.lower():
            raise ReviewLedgerError("review ledger source_policy must include ClinicalTrials.gov.")
        if "pubmed" not in self.source_policy.lower():
            raise ReviewLedgerError("review ledger source_policy must include PubMed.")
        if "open-access" not in self.source_policy.lower():
            raise ReviewLedgerError("review ledger source_policy must include open-access papers.")
        missing_rounds = sorted(REQUIRED_REVIEW_ROUNDS - set(self.required_review_rounds))
        if missing_rounds:
            raise ReviewLedgerError(f"review ledger missing required rounds: {missing_rounds}")
        round_ids = [round_.id for round_ in self.rounds]
        missing_reviewed_rounds = sorted(REQUIRED_REVIEW_ROUNDS - set(round_ids))
        if missing_reviewed_rounds:
            raise ReviewLedgerError(f"review ledger has no review entry for: {missing_reviewed_rounds}")
        duplicate_rounds = sorted({round_id for round_id in round_ids if round_ids.count(round_id) > 1})
        if duplicate_rounds:
            raise ReviewLedgerError(f"review ledger has duplicate review rounds: {duplicate_rounds}")


def load_review_ledger(path: str | Path) -> ReviewLedger:
    """Load and validate one multiperson review ledger."""

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return ReviewLedger.from_mapping(raw)


def summarize_review_ledger(path: str | Path) -> dict[str, Any]:
    """Return a compact validation-status summary for one review ledger."""

    ledger = load_review_ledger(path)
    return {
        "schema_version": ledger.schema_version,
        "checked_at": ledger.checked_at,
        "n_rounds": len(ledger.rounds),
        "round_ids": [round_.id for round_ in ledger.rounds],
        "status_counts": _counts(round_.status for round_ in ledger.rounds),
        "certification_effect": ledger.certification_effect,
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
