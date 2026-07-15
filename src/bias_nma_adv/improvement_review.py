"""Milestone improvement review ledger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


IMPROVEMENT_REVIEW_SCHEMA_VERSION = "improvement_review/v1"
REQUIRED_IMPROVEMENT_REVIEW_IDS = {
    "statistical_tier1_blocker_review",
    "source_boundary_review",
    "html_delivery_regression_review",
    "implementation_polish_review",
}
ALLOWED_REVIEW_STATUSES = {"passed_current_milestone"}
ALLOWED_OVERALL_STATUS = "passed_current_milestone_with_global_goal_blockers"


class ImprovementReviewError(ValueError):
    """Raised when a milestone review ledger is malformed or overclaims."""


@dataclass(frozen=True)
class ImprovementReviewRound:
    """One role-specific milestone review round."""

    id: str
    reviewer: str
    status: str
    scope: tuple[str, ...]
    findings: tuple[str, ...]
    actions: tuple[str, ...]
    remaining_blockers: tuple[str, ...]
    verdict: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ImprovementReviewRound":
        required = {
            "id",
            "reviewer",
            "status",
            "scope",
            "findings",
            "actions",
            "remaining_blockers",
            "verdict",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ImprovementReviewError(f"improvement review round missing keys: {missing}")
        review = cls(
            id=str(raw["id"]),
            reviewer=str(raw["reviewer"]),
            status=str(raw["status"]),
            scope=tuple(str(item) for item in raw["scope"]),
            findings=tuple(str(item) for item in raw["findings"]),
            actions=tuple(str(item) for item in raw["actions"]),
            remaining_blockers=tuple(str(item) for item in raw["remaining_blockers"]),
            verdict=str(raw["verdict"]),
        )
        review.validate()
        return review

    def validate(self) -> None:
        if self.id not in REQUIRED_IMPROVEMENT_REVIEW_IDS:
            raise ImprovementReviewError(f"unsupported improvement review id: {self.id}")
        if not self.reviewer.strip():
            raise ImprovementReviewError(f"{self.id}: reviewer must not be empty.")
        if self.status not in ALLOWED_REVIEW_STATUSES:
            raise ImprovementReviewError(f"{self.id}: unsupported status {self.status}.")
        if not self.scope:
            raise ImprovementReviewError(f"{self.id}: scope must not be empty.")
        if not self.findings or not self.actions:
            raise ImprovementReviewError(f"{self.id}: findings and actions must not be empty.")
        if not self.remaining_blockers:
            raise ImprovementReviewError(f"{self.id}: remaining_blockers must not be empty.")
        if "global_goal" not in self.verdict:
            raise ImprovementReviewError(f"{self.id}: verdict must distinguish milestone from global goal.")


@dataclass(frozen=True)
class ImprovementReview:
    """Complete milestone review ledger."""

    checked_at: str
    certification_effect: str
    overall_status: str
    goal_statement: str
    thread_limit_note: str
    source_boundary: str
    global_goal_complete: bool
    rounds: tuple[ImprovementReviewRound, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ImprovementReview":
        required = {
            "schema_version",
            "checked_at",
            "certification_effect",
            "overall_status",
            "goal_statement",
            "thread_limit_note",
            "source_boundary",
            "global_goal_complete",
            "rounds",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ImprovementReviewError(f"improvement review missing keys: {missing}")
        if raw["schema_version"] != IMPROVEMENT_REVIEW_SCHEMA_VERSION:
            raise ImprovementReviewError(
                f"schema_version must be {IMPROVEMENT_REVIEW_SCHEMA_VERSION}."
            )
        review = cls(
            checked_at=str(raw["checked_at"]),
            certification_effect=str(raw["certification_effect"]),
            overall_status=str(raw["overall_status"]),
            goal_statement=str(raw["goal_statement"]),
            thread_limit_note=str(raw["thread_limit_note"]),
            source_boundary=str(raw["source_boundary"]),
            global_goal_complete=bool(raw["global_goal_complete"]),
            rounds=tuple(ImprovementReviewRound.from_mapping(item) for item in raw["rounds"]),
        )
        review.validate()
        return review

    def validate(self) -> None:
        if not self.checked_at.strip():
            raise ImprovementReviewError("improvement review checked_at must not be empty.")
        if self.certification_effect != "none":
            raise ImprovementReviewError("improvement review cannot certify model performance.")
        if self.overall_status != ALLOWED_OVERALL_STATUS:
            raise ImprovementReviewError("improvement review overall_status must keep global blockers visible.")
        if self.global_goal_complete:
            raise ImprovementReviewError("improvement review must not mark the global goal complete.")
        if "best" not in self.goal_statement.lower() and "world-class" not in self.goal_statement.lower():
            raise ImprovementReviewError("goal_statement must preserve the strategic ambition.")
        if "thread limit" not in self.thread_limit_note.lower():
            raise ImprovementReviewError("thread_limit_note must disclose sub-agent limitation.")
        if "cannot supply model-ready effects" not in self.source_boundary:
            raise ImprovementReviewError("source_boundary must preserve protocol-only registry limits.")
        round_ids = [round_.id for round_ in self.rounds]
        missing = sorted(REQUIRED_IMPROVEMENT_REVIEW_IDS - set(round_ids))
        if missing:
            raise ImprovementReviewError(f"improvement review missing rounds: {missing}")
        duplicates = sorted({round_id for round_id in round_ids if round_ids.count(round_id) > 1})
        if duplicates:
            raise ImprovementReviewError(f"duplicate improvement review rounds: {duplicates}")


def load_improvement_review(path: str | Path) -> ImprovementReview:
    """Load and validate one improvement review ledger."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return ImprovementReview.from_mapping(payload)


def summarize_improvement_review(review: ImprovementReview) -> dict[str, Any]:
    """Return compact validation-status fields."""

    status_counts: dict[str, int] = {}
    for round_ in review.rounds:
        status_counts[round_.status] = status_counts.get(round_.status, 0) + 1
    return {
        "schema_version": IMPROVEMENT_REVIEW_SCHEMA_VERSION,
        "checked_at": review.checked_at,
        "overall_status": review.overall_status,
        "n_rounds": len(review.rounds),
        "round_ids": [round_.id for round_ in review.rounds],
        "status_counts": dict(sorted(status_counts.items())),
        "global_goal_complete": review.global_goal_complete,
        "certification_effect": review.certification_effect,
    }
