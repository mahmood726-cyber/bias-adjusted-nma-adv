import copy
from pathlib import Path

import pytest

from bias_nma_adv.improvement_review import (
    ALLOWED_OVERALL_STATUS,
    IMPROVEMENT_REVIEW_SCHEMA_VERSION,
    REQUIRED_IMPROVEMENT_REVIEW_IDS,
    ImprovementReview,
    ImprovementReviewError,
    load_improvement_review,
    summarize_improvement_review,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "validation" / "reviews" / "improvement_review_2026_07_15.toml"


def test_improvement_review_passes_current_milestone_but_not_global_goal():
    review = load_improvement_review(REVIEW)

    assert review.certification_effect == "none"
    assert review.overall_status == ALLOWED_OVERALL_STATUS
    assert review.global_goal_complete is False
    assert "thread limit" in review.thread_limit_note.lower()
    assert "cannot supply model-ready effects" in review.source_boundary
    assert {round_.id for round_ in review.rounds} == REQUIRED_IMPROVEMENT_REVIEW_IDS
    assert {round_.status for round_ in review.rounds} == {"passed_current_milestone"}
    for round_ in review.rounds:
        assert round_.findings
        assert round_.actions
        assert round_.remaining_blockers
        assert "global_goal" in round_.verdict


def test_improvement_review_summary_is_validation_status_ready():
    summary = summarize_improvement_review(load_improvement_review(REVIEW))

    assert summary == {
        "schema_version": IMPROVEMENT_REVIEW_SCHEMA_VERSION,
        "checked_at": "2026-07-15",
        "overall_status": ALLOWED_OVERALL_STATUS,
        "n_rounds": 4,
        "round_ids": [
            "statistical_tier1_blocker_review",
            "source_boundary_review",
            "html_delivery_regression_review",
            "implementation_polish_review",
        ],
        "status_counts": {"passed_current_milestone": 4},
        "global_goal_complete": False,
        "certification_effect": "none",
    }


def test_improvement_review_rejects_global_goal_completion_or_missing_thread_limit():
    raw = _review_to_mapping(load_improvement_review(REVIEW))
    raw["global_goal_complete"] = True

    with pytest.raises(ImprovementReviewError, match="global goal complete"):
        ImprovementReview.from_mapping(raw)

    raw = _review_to_mapping(load_improvement_review(REVIEW))
    raw["thread_limit_note"] = "External reviewers ran."

    with pytest.raises(ImprovementReviewError, match="thread_limit_note"):
        ImprovementReview.from_mapping(raw)


def test_improvement_review_rejects_missing_round_or_certification():
    raw = _review_to_mapping(load_improvement_review(REVIEW))
    raw["certification_effect"] = "production_certified"

    with pytest.raises(ImprovementReviewError, match="cannot certify"):
        ImprovementReview.from_mapping(raw)

    raw = _review_to_mapping(load_improvement_review(REVIEW))
    raw["rounds"] = [
        item for item in raw["rounds"] if item["id"] != "html_delivery_regression_review"
    ]

    with pytest.raises(ImprovementReviewError, match="html_delivery_regression_review"):
        ImprovementReview.from_mapping(raw)


def _review_to_mapping(review: ImprovementReview) -> dict[str, object]:
    return {
        "schema_version": IMPROVEMENT_REVIEW_SCHEMA_VERSION,
        "checked_at": review.checked_at,
        "certification_effect": review.certification_effect,
        "overall_status": review.overall_status,
        "goal_statement": review.goal_statement,
        "thread_limit_note": review.thread_limit_note,
        "source_boundary": review.source_boundary,
        "global_goal_complete": review.global_goal_complete,
        "rounds": [
            {
                "id": round_.id,
                "reviewer": round_.reviewer,
                "status": round_.status,
                "scope": list(round_.scope),
                "findings": list(round_.findings),
                "actions": list(round_.actions),
                "remaining_blockers": list(round_.remaining_blockers),
                "verdict": round_.verdict,
            }
            for round_ in copy.deepcopy(review.rounds)
        ],
    }
