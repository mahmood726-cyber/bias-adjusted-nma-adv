from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs" / "review_artifact_policy.md"
DISCLAIMER = "Historical review artifact - not validation evidence."


def test_review_artifact_policy_marks_historical_reviews_as_nonevidence():
    text = POLICY.read_text(encoding="utf-8")

    for required in (
        "docs/adversarial_review*.md",
        "docs/journal_peer_review.md",
        "docs/methods_evaluation.md",
        "not validation evidence",
        "not source extraction evidence",
        "not claims that the current implementation is superior",
        "must be treated as a hypothesis",
        "machine-verifiable artifacts",
        "cannot upgrade certification status",
        "No current review note authorizes clinical or HTA reporting",
    ):
        assert required in text


def test_review_artifact_policy_covers_existing_historical_review_files():
    historical_reviews = [
        *sorted((ROOT / "docs").glob("adversarial_review*.md")),
        ROOT / "docs" / "journal_peer_review.md",
        ROOT / "docs" / "methods_evaluation.md",
    ]

    assert historical_reviews
    assert all(path.is_file() for path in historical_reviews)

    text = POLICY.read_text(encoding="utf-8")
    assert "docs/adversarial_review*.md" in text
    assert "docs/journal_peer_review.md" in text
    assert "docs/methods_evaluation.md" in text

    for path in historical_reviews:
        lead = path.read_text(encoding="utf-8")[:900]
        assert DISCLAIMER in lead
        assert "not source extraction evidence" in lead
        assert "not clinical guidance" in lead
        assert "machine-verifiable artifacts" in lead


def test_historical_review_files_do_not_start_with_unqualified_claims():
    historical_reviews = [
        *sorted((ROOT / "docs").glob("adversarial_review*.md")),
        ROOT / "docs" / "journal_peer_review.md",
        ROOT / "docs" / "methods_evaluation.md",
    ]

    for path in historical_reviews:
        lead = path.read_text(encoding="utf-8")[:300]
        assert lead.startswith("> **Historical review artifact - not validation evidence.**"), path.name
