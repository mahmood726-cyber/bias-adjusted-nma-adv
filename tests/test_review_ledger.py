import copy
from pathlib import Path

import pytest

from bias_nma_adv.review_ledger import (
    MULTIPERSON_REVIEW_SCHEMA_VERSION,
    REQUIRED_REVIEW_ROUNDS,
    ReviewLedger,
    ReviewLedgerError,
    load_review_ledger,
    summarize_review_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "validation" / "reviews" / "multiperson_review_2026_07_15.toml"


def test_multiperson_review_ledger_covers_required_rounds():
    ledger = load_review_ledger(LEDGER)

    assert ledger.schema_version == MULTIPERSON_REVIEW_SCHEMA_VERSION
    assert ledger.certification_effect == "none"
    assert set(ledger.required_review_rounds) >= REQUIRED_REVIEW_ROUNDS
    assert {round_.id for round_ in ledger.rounds} == REQUIRED_REVIEW_ROUNDS
    for round_ in ledger.rounds:
        assert round_.scope
        assert round_.findings
        assert round_.actions or round_.next_gate


def test_review_ledger_summary_is_validation_status_ready():
    summary = summarize_review_ledger(LEDGER)

    assert summary == {
        "schema_version": MULTIPERSON_REVIEW_SCHEMA_VERSION,
        "checked_at": "2026-07-17",
        "n_rounds": 4,
        "round_ids": [
            "source_boundary_review",
            "statistical_methods_review",
            "implementation_contract_review",
            "claims_governance_review",
        ],
        "status_counts": {"actioned": 2, "tracked_next_gate": 2},
        "certification_effect": "none",
    }


def test_review_ledger_rejects_certification_claims():
    raw = _ledger_to_mapping(load_review_ledger(LEDGER))
    raw["certification_effect"] = "production_certified"

    with pytest.raises(ReviewLedgerError, match="cannot certify"):
        ReviewLedger.from_mapping(raw)


def test_review_ledger_rejects_missing_required_round():
    raw = _ledger_to_mapping(load_review_ledger(LEDGER))
    raw["rounds"] = [item for item in raw["rounds"] if item["id"] != "claims_governance_review"]

    with pytest.raises(ReviewLedgerError, match="claims_governance_review"):
        ReviewLedger.from_mapping(raw)


def test_tracked_review_rounds_must_record_next_gate():
    raw = _ledger_to_mapping(load_review_ledger(LEDGER))
    for item in raw["rounds"]:
        if item["id"] == "claims_governance_review":
            item["next_gate"] = ""

    with pytest.raises(ReviewLedgerError, match="tracked reviews must record a next_gate"):
        ReviewLedger.from_mapping(raw)


def _ledger_to_mapping(ledger: ReviewLedger) -> dict[str, object]:
    return {
        "schema_version": ledger.schema_version,
        "checked_at": ledger.checked_at,
        "certification_effect": ledger.certification_effect,
        "source_policy": ledger.source_policy,
        "required_review_rounds": list(ledger.required_review_rounds),
        "thread_limit_note": ledger.thread_limit_note,
        "rounds": [
            {
                "id": round_.id,
                "reviewer": round_.reviewer,
                "status": round_.status,
                "scope": list(round_.scope),
                "findings": list(round_.findings),
                "actions": list(round_.actions),
                "next_gate": round_.next_gate,
            }
            for round_ in copy.deepcopy(ledger.rounds)
        ],
    }
