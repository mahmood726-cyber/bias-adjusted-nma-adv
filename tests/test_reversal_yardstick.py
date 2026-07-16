from pathlib import Path
import hashlib

import pytest

from bias_nma_adv.reversal_yardstick import (
    REVERSAL_YARDSTICK_SCHEMA_VERSION,
    ReversalYardstick,
    ReversalYardstickError,
    load_reversal_yardstick,
    summarize_reversal_yardstick,
)


ROOT = Path(__file__).resolve().parents[1]
YARDSTICK = ROOT / "validation" / "reversal_yardstick.toml"


def test_reversal_yardstick_preserves_detected_not_oracle_headline():
    yardstick = load_reversal_yardstick(YARDSTICK)

    assert yardstick.certification_effect == "none"
    assert yardstick.headline_metric == "detected_taint"
    assert yardstick.oracle_only_reporting_allowed is False
    assert yardstick.global_goal_complete is False
    assert yardstick.n_cases == 11
    assert yardstick.flag_caught == 4
    assert yardstick.flag_total == 5
    assert yardstick.recover_caught == 3
    assert yardstick.recover_total == 5
    assert yardstick.detected_vs_standard_dl.estimate == pytest.approx(0.5046818182)
    assert yardstick.oracle_vs_standard_dl.estimate == pytest.approx(0.7272727273)
    assert yardstick.detector_ci[0] <= 0.5 <= yardstick.detector_ci[1]
    assert yardstick.negative_control_status == "required_not_complete"


def test_reversal_yardstick_summary_is_validation_status_ready():
    summary = summarize_reversal_yardstick(load_reversal_yardstick(YARDSTICK))

    assert summary == {
        "schema_version": REVERSAL_YARDSTICK_SCHEMA_VERSION,
        "checked_at": "2026-07-16",
        "status": "registered_not_source_backed",
        "headline_metric": "detected_taint",
        "n_cases": 11,
        "priority_reversal_cases": 5,
        "flag_caught": 4,
        "flag_total": 5,
        "recover_caught": 3,
        "recover_total": 5,
        "oracle_winrate_vs_standard_dl": pytest.approx(0.7272727273),
        "detected_winrate_vs_standard_dl": pytest.approx(0.5046818182),
        "oracle_minus_detected_gap_meandist": pytest.approx(0.0745122177),
        "detector_status": "underpowered_not_validated",
        "negative_control_status": "required_not_complete",
        "global_goal_complete": False,
        "certification_effect": "none",
    }


def test_reversal_yardstick_rejects_oracle_only_or_superiority_shortcuts():
    raw = _yardstick_to_mapping(load_reversal_yardstick(YARDSTICK))
    raw["headline_metric"] = "oracle_taint"
    with pytest.raises(ReversalYardstickError, match="detected-taint"):
        ReversalYardstick.from_mapping(raw)

    raw = _yardstick_to_mapping(load_reversal_yardstick(YARDSTICK))
    raw["oracle_only_reporting_allowed"] = True
    with pytest.raises(ReversalYardstickError, match="oracle-only"):
        ReversalYardstick.from_mapping(raw)

    raw = _yardstick_to_mapping(load_reversal_yardstick(YARDSTICK))
    raw["negative_control_status"] = "complete"
    with pytest.raises(ReversalYardstickError, match="negative controls"):
        ReversalYardstick.from_mapping(raw)


def test_reversal_yardstick_verifies_source_artifact_pins_when_available(tmp_path):
    raw = _yardstick_to_mapping(load_reversal_yardstick(YARDSTICK))
    pinned = tmp_path / "fix.md"
    pinned.write_text("stable answer key", encoding="utf-8")
    raw["source_artifact_hashes"] = {
        "fix_md": hashlib.sha256(pinned.read_bytes()).hexdigest()
    }
    yardstick = ReversalYardstick.from_mapping(raw)

    report = yardstick.verify_source_artifact_pins({"fix_md": pinned})
    assert report["status"] == "verified"
    assert report["verified_artifacts"]["fix_md"] == yardstick.source_artifact_hashes["fix_md"]

    missing = tmp_path / "missing.md"
    unavailable = yardstick.verify_source_artifact_pins({"fix_md": missing})
    assert unavailable["status"] == "unavailable"
    assert unavailable["unavailable_artifacts"] == ("fix_md",)

    drifted = tmp_path / "fix.md"
    drifted.write_text("changed answer key", encoding="utf-8")
    with pytest.raises(ReversalYardstickError, match="hash drift"):
        yardstick.verify_source_artifact_pins({"fix_md": drifted})


def _yardstick_to_mapping(yardstick: ReversalYardstick) -> dict[str, object]:
    return {
        "schema_version": REVERSAL_YARDSTICK_SCHEMA_VERSION,
        "checked_at": yardstick.checked_at,
        "status": yardstick.status,
        "certification_effect": yardstick.certification_effect,
        "purpose": yardstick.purpose,
        "allowed_evidence_sources": list(yardstick.allowed_evidence_sources),
        "protocol_only_sources": list(yardstick.protocol_only_sources),
        "truth_boundary": yardstick.truth_boundary,
        "headline_metric": yardstick.headline_metric,
        "oracle_only_reporting_allowed": yardstick.oracle_only_reporting_allowed,
        "global_goal_complete": yardstick.global_goal_complete,
        "n_cases": yardstick.n_cases,
        "priority_reversal_cases": yardstick.priority_reversal_cases,
        "case_data_status": yardstick.case_data_status,
        "negative_control_status": yardstick.negative_control_status,
        "claim_limit": yardstick.claim_limit,
        "flag_recover": {
            "flag_caught": yardstick.flag_caught,
            "flag_total": yardstick.flag_total,
            "recover_caught": yardstick.recover_caught,
            "recover_total": yardstick.recover_total,
        },
        "detector": {
            "status": yardstick.detector_status,
            "auc": yardstick.detector_auc,
            "ci_lower": yardstick.detector_ci[0],
            "ci_upper": yardstick.detector_ci[1],
        },
        "oracle": {
            "vs_standard_dl": yardstick.oracle_vs_standard_dl.__dict__,
            "vs_strong_standard": yardstick.oracle_vs_strong_standard.__dict__,
        },
        "detected": {
            "vs_standard_dl": yardstick.detected_vs_standard_dl.__dict__,
            "vs_strong_standard": yardstick.detected_vs_strong_standard.__dict__,
        },
        "mean_distance": dict(yardstick.mean_distance),
        "source_artifact_hashes": dict(yardstick.source_artifact_hashes),
        "required_next_artifacts": list(yardstick.required_next_artifacts),
    }
