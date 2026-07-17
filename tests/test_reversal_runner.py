import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bias_nma_adv.reversal_runner import (
    REVERSAL_ARENA_STATUS,
    ReversalArenaError,
    run_reversal_arena_aggregate,
)
from bias_nma_adv.reversal_yardstick import (
    REVERSAL_YARDSTICK_SCHEMA_VERSION,
    ReversalYardstick,
    load_reversal_yardstick,
)


ROOT = Path(__file__).resolve().parents[1]
YARDSTICK = ROOT / "validation" / "reversal_yardstick.toml"
RUNNER_SCRIPT = ROOT / "scripts" / "run_reversal_yardstick.py"


def test_reversal_arena_runner_verifies_pinned_aggregate_without_certifying(tmp_path):
    arena_path = tmp_path / "arena.json"
    arena = _arena_payload(load_reversal_yardstick(YARDSTICK))
    arena_path.write_text(json.dumps(arena, indent=2), encoding="utf-8")
    yardstick = _yardstick_with_arena_pin(arena_path)

    report = run_reversal_arena_aggregate(arena_path, yardstick).to_mapping()

    assert report["status"] == REVERSAL_ARENA_STATUS
    assert report["source_artifact_pin_status"] == "verified"
    assert report["n_cases"] == 11
    assert report["detected_winrate_vs_standard_dl"] == pytest.approx(0.5046818182)
    assert report["oracle_winrate_vs_standard_dl"] == pytest.approx(0.7272727273)
    assert report["source_backed_reversal_fixtures_complete"] is False
    assert report["global_goal_complete"] is False
    assert report["certification_effect"] == "none"


def test_reversal_arena_runner_fails_closed_on_hash_drift_or_metric_drift(tmp_path):
    arena_path = tmp_path / "arena.json"
    arena = _arena_payload(load_reversal_yardstick(YARDSTICK))
    arena_path.write_text(json.dumps(arena, indent=2), encoding="utf-8")
    yardstick = _yardstick_with_arena_pin(arena_path)

    arena["mean_distance"]["DL"] += 0.01
    arena_path.write_text(json.dumps(arena, indent=2), encoding="utf-8")
    with pytest.raises(ReversalArenaError, match="hash drift"):
        run_reversal_arena_aggregate(arena_path, yardstick)

    yardstick = _yardstick_with_arena_pin(arena_path)
    with pytest.raises(ReversalArenaError, match="mean_distance.DL drifted"):
        run_reversal_arena_aggregate(arena_path, yardstick)


def test_reversal_arena_runner_rejects_goal_completion_or_source_backed_shortcuts(tmp_path):
    arena_path = tmp_path / "arena.json"
    arena = _arena_payload(load_reversal_yardstick(YARDSTICK))
    arena["global_goal_complete"] = True
    arena_path.write_text(json.dumps(arena, indent=2), encoding="utf-8")
    yardstick = _yardstick_with_arena_pin(arena_path)

    with pytest.raises(ReversalArenaError, match="global goal"):
        run_reversal_arena_aggregate(arena_path, yardstick)

    arena["global_goal_complete"] = False
    arena["source_backed_reversal_fixtures_complete"] = True
    arena_path.write_text(json.dumps(arena, indent=2), encoding="utf-8")
    yardstick = _yardstick_with_arena_pin(arena_path)

    with pytest.raises(ReversalArenaError, match="source-backed"):
        run_reversal_arena_aggregate(arena_path, yardstick)


def test_reversal_arena_runner_cli_reports_noncertifying_status(tmp_path):
    arena_path = tmp_path / "arena.json"
    arena = _arena_payload(load_reversal_yardstick(YARDSTICK))
    arena_path.write_text(json.dumps(arena, indent=2), encoding="utf-8")
    yardstick = _yardstick_with_arena_pin(arena_path)
    yardstick_path = tmp_path / "yardstick.toml"
    _write_minimal_yardstick_toml(yardstick, yardstick_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--yardstick",
            str(yardstick_path),
            "--arena-json",
            str(arena_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == REVERSAL_ARENA_STATUS
    assert payload["certification_effect"] == "none"
    assert payload["global_goal_complete"] is False


def _arena_payload(yardstick: ReversalYardstick) -> dict[str, object]:
    row_names = [f"case_{index:02d}" for index in range(yardstick.n_cases)]
    rows = []
    for index, name in enumerate(row_names):
        rows.append(
            {
                "name": name,
                "set": "A" if index < 7 else "B",
                "taint": "aggregate fixture",
                "truth": 0.0,
                "DL": yardstick.mean_distance["standard_dl"],
                "HKSJ": yardstick.mean_distance["standard_dl"],
                "TrimFill": yardstick.mean_distance["strong_standard"],
                "PETPEESE": yardstick.mean_distance["strong_standard"],
                "strong": yardstick.mean_distance["strong_standard"],
                "OURS_oracle": yardstick.mean_distance["ours_oracle"],
                "p_detect": 0.5,
                "ceiling": 0.0,
            }
        )
    return {
        "n_cases": yardstick.n_cases,
        "table": rows,
        "oracle": {
            "vs_standard_DL": [
                yardstick.oracle_vs_standard_dl.estimate,
                yardstick.oracle_vs_standard_dl.ci_lower,
                yardstick.oracle_vs_standard_dl.ci_upper,
            ],
            "vs_strong_standard": [
                yardstick.oracle_vs_strong_standard.estimate,
                yardstick.oracle_vs_strong_standard.ci_lower,
                yardstick.oracle_vs_strong_standard.ci_upper,
            ],
        },
        "detected": {
            "vs_standard_DL": [
                yardstick.detected_vs_standard_dl.estimate,
                yardstick.detected_vs_standard_dl.ci_lower,
                yardstick.detected_vs_standard_dl.ci_upper,
            ],
            "vs_strong_standard": [
                yardstick.detected_vs_strong_standard.estimate,
                yardstick.detected_vs_strong_standard.ci_lower,
                yardstick.detected_vs_strong_standard.ci_upper,
            ],
            "note": "STRICT win-rate; ties excluded.",
        },
        "mean_distance": {
            "DL": yardstick.mean_distance["standard_dl"],
            "strong_standard": yardstick.mean_distance["strong_standard"],
            "OURS_oracle": yardstick.mean_distance["ours_oracle"],
            "OURS_detected": yardstick.mean_distance["ours_detected"],
        },
        "oracle_minus_detected_gap_meandist": yardstick.mean_distance[
            "oracle_minus_detected_gap"
        ],
        "subclass": {
            "verification-target (registry/harms/fraud)": {"cases": row_names[:7]},
            "small-study-effect (strong standard's turf)": {"cases": row_names[7:9]},
            "observational-confounding (outside pillars)": {"cases": row_names[9:]},
        },
    }


def _yardstick_with_arena_pin(arena_path: Path) -> ReversalYardstick:
    raw = _yardstick_to_mapping(load_reversal_yardstick(YARDSTICK))
    raw["source_artifact_hashes"] = {
        "fix_md": "0" * 64,
        "arena_json": hashlib.sha256(arena_path.read_bytes()).hexdigest(),
    }
    return ReversalYardstick.from_mapping(raw)


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


def _write_minimal_yardstick_toml(yardstick: ReversalYardstick, path: Path) -> None:
    def q(value: object) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def qlist(values: object) -> str:
        return "[" + ", ".join(q(item) for item in values) + "]"

    lines = [
        f"schema_version = {q(REVERSAL_YARDSTICK_SCHEMA_VERSION)}",
        f"checked_at = {q(yardstick.checked_at)}",
        f"status = {q(yardstick.status)}",
        f"certification_effect = {q(yardstick.certification_effect)}",
        f"purpose = {q(yardstick.purpose)}",
        f"allowed_evidence_sources = {qlist(yardstick.allowed_evidence_sources)}",
        f"protocol_only_sources = {qlist(yardstick.protocol_only_sources)}",
        f"truth_boundary = {q(yardstick.truth_boundary)}",
        f"headline_metric = {q(yardstick.headline_metric)}",
        "oracle_only_reporting_allowed = false",
        "global_goal_complete = false",
        f"n_cases = {yardstick.n_cases}",
        f"priority_reversal_cases = {yardstick.priority_reversal_cases}",
        f"case_data_status = {q(yardstick.case_data_status)}",
        f"negative_control_status = {q(yardstick.negative_control_status)}",
        f"claim_limit = {q(yardstick.claim_limit)}",
        f"required_next_artifacts = {qlist(yardstick.required_next_artifacts)}",
        "",
        "[source_artifact_hashes]",
    ]
    lines.extend(
        f"{key} = {q(value)}"
        for key, value in yardstick.source_artifact_hashes.items()
    )
    lines.extend(
        [
            "",
            "[flag_recover]",
            f"flag_caught = {yardstick.flag_caught}",
            f"flag_total = {yardstick.flag_total}",
            f"recover_caught = {yardstick.recover_caught}",
            f"recover_total = {yardstick.recover_total}",
            "",
            "[detector]",
            f"status = {q(yardstick.detector_status)}",
            f"auc = {yardstick.detector_auc}",
            f"ci_lower = {yardstick.detector_ci[0]}",
            f"ci_upper = {yardstick.detector_ci[1]}",
            "",
        ]
    )
    for section, intervals in (
        (
            "oracle",
            {
                "vs_standard_dl": yardstick.oracle_vs_standard_dl,
                "vs_strong_standard": yardstick.oracle_vs_strong_standard,
            },
        ),
        (
            "detected",
            {
                "vs_standard_dl": yardstick.detected_vs_standard_dl,
                "vs_strong_standard": yardstick.detected_vs_strong_standard,
            },
        ),
    ):
        for comparator, interval in intervals.items():
            lines.extend(
                [
                    f"[{section}.{comparator}]",
                    f"estimate = {interval.estimate}",
                    f"ci_lower = {interval.ci_lower}",
                    f"ci_upper = {interval.ci_upper}",
                    "",
                ]
            )
    lines.append("[mean_distance]")
    lines.extend(f"{key} = {value}" for key, value in yardstick.mean_distance.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
