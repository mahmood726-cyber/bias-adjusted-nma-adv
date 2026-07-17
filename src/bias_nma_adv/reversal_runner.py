"""Aggregate reversal-answer-key runner.

The external reversal arena is an aggregate scored artifact, not a source-backed
clinical dataset. This module verifies that the pinned aggregate JSON is present,
matches the governance yardstick, and remains non-certifying.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from bias_nma_adv.reversal_yardstick import (
    REVERSAL_YARDSTICK_SCHEMA_VERSION,
    ReversalYardstick,
    ReversalYardstickError,
    WinRateInterval,
    load_reversal_yardstick,
)


REVERSAL_ARENA_AGGREGATE_SCHEMA_VERSION = "reversal_arena_aggregate/v1"
REVERSAL_ARENA_STATUS = "aggregate_verified_not_source_backed"

_MEAN_DISTANCE_KEYS = {
    "DL": "standard_dl",
    "strong_standard": "strong_standard",
    "OURS_oracle": "ours_oracle",
    "OURS_detected": "ours_detected",
}
_REQUIRED_ROW_KEYS = {
    "name",
    "set",
    "taint",
    "truth",
    "DL",
    "HKSJ",
    "TrimFill",
    "PETPEESE",
    "strong",
    "OURS_oracle",
    "p_detect",
    "ceiling",
}
_REQUIRED_SUBCLASSES = {
    "verification-target (registry/harms/fraud)",
    "small-study-effect (strong standard's turf)",
    "observational-confounding (outside pillars)",
}


class ReversalArenaError(ReversalYardstickError):
    """Raised when a reversal aggregate run is malformed or overclaims."""


@dataclass(frozen=True)
class ReversalArenaReport:
    """Non-certifying verification report for a pinned reversal arena JSON."""

    status: str
    n_cases: int
    source_artifact_pin_status: str
    detected_winrate_vs_standard_dl: float
    oracle_winrate_vs_standard_dl: float
    oracle_minus_detected_gap_meandist: float
    certification_effect: str = "none"
    global_goal_complete: bool = False
    source_backed_reversal_fixtures_complete: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": REVERSAL_ARENA_AGGREGATE_SCHEMA_VERSION,
            "yardstick_schema_version": REVERSAL_YARDSTICK_SCHEMA_VERSION,
            "status": self.status,
            "n_cases": self.n_cases,
            "source_artifact_pin_status": self.source_artifact_pin_status,
            "detected_winrate_vs_standard_dl": self.detected_winrate_vs_standard_dl,
            "oracle_winrate_vs_standard_dl": self.oracle_winrate_vs_standard_dl,
            "oracle_minus_detected_gap_meandist": self.oracle_minus_detected_gap_meandist,
            "certification_effect": self.certification_effect,
            "global_goal_complete": self.global_goal_complete,
            "source_backed_reversal_fixtures_complete": (
                self.source_backed_reversal_fixtures_complete
            ),
        }


def run_reversal_arena_aggregate(
    arena_json_path: str | Path,
    yardstick: ReversalYardstick,
    *,
    tolerance: float = 1e-9,
) -> ReversalArenaReport:
    """Verify a pinned external reversal-arena aggregate JSON.

    This function does not fit the reversal cases and does not validate clinical
    source data. It only checks the aggregate answer-key file against its pinned
    SHA-256 and the committed yardstick metrics.
    """

    arena_path = Path(arena_json_path)
    try:
        pin_report = yardstick.verify_source_artifact_pins({"arena_json": arena_path})
    except ReversalYardstickError as exc:
        raise ReversalArenaError(str(exc)) from exc
    if pin_report["status"] != "verified":
        raise ReversalArenaError("arena_json source artifact is unavailable.")

    with arena_path.open("r", encoding="utf-8") as handle:
        arena = json.load(handle)
    _validate_arena_claim_boundaries(arena)
    _validate_arena_structure(arena, yardstick)
    _validate_arena_metrics(arena, yardstick, tolerance=tolerance)

    return ReversalArenaReport(
        status=REVERSAL_ARENA_STATUS,
        n_cases=yardstick.n_cases,
        source_artifact_pin_status=pin_report["status"],
        detected_winrate_vs_standard_dl=yardstick.detected_vs_standard_dl.estimate,
        oracle_winrate_vs_standard_dl=yardstick.oracle_vs_standard_dl.estimate,
        oracle_minus_detected_gap_meandist=yardstick.mean_distance[
            "oracle_minus_detected_gap"
        ],
    )


def run_reversal_arena_aggregate_from_paths(
    arena_json_path: str | Path,
    yardstick_path: str | Path,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Load paths, verify the aggregate arena, and return a JSON-ready report."""

    try:
        yardstick = load_reversal_yardstick(yardstick_path)
    except ReversalYardstickError as exc:
        raise ReversalArenaError(str(exc)) from exc
    return run_reversal_arena_aggregate(
        arena_json_path,
        yardstick,
        tolerance=tolerance,
    ).to_mapping()


def _validate_arena_claim_boundaries(arena: Mapping[str, Any]) -> None:
    if arena.get("certification_effect", "none") != "none":
        raise ReversalArenaError("reversal arena aggregate cannot certify performance.")
    if bool(arena.get("global_goal_complete", False)):
        raise ReversalArenaError("reversal arena aggregate cannot mark the global goal complete.")
    if bool(arena.get("source_backed_reversal_fixtures_complete", False)):
        raise ReversalArenaError(
            "aggregate arena cannot claim source-backed reversal fixtures are complete."
        )


def _validate_arena_structure(arena: Mapping[str, Any], yardstick: ReversalYardstick) -> None:
    for key in (
        "n_cases",
        "table",
        "oracle",
        "detected",
        "mean_distance",
        "oracle_minus_detected_gap_meandist",
        "subclass",
    ):
        if key not in arena:
            raise ReversalArenaError(f"arena_json missing {key}.")
    if int(arena["n_cases"]) != yardstick.n_cases:
        raise ReversalArenaError("arena_json n_cases does not match the yardstick.")
    rows = arena["table"]
    if not isinstance(rows, list) or len(rows) != yardstick.n_cases:
        raise ReversalArenaError("arena_json table length does not match n_cases.")

    names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReversalArenaError("arena_json table rows must be objects.")
        missing = sorted(_REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise ReversalArenaError(f"arena_json table row missing keys: {missing}")
        name = str(row["name"]).strip()
        if not name:
            raise ReversalArenaError("arena_json table row has an empty name.")
        if name in names:
            raise ReversalArenaError(f"arena_json table contains duplicate case: {name}")
        names.add(name)
        if str(row["set"]) not in {"A", "B"}:
            raise ReversalArenaError("arena_json table set must be A or B.")
        for numeric_key in (
            "truth",
            "DL",
            "HKSJ",
            "TrimFill",
            "PETPEESE",
            "strong",
            "OURS_oracle",
            "p_detect",
            "ceiling",
        ):
            _coerce_finite(row[numeric_key], f"table.{name}.{numeric_key}")
        p_detect = float(row["p_detect"])
        if not 0.0 <= p_detect <= 1.0:
            raise ReversalArenaError("arena_json p_detect must be within [0, 1].")
        if float(row["ceiling"]) < 0.0:
            raise ReversalArenaError("arena_json ceiling must be non-negative.")

    subclasses = arena["subclass"]
    if not isinstance(subclasses, Mapping):
        raise ReversalArenaError("arena_json subclass must be an object.")
    missing_subclasses = sorted(_REQUIRED_SUBCLASSES - set(subclasses))
    if missing_subclasses:
        raise ReversalArenaError(f"arena_json missing subclasses: {missing_subclasses}")
    subclass_cases: set[str] = set()
    for subclass_id, subclass in subclasses.items():
        if not isinstance(subclass, Mapping) or "cases" not in subclass:
            raise ReversalArenaError(f"arena_json subclass {subclass_id} missing cases.")
        for case_name in subclass["cases"]:
            case_name = str(case_name)
            if case_name not in names:
                raise ReversalArenaError(f"arena_json subclass case is absent: {case_name}")
            subclass_cases.add(case_name)
    if subclass_cases != names:
        raise ReversalArenaError("arena_json subclasses must cover every table case exactly.")


def _validate_arena_metrics(
    arena: Mapping[str, Any],
    yardstick: ReversalYardstick,
    *,
    tolerance: float,
) -> None:
    _assert_interval_matches(
        "oracle.vs_standard_DL",
        arena["oracle"]["vs_standard_DL"],
        yardstick.oracle_vs_standard_dl,
        tolerance,
    )
    _assert_interval_matches(
        "oracle.vs_strong_standard",
        arena["oracle"]["vs_strong_standard"],
        yardstick.oracle_vs_strong_standard,
        tolerance,
    )
    _assert_interval_matches(
        "detected.vs_standard_DL",
        arena["detected"]["vs_standard_DL"],
        yardstick.detected_vs_standard_dl,
        tolerance,
    )
    _assert_interval_matches(
        "detected.vs_strong_standard",
        arena["detected"]["vs_strong_standard"],
        yardstick.detected_vs_strong_standard,
        tolerance,
    )
    if "ties" not in str(arena["detected"].get("note", "")).lower():
        raise ReversalArenaError("detected-taint note must disclose that ties are excluded.")

    mean_distance = arena["mean_distance"]
    for arena_key, yardstick_key in _MEAN_DISTANCE_KEYS.items():
        if arena_key not in mean_distance:
            raise ReversalArenaError(f"arena_json mean_distance missing {arena_key}.")
        _assert_close(
            f"mean_distance.{arena_key}",
            float(mean_distance[arena_key]),
            yardstick.mean_distance[yardstick_key],
            tolerance,
        )
    _assert_close(
        "oracle_minus_detected_gap_meandist",
        float(arena["oracle_minus_detected_gap_meandist"]),
        yardstick.mean_distance["oracle_minus_detected_gap"],
        tolerance,
    )
    _assert_close(
        "mean_distance_gap_identity",
        float(mean_distance["OURS_detected"]) - float(mean_distance["OURS_oracle"]),
        float(arena["oracle_minus_detected_gap_meandist"]),
        tolerance,
    )

    if (
        yardstick.detected_vs_standard_dl.estimate
        > yardstick.oracle_vs_standard_dl.estimate
    ):
        raise ReversalArenaError("detected performance cannot exceed oracle performance.")


def _assert_interval_matches(
    label: str,
    raw: object,
    expected: WinRateInterval,
    tolerance: float,
) -> None:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ReversalArenaError(f"{label} must be a three-value interval.")
    actual = WinRateInterval(
        estimate=float(raw[0]),
        ci_lower=float(raw[1]),
        ci_upper=float(raw[2]),
    )
    actual.validate()
    _assert_close(f"{label}.estimate", actual.estimate, expected.estimate, tolerance)
    _assert_close(f"{label}.ci_lower", actual.ci_lower, expected.ci_lower, tolerance)
    _assert_close(f"{label}.ci_upper", actual.ci_upper, expected.ci_upper, tolerance)


def _assert_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    if abs(actual - expected) > tolerance:
        raise ReversalArenaError(
            f"{label} drifted: expected {expected:.12g}, observed {actual:.12g}."
        )


def _coerce_finite(value: object, label: str) -> float:
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise ReversalArenaError(f"{label} must be finite.")
    return numeric
