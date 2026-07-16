"""Reversal-answer-key yardstick governance.

This module validates a non-certifying acceptance yardstick for input-verified
evidence synthesis. It deliberately separates oracle-taint performance from
detected-taint performance and FLAG from RECOVER, because those are different
claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES, PROTOCOL_ONLY_SOURCE_TYPES


REVERSAL_YARDSTICK_SCHEMA_VERSION = "reversal_yardstick/v1"


class ReversalYardstickError(ValueError):
    """Raised when the reversal yardstick is malformed or too permissive."""


@dataclass(frozen=True)
class WinRateInterval:
    """One win-rate estimate and confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "WinRateInterval":
        for key in ("estimate", "ci_lower", "ci_upper"):
            if key not in raw:
                raise ReversalYardstickError(f"win-rate interval missing {key}.")
        interval = cls(
            estimate=float(raw["estimate"]),
            ci_lower=float(raw["ci_lower"]),
            ci_upper=float(raw["ci_upper"]),
        )
        interval.validate()
        return interval

    def validate(self) -> None:
        if not 0.0 <= self.ci_lower <= self.estimate <= self.ci_upper <= 1.0:
            raise ReversalYardstickError("win-rate interval must be ordered within [0, 1].")


@dataclass(frozen=True)
class ReversalYardstick:
    """Validated reversal-yardstick metadata."""

    checked_at: str
    status: str
    certification_effect: str
    purpose: str
    allowed_evidence_sources: tuple[str, ...]
    protocol_only_sources: tuple[str, ...]
    truth_boundary: str
    headline_metric: str
    oracle_only_reporting_allowed: bool
    global_goal_complete: bool
    n_cases: int
    priority_reversal_cases: int
    case_data_status: str
    negative_control_status: str
    claim_limit: str
    flag_caught: int
    flag_total: int
    recover_caught: int
    recover_total: int
    detector_status: str
    detector_auc: float
    detector_ci: tuple[float, float]
    oracle_vs_standard_dl: WinRateInterval
    oracle_vs_strong_standard: WinRateInterval
    detected_vs_standard_dl: WinRateInterval
    detected_vs_strong_standard: WinRateInterval
    mean_distance: dict[str, float]
    required_next_artifacts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ReversalYardstick":
        required = {
            "schema_version",
            "checked_at",
            "status",
            "certification_effect",
            "purpose",
            "allowed_evidence_sources",
            "protocol_only_sources",
            "truth_boundary",
            "headline_metric",
            "oracle_only_reporting_allowed",
            "global_goal_complete",
            "n_cases",
            "priority_reversal_cases",
            "case_data_status",
            "negative_control_status",
            "claim_limit",
            "flag_recover",
            "detector",
            "oracle",
            "detected",
            "mean_distance",
            "required_next_artifacts",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ReversalYardstickError(f"reversal yardstick missing keys: {missing}")
        if raw["schema_version"] != REVERSAL_YARDSTICK_SCHEMA_VERSION:
            raise ReversalYardstickError(
                f"schema_version must be {REVERSAL_YARDSTICK_SCHEMA_VERSION}."
            )
        flag_recover = raw["flag_recover"]
        detector = raw["detector"]
        oracle = raw["oracle"]
        detected = raw["detected"]
        yardstick = cls(
            checked_at=str(raw["checked_at"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            purpose=str(raw["purpose"]),
            allowed_evidence_sources=tuple(str(item) for item in raw["allowed_evidence_sources"]),
            protocol_only_sources=tuple(str(item) for item in raw["protocol_only_sources"]),
            truth_boundary=str(raw["truth_boundary"]),
            headline_metric=str(raw["headline_metric"]),
            oracle_only_reporting_allowed=bool(raw["oracle_only_reporting_allowed"]),
            global_goal_complete=bool(raw["global_goal_complete"]),
            n_cases=int(raw["n_cases"]),
            priority_reversal_cases=int(raw["priority_reversal_cases"]),
            case_data_status=str(raw["case_data_status"]),
            negative_control_status=str(raw["negative_control_status"]),
            claim_limit=str(raw["claim_limit"]),
            flag_caught=int(flag_recover["flag_caught"]),
            flag_total=int(flag_recover["flag_total"]),
            recover_caught=int(flag_recover["recover_caught"]),
            recover_total=int(flag_recover["recover_total"]),
            detector_status=str(detector["status"]),
            detector_auc=float(detector["auc"]),
            detector_ci=(float(detector["ci_lower"]), float(detector["ci_upper"])),
            oracle_vs_standard_dl=WinRateInterval.from_mapping(oracle["vs_standard_dl"]),
            oracle_vs_strong_standard=WinRateInterval.from_mapping(oracle["vs_strong_standard"]),
            detected_vs_standard_dl=WinRateInterval.from_mapping(detected["vs_standard_dl"]),
            detected_vs_strong_standard=WinRateInterval.from_mapping(detected["vs_strong_standard"]),
            mean_distance={str(key): float(value) for key, value in raw["mean_distance"].items()},
            required_next_artifacts=tuple(str(item) for item in raw["required_next_artifacts"]),
        )
        yardstick.validate()
        return yardstick

    def validate(self) -> None:
        if tuple(sorted(self.allowed_evidence_sources)) != tuple(
            sorted(EFFECT_EVIDENCE_SOURCE_TYPES)
        ):
            raise ReversalYardstickError("allowed_evidence_sources drifted from source boundary.")
        if tuple(sorted(self.protocol_only_sources)) != tuple(sorted(PROTOCOL_ONLY_SOURCE_TYPES)):
            raise ReversalYardstickError("protocol_only_sources drifted from source boundary.")
        if self.certification_effect != "none":
            raise ReversalYardstickError("reversal yardstick cannot certify model performance.")
        if self.global_goal_complete:
            raise ReversalYardstickError("reversal yardstick must not mark the global goal complete.")
        if self.headline_metric != "detected_taint":
            raise ReversalYardstickError("detected-taint performance must be the headline metric.")
        if self.oracle_only_reporting_allowed:
            raise ReversalYardstickError("oracle-only reporting is not allowed.")
        if self.n_cases < 11 or self.priority_reversal_cases < 5:
            raise ReversalYardstickError("reversal yardstick must preserve the case counts.")
        if self.flag_total != self.priority_reversal_cases or self.recover_total != self.priority_reversal_cases:
            raise ReversalYardstickError("FLAG and RECOVER totals must match priority cases.")
        if self.flag_caught <= self.recover_caught:
            raise ReversalYardstickError("FLAG and RECOVER must remain separate; FLAG should exceed RECOVER here.")
        if "not_validated" not in self.detector_status and "underpowered" not in self.detector_status:
            raise ReversalYardstickError("detector status must disclose underpowered/not-validated state.")
        ci_lower, ci_upper = self.detector_ci
        if not ci_lower <= 0.5 <= ci_upper:
            raise ReversalYardstickError("detector CI must disclose that chance performance remains plausible.")
        if self.detected_vs_standard_dl.estimate > self.oracle_vs_standard_dl.estimate:
            raise ReversalYardstickError("detected performance cannot exceed oracle performance.")
        if self.detected_vs_strong_standard.estimate > self.oracle_vs_strong_standard.estimate:
            raise ReversalYardstickError("detected performance cannot exceed oracle performance.")
        required_mean_keys = {
            "standard_dl",
            "strong_standard",
            "ours_oracle",
            "ours_detected",
            "oracle_minus_detected_gap",
        }
        if not required_mean_keys <= set(self.mean_distance):
            raise ReversalYardstickError("mean_distance is missing required methods.")
        if self.mean_distance["ours_detected"] < self.mean_distance["ours_oracle"]:
            raise ReversalYardstickError("detected mean distance cannot beat oracle mean distance.")
        if self.negative_control_status != "required_not_complete":
            raise ReversalYardstickError("negative controls must remain an explicit blocker.")
        if "No tier-one superiority" not in self.claim_limit:
            raise ReversalYardstickError("claim_limit must block unsupported superiority claims.")
        required_artifacts = {
            "matched_negative_control_case_library",
            "automated_blind_trigger_runner_frozen_before_scoring",
            "oracle_and_detected_taint_report",
            "external_multiperson_review_before_any_superiority_claim",
        }
        if not required_artifacts <= set(self.required_next_artifacts):
            raise ReversalYardstickError("required_next_artifacts is missing yardstick blockers.")


def load_reversal_yardstick(path: str | Path) -> ReversalYardstick:
    """Load and validate the reversal yardstick."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return ReversalYardstick.from_mapping(payload)


def summarize_reversal_yardstick(yardstick: ReversalYardstick) -> dict[str, Any]:
    """Return validation-status-friendly yardstick fields."""

    return {
        "schema_version": REVERSAL_YARDSTICK_SCHEMA_VERSION,
        "checked_at": yardstick.checked_at,
        "status": yardstick.status,
        "headline_metric": yardstick.headline_metric,
        "n_cases": yardstick.n_cases,
        "priority_reversal_cases": yardstick.priority_reversal_cases,
        "flag_caught": yardstick.flag_caught,
        "flag_total": yardstick.flag_total,
        "recover_caught": yardstick.recover_caught,
        "recover_total": yardstick.recover_total,
        "oracle_winrate_vs_standard_dl": yardstick.oracle_vs_standard_dl.estimate,
        "detected_winrate_vs_standard_dl": yardstick.detected_vs_standard_dl.estimate,
        "oracle_minus_detected_gap_meandist": yardstick.mean_distance[
            "oracle_minus_detected_gap"
        ],
        "detector_status": yardstick.detector_status,
        "negative_control_status": yardstick.negative_control_status,
        "global_goal_complete": yardstick.global_goal_complete,
        "certification_effect": yardstick.certification_effect,
    }
