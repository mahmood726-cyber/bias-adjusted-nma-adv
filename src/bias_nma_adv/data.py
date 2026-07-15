"""Structured data models and datasets for Advanced Bias-Adjusted NMA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

class ValidationError(ValueError):
    """Raised when data fails schema validation."""
    pass

@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    design: str  # "rct", "nrs", or "other"
    rob_weight: float = 1.0  # Quality score / weight in (0, 1]
    covariates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.design not in {"rct", "nrs", "other"}:
            raise ValidationError("Field 'design' must be one of: rct, nrs, other.")
        if not (0.0 < self.rob_weight <= 1.0):
            raise ValidationError("rob_weight must be in the range (0, 1].")

@dataclass(frozen=True)
class ArmRecord:
    study_id: str
    arm_id: str
    treatment_id: str
    n: int

    def __post_init__(self):
        if self.n <= 0:
            raise ValidationError("Arm sample size 'n' must be > 0.")

@dataclass(frozen=True)
class OutcomeADRecord:
    study_id: str
    arm_id: str
    outcome_id: str
    measure_type: str  # "binary" or "continuous"
    value: float  # events for binary, mean for continuous
    se: float | None = None  # standard error (required for continuous)

    def __post_init__(self):
        if self.measure_type not in {"binary", "continuous"}:
            raise ValidationError("Field 'measure_type' must be one of: binary, continuous.")
        if self.measure_type == "continuous":
            if self.se is None or self.se <= 0.0:
                raise ValidationError("Continuous outcomes require standard error 'se' > 0.")
        else:
            if self.value < 0.0:
                raise ValidationError("Binary outcome events 'value' must be >= 0.")
            if self.se is not None and self.se <= 0.0:
                raise ValidationError("Field 'se' must be > 0 when provided.")

class EvidenceDataset:
    """In-memory database of study, arm, and outcome records."""

    def __init__(self):
        self.studies: dict[str, StudyRecord] = {}
        self.arms: dict[tuple[str, str], ArmRecord] = {}
        self.outcomes_ad: list[OutcomeADRecord] = []

    def add_study(self, study_id: str, design: str, rob_weight: float = 1.0, covariates: dict[str, float] | None = None) -> None:
        covs = covariates or {}
        self.studies[study_id] = StudyRecord(
            study_id=study_id,
            design=design,
            rob_weight=float(rob_weight),
            covariates={k: float(v) for k, v in covs.items()}
        )

    def add_arm(self, study_id: str, arm_id: str, treatment_id: str, n: int) -> None:
        self.arms[(study_id, arm_id)] = ArmRecord(
            study_id=study_id,
            arm_id=arm_id,
            treatment_id=treatment_id,
            n=int(n)
        )

    def add_outcome_ad(self, study_id: str, arm_id: str, outcome_id: str, measure_type: str, value: float, se: float | None = None) -> None:
        self.outcomes_ad.append(OutcomeADRecord(
            study_id=study_id,
            arm_id=arm_id,
            outcome_id=outcome_id,
            measure_type=measure_type,
            value=float(value),
            se=float(se) if se is not None else None
        ))

    def arm_lookup(self) -> dict[tuple[str, str], ArmRecord]:
        return self.arms

    def outcomes_by_study_outcome(self, study_id: str, outcome_id: str) -> list[OutcomeADRecord]:
        return [o for o in self.outcomes_ad if o.study_id == study_id and o.outcome_id == outcome_id]

    def measure_type_for_outcome(self, outcome_id: str) -> str:
        for o in self.outcomes_ad:
            if o.outcome_id == outcome_id:
                return o.measure_type
        raise ValidationError(f"Outcome '{outcome_id}' not found in dataset.")
