"""Structured data models and datasets for Advanced Bias-Adjusted NMA."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import re
import tomllib
from typing import Any, Mapping

class ValidationError(ValueError):
    """Raised when data fails schema validation."""
    pass


STUDY_DESIGN_POLICY_SCHEMA_VERSION = "study_design_policy/v1"


@dataclass(frozen=True)
class StudyDesignPolicy:
    """Policy entry for a study-design or evidence-frame label."""

    id: str
    role: str
    label: str

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "StudyDesignPolicy":
        design_id = str(raw["id"]).strip().lower()
        if re.match(r"^[a-z][a-z0-9_]{1,63}$", design_id) is None:
            raise ValidationError(f"invalid study design id: {design_id!r}.")
        return cls(
            id=design_id,
            role=str(raw["role"]).strip().lower(),
            label=str(raw["label"]).strip(),
        )


def load_study_design_policies() -> dict[str, StudyDesignPolicy]:
    """Load study-design policies bundled with the package."""

    policy_text = resources.files(__package__).joinpath("study_design_policy.toml").read_text(
        encoding="utf-8"
    )
    raw = tomllib.loads(policy_text)
    if raw.get("schema_version") != STUDY_DESIGN_POLICY_SCHEMA_VERSION:
        raise ValidationError(
            f"study design policy schema_version must be {STUDY_DESIGN_POLICY_SCHEMA_VERSION}."
        )
    policies = {
        policy.id: policy
        for policy in (
            StudyDesignPolicy.from_mapping(item) for item in raw.get("designs", [])
        )
    }
    if {"rct", "nrs", "other"} - set(policies):
        raise ValidationError("study design policy must preserve rct, nrs, and other.")
    if len(policies) != len(raw.get("designs", [])):
        raise ValidationError("study design policy contains duplicate design ids.")
    return policies


STUDY_DESIGN_POLICIES = load_study_design_policies()
ALLOWED_STUDY_DESIGNS = frozenset(STUDY_DESIGN_POLICIES)
EFFECT_STUDY_DESIGNS = frozenset(
    design_id
    for design_id, policy in STUDY_DESIGN_POLICIES.items()
    if policy.role == "effect_design"
)

@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    design: str
    rob_weight: float = 1.0  # Quality score / weight in (0, 1]
    covariates: dict[str, float] = field(default_factory=dict)
    indirectness: str | None = None

    def __post_init__(self):
        normalized_design = self.design.strip().lower()
        if normalized_design not in ALLOWED_STUDY_DESIGNS:
            raise ValidationError(
                "Field 'design' must be one of the configured study designs: "
                f"{sorted(ALLOWED_STUDY_DESIGNS)}."
            )
        object.__setattr__(self, "design", normalized_design)
        if not (0.0 < self.rob_weight <= 1.0):
            raise ValidationError("rob_weight must be in the range (0, 1].")
        if self.indirectness is not None:
            indirectness = self.indirectness.strip()
            if not indirectness:
                raise ValidationError("indirectness must not be blank when provided.")
            object.__setattr__(self, "indirectness", indirectness)

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

    def add_study(
        self,
        study_id: str,
        design: str,
        rob_weight: float = 1.0,
        covariates: dict[str, float] | None = None,
        indirectness: str | None = None,
    ) -> None:
        covs = covariates or {}
        self.studies[study_id] = StudyRecord(
            study_id=study_id,
            design=design,
            rob_weight=float(rob_weight),
            covariates={k: float(v) for k, v in covs.items()},
            indirectness=indirectness,
        )

    def add_arm(self, study_id: str, arm_id: str, treatment_id: str, n: int) -> None:
        self._validate_model_ready_study(study_id)
        self.arms[(study_id, arm_id)] = ArmRecord(
            study_id=study_id,
            arm_id=arm_id,
            treatment_id=treatment_id,
            n=int(n)
        )

    def add_outcome_ad(self, study_id: str, arm_id: str, outcome_id: str, measure_type: str, value: float, se: float | None = None) -> None:
        self._validate_model_ready_study(study_id)
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

    def _validate_model_ready_study(self, study_id: str) -> None:
        study = self.studies.get(study_id)
        if study is None:
            return
        if study.design not in EFFECT_STUDY_DESIGNS:
            raise ValidationError(
                f"Study '{study_id}' has metadata-only design '{study.design}' and "
                "cannot carry model-ready arms or outcomes."
            )
