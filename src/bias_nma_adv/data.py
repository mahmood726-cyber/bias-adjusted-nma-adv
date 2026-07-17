"""Structured data models and datasets for Advanced Bias-Adjusted NMA."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
import re
import tomllib
from typing import Any, Mapping

from bias_nma_adv.evidence_sources import (
    ALLOWED_SOURCE_TYPES,
    EFFECT_EVIDENCE_SOURCE_TYPES,
    PUBLISHED_EFFECT_SOURCE_TYPES,
    REGISTRY_FIRST_EFFECT_SOURCE_TYPES,
)


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
    source_type: str | None = None

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
        object.__setattr__(self, "source_type", _validated_source_type(self.source_type))

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
    source_type: str | None = None

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
        source_type = _validated_source_type(self.source_type)
        if source_type is not None and source_type not in EFFECT_EVIDENCE_SOURCE_TYPES:
            raise ValidationError(
                f"Outcome source_type '{source_type}' cannot supply model-ready effects."
            )
        object.__setattr__(self, "source_type", source_type)

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
        source_type: str | None = None,
    ) -> None:
        covs = covariates or {}
        self.studies[study_id] = StudyRecord(
            study_id=study_id,
            design=design,
            rob_weight=float(rob_weight),
            covariates={k: float(v) for k, v in covs.items()},
            indirectness=indirectness,
            source_type=source_type,
        )

    def add_arm(self, study_id: str, arm_id: str, treatment_id: str, n: int) -> None:
        self._validate_model_ready_study(study_id)
        self.arms[(study_id, arm_id)] = ArmRecord(
            study_id=study_id,
            arm_id=arm_id,
            treatment_id=treatment_id,
            n=int(n)
        )

    def add_outcome_ad(
        self,
        study_id: str,
        arm_id: str,
        outcome_id: str,
        measure_type: str,
        value: float,
        se: float | None = None,
        source_type: str | None = None,
    ) -> None:
        self._validate_model_ready_study(study_id)
        self.outcomes_ad.append(OutcomeADRecord(
            study_id=study_id,
            arm_id=arm_id,
            outcome_id=outcome_id,
            measure_type=measure_type,
            value=float(value),
            se=float(se) if se is not None else None,
            source_type=source_type,
        ))

    def arm_lookup(self) -> dict[tuple[str, str], ArmRecord]:
        return self.arms

    def outcomes_by_study_outcome(self, study_id: str, outcome_id: str) -> list[OutcomeADRecord]:
        return [o for o in self.outcomes_ad if o.study_id == study_id and o.outcome_id == outcome_id]

    def measure_type_for_outcome(self, outcome_id: str) -> str:
        measure_types = {
            outcome.measure_type
            for outcome in self.outcomes_ad
            if outcome.outcome_id == outcome_id
        }
        if not measure_types:
            raise ValidationError(f"Outcome '{outcome_id}' not found in dataset.")
        if len(measure_types) > 1:
            raise ValidationError(
                f"Outcome '{outcome_id}' has multiple measure_type values: "
                f"{sorted(measure_types)}."
            )
        return next(iter(measure_types))

    def subset_by_input_source(self, mode: str) -> "EvidenceDataset":
        """Return a copy containing only rows for one declared input source lane.

        The estimator never sees the mode. Selection happens before fitting so
        as-published versus registry-first comparisons isolate the input layer.
        Untagged rows are excluded from both lanes.
        """

        if mode == "as_published":
            allowed_sources = PUBLISHED_EFFECT_SOURCE_TYPES
        elif mode == "registry_first":
            allowed_sources = REGISTRY_FIRST_EFFECT_SOURCE_TYPES
        else:
            raise ValidationError("mode must be 'as_published' or 'registry_first'.")

        selected_outcomes = []
        for outcome in self.outcomes_ad:
            study = self.studies.get(outcome.study_id)
            source_type = outcome.source_type or (study.source_type if study else None)
            if source_type in allowed_sources:
                selected_outcomes.append(outcome)

        selected_studies = {outcome.study_id for outcome in selected_outcomes}
        subset = EvidenceDataset()
        subset.studies = {
            study_id: study
            for study_id, study in self.studies.items()
            if study_id in selected_studies
        }
        subset.arms = {
            key: arm
            for key, arm in self.arms.items()
            if key[0] in selected_studies
        }
        subset.outcomes_ad = list(selected_outcomes)
        return subset

    def _validate_model_ready_study(self, study_id: str) -> None:
        study = self.studies.get(study_id)
        if study is None:
            return
        if study.design not in EFFECT_STUDY_DESIGNS:
            raise ValidationError(
                f"Study '{study_id}' has metadata-only design '{study.design}' and "
                "cannot carry model-ready arms or outcomes."
            )
        if study.source_type is not None and study.source_type not in EFFECT_EVIDENCE_SOURCE_TYPES:
            raise ValidationError(
                f"Study '{study_id}' has protocol-only source_type '{study.source_type}' "
                "and cannot carry model-ready arms or outcomes."
            )


def _validated_source_type(source_type: str | None) -> str | None:
    if source_type is None:
        return None
    normalized = source_type.strip()
    if not normalized:
        raise ValidationError("source_type must not be blank when provided.")
    if normalized not in ALLOWED_SOURCE_TYPES:
        raise ValidationError(
            f"source_type must be one of the configured source types: {sorted(ALLOWED_SOURCE_TYPES)}."
        )
    return normalized
