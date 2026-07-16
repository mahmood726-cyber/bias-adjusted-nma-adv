"""Fail-closed importer for RapidMeta-style app-index exports.

The repository does not assume a private RapidMeta schema. This adapter defines
the minimal public contract this project can safely consume and refuses
ambiguous or protocol-only inputs before they reach the estimator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bias_nma_adv.data import EvidenceDataset, ValidationError
from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES


RAPIDMETA_APP_INDEX_SCHEMA_VERSION = "rapidmeta_app_index/v1"


class RapidMetaAdapterError(ValueError):
    """Raised when a RapidMeta app-index payload is malformed."""


def evidence_dataset_from_rapidmeta_app_index(
    path: str | Path,
    *,
    analysis_id: str | None = None,
) -> EvidenceDataset:
    """Load a binary arm-level analysis from a strict app-index JSON export."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    analysis = _select_analysis(payload, analysis_id=analysis_id)
    return _analysis_to_dataset(analysis)


def _select_analysis(
    payload: dict[str, Any],
    *,
    analysis_id: str | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RapidMetaAdapterError("RapidMeta app-index must be a JSON object.")
    if payload.get("schema_version") != RAPIDMETA_APP_INDEX_SCHEMA_VERSION:
        raise RapidMetaAdapterError(
            f"schema_version must be {RAPIDMETA_APP_INDEX_SCHEMA_VERSION}."
        )
    analyses = payload.get("analyses")
    if not isinstance(analyses, list) or not analyses:
        raise RapidMetaAdapterError("analyses must be a non-empty list.")

    if analysis_id is None:
        if len(analyses) != 1:
            raise RapidMetaAdapterError(
                "analysis_id is required when app-index contains multiple analyses."
            )
        analysis = analyses[0]
    else:
        matches = [
            analysis
            for analysis in analyses
            if isinstance(analysis, dict) and analysis.get("analysis_id") == analysis_id
        ]
        if len(matches) != 1:
            raise RapidMetaAdapterError(
                f"analysis_id '{analysis_id}' must match exactly one analysis."
            )
        analysis = matches[0]

    if not isinstance(analysis, dict):
        raise RapidMetaAdapterError("analysis entries must be JSON objects.")
    return analysis


def _analysis_to_dataset(analysis: dict[str, Any]) -> EvidenceDataset:
    outcome_id = _required_text(analysis, "outcome_id")
    if analysis.get("measure_type") != "binary":
        raise RapidMetaAdapterError("only binary arm-level analyses are supported.")
    studies = analysis.get("studies")
    if not isinstance(studies, list) or not studies:
        raise RapidMetaAdapterError("analysis studies must be a non-empty list.")

    dataset = EvidenceDataset()
    for study in studies:
        if not isinstance(study, dict):
            raise RapidMetaAdapterError("study entries must be JSON objects.")
        study_id = _required_text(study, "study_id")
        design = _required_text(study, "design")
        source_type = _required_text(study, "source_type")
        if source_type not in EFFECT_EVIDENCE_SOURCE_TYPES:
            raise RapidMetaAdapterError(
                f"{study_id}: source_type '{source_type}' cannot supply model-ready effects."
            )
        dataset.add_study(study_id, design, source_type=source_type)

        arms = study.get("arms")
        if not isinstance(arms, list) or len(arms) < 2:
            raise RapidMetaAdapterError(f"{study_id}: at least two arms are required.")
        for arm in arms:
            if not isinstance(arm, dict):
                raise RapidMetaAdapterError(f"{study_id}: arm entries must be JSON objects.")
            arm_id = _required_text(arm, "arm_id")
            treatment_id = _required_text(arm, "treatment_id")
            n = _required_non_negative_int(arm, "n")
            events = _required_non_negative_int(arm, "events")
            if events > n:
                raise RapidMetaAdapterError(
                    f"{study_id}/{arm_id}: events cannot exceed arm n."
                )
            dataset.add_arm(study_id, arm_id, treatment_id, n)
            dataset.add_outcome_ad(
                study_id,
                arm_id,
                outcome_id,
                "binary",
                events,
                source_type=source_type,
            )

    try:
        dataset.measure_type_for_outcome(outcome_id)
    except ValidationError as exc:
        raise RapidMetaAdapterError(str(exc)) from exc
    return dataset


def _required_text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None:
        raise RapidMetaAdapterError(f"{key} is required.")
    text = str(value).strip()
    if not text:
        raise RapidMetaAdapterError(f"{key} must not be blank.")
    return text


def _required_non_negative_int(raw: dict[str, Any], key: str) -> int:
    if key not in raw:
        raise RapidMetaAdapterError(f"{key} is required.")
    value = raw[key]
    if isinstance(value, bool):
        raise RapidMetaAdapterError(f"{key} must be a non-negative integer.")
    if isinstance(value, int):
        integer = value
    elif isinstance(value, str) and value.strip().isdigit():
        integer = int(value.strip())
    else:
        raise RapidMetaAdapterError(f"{key} must be a non-negative integer.")
    if integer < 0:
        raise RapidMetaAdapterError(f"{key} must be a non-negative integer.")
    return integer
