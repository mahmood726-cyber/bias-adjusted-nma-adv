"""Validation for source-backed event-count verification snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import string
from typing import Any

from bias_nma_adv.data import ValidationError


EVENT_COUNT_VERIFICATION_SCHEMA_VERSION = "event_count_verification/v1"


@dataclass(frozen=True)
class EventCountVerificationRecord:
    """One PubMed abstract event-count check for a two-arm real-meta study."""

    study_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    evidence_scope: str
    abstract_sha256: str
    active_events: int
    active_n: int
    control_events: int
    control_n: int
    active_count_token: str
    control_count_token: str
    active_source_terms: tuple[str, ...]
    control_source_terms: tuple[str, ...]
    active_count_token_found: bool
    control_count_token_found: bool
    active_term_near_count: bool
    control_term_near_count: bool
    verified: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "EventCountVerificationRecord":
        required = {
            "study_id",
            "pmid",
            "outcome_id",
            "outcome_label",
            "evidence_scope",
            "abstract_sha256",
            "active_events",
            "active_n",
            "control_events",
            "control_n",
            "active_count_token",
            "control_count_token",
            "active_source_terms",
            "control_source_terms",
            "active_count_token_found",
            "control_count_token_found",
            "active_term_near_count",
            "control_term_near_count",
            "verified",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"event-count verification record missing required keys: {missing}")
        active_source_terms = raw["active_source_terms"]
        control_source_terms = raw["control_source_terms"]
        if not isinstance(active_source_terms, (list, tuple)) or not isinstance(control_source_terms, (list, tuple)):
            raise ValidationError("event-count verification source terms must be lists.")
        record = cls(
            study_id=str(raw["study_id"]),
            pmid=str(raw["pmid"]),
            outcome_id=str(raw["outcome_id"]),
            outcome_label=str(raw["outcome_label"]),
            evidence_scope=str(raw["evidence_scope"]),
            abstract_sha256=str(raw["abstract_sha256"]),
            active_events=int(raw["active_events"]),
            active_n=int(raw["active_n"]),
            control_events=int(raw["control_events"]),
            control_n=int(raw["control_n"]),
            active_count_token=str(raw["active_count_token"]),
            control_count_token=str(raw["control_count_token"]),
            active_source_terms=tuple(str(term) for term in active_source_terms),
            control_source_terms=tuple(str(term) for term in control_source_terms),
            active_count_token_found=bool(raw["active_count_token_found"]),
            control_count_token_found=bool(raw["control_count_token_found"]),
            active_term_near_count=bool(raw["active_term_near_count"]),
            control_term_near_count=bool(raw["control_term_near_count"]),
            verified=bool(raw["verified"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("event-count verification study_id must not be empty.")
        if not self.pmid.isdigit():
            raise ValidationError(f"{self.study_id}: PMID must be numeric.")
        if self.evidence_scope != "pubmed_abstract_event_count_tokens":
            raise ValidationError(f"{self.study_id}: unsupported evidence_scope '{self.evidence_scope}'.")
        if not _looks_like_sha256(self.abstract_sha256):
            raise ValidationError(f"{self.study_id}: abstract_sha256 is not a SHA-256 digest.")
        if min(self.active_events, self.active_n, self.control_events, self.control_n) <= 0:
            raise ValidationError(f"{self.study_id}: event counts and denominators must be positive.")
        if self.active_events > self.active_n or self.control_events > self.control_n:
            raise ValidationError(f"{self.study_id}: events cannot exceed denominators.")
        if self.active_count_token != f"{self.active_events} of {self.active_n}":
            raise ValidationError(f"{self.study_id}: active_count_token does not match active counts.")
        if self.control_count_token != f"{self.control_events} of {self.control_n}":
            raise ValidationError(f"{self.study_id}: control_count_token does not match control counts.")
        if not self.active_source_terms or not self.control_source_terms:
            raise ValidationError(f"{self.study_id}: source terms must be non-empty.")
        if self.verified and not all(
            (
                self.active_count_token_found,
                self.control_count_token_found,
                self.active_term_near_count,
                self.control_term_near_count,
            )
        ):
            raise ValidationError(f"{self.study_id}: verified record is missing count or treatment evidence.")


@dataclass(frozen=True)
class EventCountVerificationReport:
    """One PubMed abstract event-count verification snapshot."""

    benchmark_id: str
    checked_at: str
    dataset: str
    dataset_sha256: str
    source_manifest: str
    source_manifest_sha256: str
    status: str
    certification_effect: str
    records: tuple[EventCountVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "EventCountVerificationReport":
        required = {
            "schema_version",
            "benchmark_id",
            "checked_at",
            "dataset",
            "dataset_sha256",
            "source_manifest",
            "source_manifest_sha256",
            "status",
            "certification_effect",
            "records",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"event-count verification report missing required keys: {missing}")
        if raw["schema_version"] != EVENT_COUNT_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                f"event-count verification schema_version must be {EVENT_COUNT_VERIFICATION_SCHEMA_VERSION}."
            )
        records = tuple(EventCountVerificationRecord.from_mapping(item) for item in raw["records"])
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            dataset=str(raw["dataset"]),
            dataset_sha256=str(raw["dataset_sha256"]),
            source_manifest=str(raw["source_manifest"]),
            source_manifest_sha256=str(raw["source_manifest_sha256"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            records=records,
        )
        report.validate()
        return report

    def validate(self) -> None:
        if not self.benchmark_id.strip():
            raise ValidationError("event-count verification benchmark_id must not be empty.")
        if not self.checked_at.strip():
            raise ValidationError("event-count verification checked_at must not be empty.")
        if not _looks_like_sha256(self.dataset_sha256):
            raise ValidationError("event-count verification dataset_sha256 is not a SHA-256 digest.")
        if not _looks_like_sha256(self.source_manifest_sha256):
            raise ValidationError("event-count verification source_manifest_sha256 is not a SHA-256 digest.")
        if self.status not in {"verified", "failed"}:
            raise ValidationError(f"event-count verification status '{self.status}' is not supported.")
        if self.certification_effect != "none":
            raise ValidationError("event-count verification reports cannot certify model performance.")
        if not self.records:
            raise ValidationError("event-count verification report must contain records.")
        if self.status == "verified" and any(not record.verified for record in self.records):
            raise ValidationError("verified event-count report cannot contain unverified records.")


def load_event_count_verification_report(path: str | Path) -> EventCountVerificationReport:
    """Load and validate one event-count verification JSON report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EventCountVerificationReport.from_mapping(payload)


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)
