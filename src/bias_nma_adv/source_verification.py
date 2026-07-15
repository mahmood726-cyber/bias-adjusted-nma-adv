"""Validation for live public-source verification snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import string
from typing import Any

from bias_nma_adv.data import ValidationError
from bias_nma_adv.evidence_sources import ALLOWED_SOURCE_TYPES


SOURCE_VERIFICATION_SCHEMA_VERSION = "source_verification/v1"

ALLOWED_SOURCE_VERIFICATION_STATUSES = {
    "verified",
    "partial",
    "failed",
}


@dataclass(frozen=True)
class SourceVerificationRecord:
    """One live-source identity check for a manifest source entry."""

    study_id: str
    source_type: str
    identifier: str
    manifest_url: str
    api_url: str
    http_status: int
    identity_verified: bool
    response_sha256: str
    title: str
    evidence_scope: str
    details: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SourceVerificationRecord":
        required = {
            "study_id",
            "source_type",
            "identifier",
            "manifest_url",
            "api_url",
            "http_status",
            "identity_verified",
            "response_sha256",
            "title",
            "evidence_scope",
            "details",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"source verification record missing required keys: {missing}")
        record = cls(
            study_id=str(raw["study_id"]),
            source_type=str(raw["source_type"]),
            identifier=str(raw["identifier"]),
            manifest_url=str(raw["manifest_url"]),
            api_url=str(raw["api_url"]),
            http_status=int(raw["http_status"]),
            identity_verified=bool(raw["identity_verified"]),
            response_sha256=str(raw["response_sha256"]),
            title=str(raw["title"]),
            evidence_scope=str(raw["evidence_scope"]),
            details=dict(raw["details"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("source verification study_id must not be empty.")
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValidationError(f"{self.study_id}: unsupported source_type '{self.source_type}'.")
        if self.source_type not in {"clinicaltrials_gov", "pubmed_abstract"}:
            raise ValidationError(
                f"{self.study_id}: source verification currently supports CT.gov and PubMed abstract records."
            )
        if not self.identifier.strip():
            raise ValidationError(f"{self.study_id}: identifier must not be empty.")
        if not self.manifest_url.startswith(("https://", "http://")):
            raise ValidationError(f"{self.study_id}: manifest_url must be absolute HTTP(S).")
        if not self.api_url.startswith(("https://", "http://")):
            raise ValidationError(f"{self.study_id}: api_url must be absolute HTTP(S).")
        if self.http_status != 200:
            raise ValidationError(f"{self.study_id}: source verification HTTP status is not 200.")
        if not self.identity_verified:
            raise ValidationError(f"{self.study_id}: source identity was not verified.")
        if not _looks_like_sha256(self.response_sha256):
            raise ValidationError(f"{self.study_id}: response_sha256 is not a SHA-256 digest.")
        if not self.title.strip():
            raise ValidationError(f"{self.study_id}: source title must not be empty.")
        if self.evidence_scope != "identity_and_reachability":
            raise ValidationError(
                f"{self.study_id}: unsupported evidence_scope '{self.evidence_scope}'."
            )


@dataclass(frozen=True)
class SourceVerificationReport:
    """One source-check snapshot for a source manifest."""

    benchmark_id: str
    checked_at: str
    source_manifest: str
    source_manifest_sha256: str
    status: str
    records: tuple[SourceVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SourceVerificationReport":
        required = {
            "schema_version",
            "benchmark_id",
            "checked_at",
            "source_manifest",
            "source_manifest_sha256",
            "status",
            "records",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"source verification report missing required keys: {missing}")
        if raw["schema_version"] != SOURCE_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                f"source verification schema_version must be {SOURCE_VERIFICATION_SCHEMA_VERSION}."
            )
        records = tuple(SourceVerificationRecord.from_mapping(item) for item in raw["records"])
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            source_manifest=str(raw["source_manifest"]),
            source_manifest_sha256=str(raw["source_manifest_sha256"]),
            status=str(raw["status"]),
            records=records,
        )
        report.validate()
        return report

    def validate(self) -> None:
        if not self.benchmark_id.strip():
            raise ValidationError("source verification benchmark_id must not be empty.")
        if not self.checked_at.strip():
            raise ValidationError("source verification checked_at must not be empty.")
        if not self.source_manifest.strip():
            raise ValidationError("source verification source_manifest must not be empty.")
        if not _looks_like_sha256(self.source_manifest_sha256):
            raise ValidationError("source verification source_manifest_sha256 is not a SHA-256 digest.")
        if self.status not in ALLOWED_SOURCE_VERIFICATION_STATUSES:
            raise ValidationError(f"source verification status '{self.status}' is not supported.")
        if not self.records:
            raise ValidationError("source verification report must contain records.")
        failures = [record for record in self.records if not record.identity_verified]
        if self.status == "verified" and failures:
            raise ValidationError("verified source report cannot contain failed identity checks.")


def load_source_verification_report(path: str | Path) -> SourceVerificationReport:
    """Load and validate one source-verification JSON report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SourceVerificationReport.from_mapping(payload)


def summarize_source_verification(report: SourceVerificationReport) -> dict[str, int]:
    """Count source-verification records by source type."""

    summary: dict[str, int] = {}
    for record in report.records:
        summary[record.source_type] = summary.get(record.source_type, 0) + 1
    return dict(sorted(summary.items()))


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)
