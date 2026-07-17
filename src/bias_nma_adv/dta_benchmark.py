"""Source-backed diagnostic test accuracy benchmark support."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import string
import tomllib
from typing import Any
from urllib.parse import urlparse

from bias_nma_adv.data import ValidationError
from bias_nma_adv.dta import DTAStudy
from bias_nma_adv.dta import fit_bivariate_dta_reml
from bias_nma_adv.real_meta import sha256_file


DTA_MANIFEST_SCHEMA_VERSION = "dta_open_access_manifest/v1"
DTA_VERIFICATION_SCHEMA_VERSION = "dta_source_verification/v1"
DTA_BENCHMARK_SCHEMA_VERSION = "dta_benchmark/v1"

_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")


@dataclass(frozen=True)
class DTASourceRecord:
    """One source-backed diagnostic 2x2 row."""

    study_id: str
    citation: str
    country: str
    cancer: str
    index_test: str
    reference_standard: str
    threshold: str
    tp: int
    fp: int
    fn: int
    tn: int
    source_type: str
    source_doi: str
    table_doi: str
    table_id: str
    row_label: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DTASourceRecord":
        required = {
            "study_id",
            "citation",
            "country",
            "cancer",
            "index_test",
            "reference_standard",
            "threshold",
            "tp",
            "fp",
            "fn",
            "tn",
            "source_type",
            "source_doi",
            "table_doi",
            "table_id",
            "row_label",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"DTA source record missing required keys: {missing}")
        record = cls(
            study_id=str(raw["study_id"]),
            citation=str(raw["citation"]),
            country=str(raw["country"]),
            cancer=str(raw["cancer"]),
            index_test=str(raw["index_test"]),
            reference_standard=str(raw["reference_standard"]),
            threshold=str(raw["threshold"]),
            tp=int(raw["tp"]),
            fp=int(raw["fp"]),
            fn=int(raw["fn"]),
            tn=int(raw["tn"]),
            source_type=str(raw["source_type"]),
            source_doi=str(raw["source_doi"]),
            table_doi=str(raw["table_doi"]),
            table_id=str(raw["table_id"]),
            row_label=str(raw["row_label"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.study_id.strip():
            raise ValidationError("DTA study_id must not be empty.")
        for field in ("citation", "country", "cancer", "index_test", "reference_standard", "row_label"):
            if not getattr(self, field).strip():
                raise ValidationError(f"{self.study_id}: {field} must not be empty.")
        if self.source_type != "open_access_paper":
            raise ValidationError(f"{self.study_id}: DTA rows must come from an open-access paper.")
        if not _DOI_RE.match(self.source_doi) or not _DOI_RE.match(self.table_doi):
            raise ValidationError(f"{self.study_id}: malformed DOI.")
        if not self.table_id.strip():
            raise ValidationError(f"{self.study_id}: table_id must not be empty.")
        DTAStudy(
            study_id=self.study_id,
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
        ).validate()

    def to_dta_study(self) -> DTAStudy:
        return DTAStudy(
            study_id=self.study_id,
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
        )


@dataclass(frozen=True)
class DTAOpenAccessManifest:
    """Manifest for a DOI-backed open-access DTA table."""

    benchmark_id: str
    source_policy: str
    evidence_mode: str
    status: str
    certification_effect: str
    article_title: str
    article_doi: str
    source_url: str
    manuscript_xml_url: str
    table_id: str
    table_doi: str
    index_test: str
    reference_standard: str
    threshold_rule: str
    access_statement: str
    records: tuple[DTASourceRecord, ...]
    manifest_sha256: str | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        manifest_sha256: str | None = None,
    ) -> "DTAOpenAccessManifest":
        required = {
            "schema_version",
            "benchmark_id",
            "source_policy",
            "evidence_mode",
            "status",
            "certification_effect",
            "article_title",
            "article_doi",
            "source_url",
            "manuscript_xml_url",
            "table_id",
            "table_doi",
            "index_test",
            "reference_standard",
            "threshold_rule",
            "access_statement",
            "records",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"DTA manifest missing required keys: {missing}")
        if raw["schema_version"] != DTA_MANIFEST_SCHEMA_VERSION:
            raise ValidationError(
                f"DTA manifest schema_version must be {DTA_MANIFEST_SCHEMA_VERSION}."
            )
        manifest = cls(
            benchmark_id=str(raw["benchmark_id"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            article_title=str(raw["article_title"]),
            article_doi=str(raw["article_doi"]),
            source_url=str(raw["source_url"]),
            manuscript_xml_url=str(raw["manuscript_xml_url"]),
            table_id=str(raw["table_id"]),
            table_doi=str(raw["table_doi"]),
            index_test=str(raw["index_test"]),
            reference_standard=str(raw["reference_standard"]),
            threshold_rule=str(raw["threshold_rule"]),
            access_statement=str(raw["access_statement"]),
            records=tuple(DTASourceRecord.from_mapping(item) for item in raw["records"]),
            manifest_sha256=manifest_sha256,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.source_policy != "clinicaltrials_gov + pubmed_abstract + open_access_paper only":
            raise ValidationError("DTA source_policy is outside the evidence boundary.")
        if self.evidence_mode != "open_access_jats_table_2x2":
            raise ValidationError("DTA evidence_mode is unsupported.")
        if self.status != "candidate_source_verified":
            raise ValidationError("DTA manifest status must be candidate_source_verified.")
        if self.certification_effect != "none":
            raise ValidationError("DTA manifests cannot certify model performance.")
        if not _DOI_RE.match(self.article_doi) or not _DOI_RE.match(self.table_doi):
            raise ValidationError("DTA manifest has malformed DOI.")
        for url_field in ("source_url", "manuscript_xml_url"):
            parsed = urlparse(getattr(self, url_field))
            if parsed.scheme != "https" or parsed.hostname != "journals.plos.org":
                raise ValidationError(f"DTA {url_field} must be an HTTPS PLOS URL.")
        if self.table_id != "pone.0180511.t001":
            raise ValidationError("DTA manifest table_id does not match the source table.")
        if len(self.records) < 5:
            raise ValidationError("DTA manifest requires at least five 2x2 rows.")
        ids = [record.study_id for record in self.records]
        duplicates = sorted({study_id for study_id in ids if ids.count(study_id) > 1})
        if duplicates:
            raise ValidationError(f"DTA manifest contains duplicate study IDs: {duplicates}")
        for record in self.records:
            if record.source_doi != self.article_doi:
                raise ValidationError(f"{record.study_id}: record source DOI mismatch.")
            if record.table_doi != self.table_doi or record.table_id != self.table_id:
                raise ValidationError(f"{record.study_id}: record table locator mismatch.")
            if record.index_test != self.index_test:
                raise ValidationError(f"{record.study_id}: record index_test mismatch.")
            if record.reference_standard != self.reference_standard:
                raise ValidationError(f"{record.study_id}: record reference_standard mismatch.")
        if "open access" not in self.access_statement.lower():
            raise ValidationError("DTA access_statement must identify the paper as open access.")


@dataclass(frozen=True)
class DTAVerificationRecord:
    """One verified DTA source row from the open-access table."""

    study_id: str
    source_type: str
    evidence_scope: str
    row_label: str
    tp: int
    fp: int
    fn: int
    tn: int
    row_tokens_found: bool
    verified: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DTAVerificationRecord":
        required = {
            "study_id",
            "source_type",
            "evidence_scope",
            "row_label",
            "tp",
            "fp",
            "fn",
            "tn",
            "row_tokens_found",
            "verified",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"DTA verification record missing keys: {missing}")
        record = cls(
            study_id=str(raw["study_id"]),
            source_type=str(raw["source_type"]),
            evidence_scope=str(raw["evidence_scope"]),
            row_label=str(raw["row_label"]),
            tp=int(raw["tp"]),
            fp=int(raw["fp"]),
            fn=int(raw["fn"]),
            tn=int(raw["tn"]),
            row_tokens_found=bool(raw["row_tokens_found"]),
            verified=bool(raw["verified"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.source_type != "open_access_paper":
            raise ValidationError(f"{self.study_id}: DTA verification source_type is unsupported.")
        if self.evidence_scope != "open_access_jats_table_2x2":
            raise ValidationError(f"{self.study_id}: unsupported DTA evidence_scope.")
        DTAStudy(
            study_id=self.study_id,
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
        ).validate()
        if self.verified and not self.row_tokens_found:
            raise ValidationError(f"{self.study_id}: verified DTA row lacks source-token evidence.")


@dataclass(frozen=True)
class DTAVerificationReport:
    """Source verification report for an open-access DTA table."""

    benchmark_id: str
    checked_at: str
    manifest: str
    manifest_sha256: str
    status: str
    certification_effect: str
    article_doi: str
    source_url: str
    table_id: str
    table_doi: str
    table_sha256: str
    records: tuple[DTAVerificationRecord, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DTAVerificationReport":
        required = {
            "schema_version",
            "benchmark_id",
            "checked_at",
            "manifest",
            "manifest_sha256",
            "status",
            "certification_effect",
            "article_doi",
            "source_url",
            "table_id",
            "table_doi",
            "table_sha256",
            "records",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValidationError(f"DTA verification report missing keys: {missing}")
        if raw["schema_version"] != DTA_VERIFICATION_SCHEMA_VERSION:
            raise ValidationError(
                f"DTA verification schema_version must be {DTA_VERIFICATION_SCHEMA_VERSION}."
            )
        report = cls(
            benchmark_id=str(raw["benchmark_id"]),
            checked_at=str(raw["checked_at"]),
            manifest=str(raw["manifest"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            article_doi=str(raw["article_doi"]),
            source_url=str(raw["source_url"]),
            table_id=str(raw["table_id"]),
            table_doi=str(raw["table_doi"]),
            table_sha256=str(raw["table_sha256"]),
            records=tuple(DTAVerificationRecord.from_mapping(item) for item in raw["records"]),
        )
        report.validate()
        return report

    def validate(self) -> None:
        if self.status != "verified":
            raise ValidationError("DTA verification report status must be verified.")
        if self.certification_effect != "none":
            raise ValidationError("DTA verification report cannot certify.")
        if not _DOI_RE.match(self.article_doi) or not _DOI_RE.match(self.table_doi):
            raise ValidationError("DTA verification report has malformed DOI.")
        if not _looks_like_sha256(self.manifest_sha256) or not _looks_like_sha256(self.table_sha256):
            raise ValidationError("DTA verification report SHA values are malformed.")
        if not self.records:
            raise ValidationError("DTA verification report must include records.")
        if self.status == "verified" and any(not record.verified for record in self.records):
            raise ValidationError("verified DTA report cannot contain unverified rows.")


def load_dta_manifest(path: str | Path) -> DTAOpenAccessManifest:
    manifest_path = Path(path)
    with manifest_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return DTAOpenAccessManifest.from_mapping(
        payload,
        manifest_sha256=sha256_file(manifest_path),
    )


def load_dta_verification_report(path: str | Path) -> DTAVerificationReport:
    return DTAVerificationReport.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def validate_dta_source_bundle(
    manifest: DTAOpenAccessManifest,
    report: DTAVerificationReport,
) -> dict[str, Any]:
    """Validate that the source verification rows match the DTA manifest."""

    if manifest.benchmark_id != report.benchmark_id:
        raise ValidationError("DTA verification benchmark_id mismatch.")
    if manifest.manifest_sha256 is not None and report.manifest_sha256 != manifest.manifest_sha256:
        raise ValidationError("DTA verification manifest SHA mismatch.")
    if manifest.article_doi != report.article_doi:
        raise ValidationError("DTA verification article DOI mismatch.")
    if manifest.source_url != report.source_url:
        raise ValidationError("DTA verification source URL mismatch.")
    if manifest.table_id != report.table_id or manifest.table_doi != report.table_doi:
        raise ValidationError("DTA verification table locator mismatch.")
    by_id = {record.study_id: record for record in report.records}
    if set(by_id) != {record.study_id for record in manifest.records}:
        raise ValidationError("DTA verification records do not match manifest rows.")
    for record in manifest.records:
        observed = by_id[record.study_id]
        expected_counts = (record.tp, record.fp, record.fn, record.tn)
        observed_counts = (observed.tp, observed.fp, observed.fn, observed.tn)
        if observed_counts != expected_counts:
            raise ValidationError(f"{record.study_id}: DTA verification counts mismatch.")
    return {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": report.status,
        "certification_effect": report.certification_effect,
        "n_studies": len(manifest.records),
        "source_counts": {"open_access_paper": len(manifest.records)},
    }


def run_dta_benchmark(
    manifest_path: str | Path,
    *,
    verification_report_path: str | Path,
) -> dict[str, Any]:
    """Run the local bivariate DTA prototype over source-backed 2x2 rows."""

    manifest = load_dta_manifest(manifest_path)
    report = load_dta_verification_report(verification_report_path)
    source_bundle = validate_dta_source_bundle(manifest, report)
    fit = fit_bivariate_dta_reml(
        [record.to_dta_study() for record in manifest.records],
        continuity_correction=0.5,
        correction_control="all",
    )
    return {
        "schema_version": DTA_BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": manifest.benchmark_id,
        "status": "local_pass",
        "certification_effect": "none",
        "source_policy": manifest.source_policy,
        "evidence_mode": manifest.evidence_mode,
        "effect_scale": "logit_sensitivity_and_logit_false_positive_rate",
        "source_manifest": _relpath(Path(manifest_path), Path.cwd()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_verification_report": _relpath(Path(verification_report_path), Path.cwd()),
        "source_verification_report_sha256": sha256_file(verification_report_path),
        "source_bundle": source_bundle,
        "n_studies": len(manifest.records),
        "limitations": [
            "open-access table rows are source-backed but not independently re-extracted from primary studies",
            "not HSROC reference matched",
            "not clinical diagnostic accuracy guidance",
            "does not certify model performance",
        ],
        "model_config": {
            "index_test": manifest.index_test,
            "reference_standard": manifest.reference_standard,
            "threshold_rule": manifest.threshold_rule,
            "candidate_model": fit.method,
            "continuity_correction": 0.5,
            "correction_control": "all",
        },
        "study_effects": [asdict(record) for record in manifest.records],
        "candidate": {
            "bivariate_logitnormal_reml": fit.to_dict(),
        },
    }


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
