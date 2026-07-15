"""Certification gates for reference matching and production claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string
import tomllib
from typing import Any


class CertificationError(ValueError):
    """Raised when certification metadata is malformed or overclaims evidence."""


ALLOWED_STATUSES = {
    "planned",
    "experimental",
    "numerically_verified",
    "reference_matched",
    "simulation_validated",
    "externally_reproduced",
    "production_certified",
}

EVIDENCE_REQUIRED_STATUSES = {
    "reference_matched",
    "simulation_validated",
    "externally_reproduced",
    "production_certified",
}

PRODUCTION_REQUIRED_STATUSES = {
    "reference_matched",
    "simulation_validated",
    "externally_reproduced",
}

REFERENCE_RUN_SCHEMA_VERSION = "reference_run/v1"

ALLOWED_REFERENCE_RUN_STATUSES = {
    "unavailable",
    "failed",
    "passed",
}

ALLOWED_CERTIFICATION_EFFECTS = {
    "none",
    "evidence_candidate",
}


@dataclass(frozen=True)
class ReferenceTarget:
    """One external benchmark target for a platform module."""

    id: str
    domain: str
    module: str
    reference_method: str
    status: str
    acceptance_criteria: tuple[str, ...]
    evidence_artifacts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ReferenceTarget":
        missing = {
            key
            for key in (
                "id",
                "domain",
                "module",
                "reference_method",
                "status",
                "acceptance_criteria",
                "evidence_artifacts",
            )
            if key not in raw
        }
        if missing:
            raise CertificationError(f"Reference target missing required keys: {sorted(missing)}")

        criteria = tuple(str(item) for item in raw["acceptance_criteria"])
        artifacts = tuple(str(item) for item in raw["evidence_artifacts"])
        target = cls(
            id=str(raw["id"]),
            domain=str(raw["domain"]),
            module=str(raw["module"]),
            reference_method=str(raw["reference_method"]),
            status=str(raw["status"]),
            acceptance_criteria=criteria,
            evidence_artifacts=artifacts,
        )
        target.validate()
        return target

    def validate(self) -> None:
        if not self.id:
            raise CertificationError("Reference target id must not be empty.")
        if self.status not in ALLOWED_STATUSES:
            raise CertificationError(f"{self.id}: unsupported status '{self.status}'.")
        if not self.acceptance_criteria:
            raise CertificationError(f"{self.id}: acceptance_criteria must not be empty.")
        if self.status in EVIDENCE_REQUIRED_STATUSES and not self.evidence_artifacts:
            raise CertificationError(
                f"{self.id}: status '{self.status}' requires evidence_artifacts."
            )


@dataclass(frozen=True)
class ReferenceRunReport:
    """One machine-readable execution report from an external reference adapter."""

    target_id: str
    adapter_id: str
    reference_method: str
    status: str
    certification_effect: str
    command: tuple[str, ...]
    executable: str
    executable_found: bool
    package_versions: dict[str, str]
    input_artifacts: tuple[str, ...]
    input_sha256: dict[str, str]
    output_artifacts: tuple[str, ...]
    output_sha256: dict[str, str]
    tolerance: str
    skip_reason: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ReferenceRunReport":
        missing = {
            key
            for key in (
                "schema_version",
                "target_id",
                "adapter_id",
                "reference_method",
                "status",
                "certification_effect",
                "command",
                "executable",
                "executable_found",
                "package_versions",
                "input_artifacts",
                "input_sha256",
                "output_artifacts",
                "output_sha256",
                "tolerance",
                "skip_reason",
            )
            if key not in raw
        }
        if missing:
            raise CertificationError(f"Reference run report missing required keys: {sorted(missing)}")
        if raw["schema_version"] != REFERENCE_RUN_SCHEMA_VERSION:
            raise CertificationError(
                f"reference run schema_version must be {REFERENCE_RUN_SCHEMA_VERSION}."
            )

        report = cls(
            target_id=str(raw["target_id"]),
            adapter_id=str(raw["adapter_id"]),
            reference_method=str(raw["reference_method"]),
            status=str(raw["status"]),
            certification_effect=str(raw["certification_effect"]),
            command=tuple(str(item) for item in raw["command"]),
            executable=str(raw["executable"]),
            executable_found=bool(raw["executable_found"]),
            package_versions={str(key): str(value) for key, value in raw["package_versions"].items()},
            input_artifacts=tuple(str(item) for item in raw["input_artifacts"]),
            input_sha256={str(key): str(value) for key, value in raw["input_sha256"].items()},
            output_artifacts=tuple(str(item) for item in raw["output_artifacts"]),
            output_sha256={str(key): str(value) for key, value in raw["output_sha256"].items()},
            tolerance=str(raw["tolerance"]),
            skip_reason=str(raw["skip_reason"]),
        )
        report.validate()
        return report

    @property
    def is_certification_evidence_candidate(self) -> bool:
        return self.status == "passed" and self.certification_effect == "evidence_candidate"

    def validate(self) -> None:
        if not self.target_id.strip():
            raise CertificationError("Reference run target_id must not be empty.")
        if not self.adapter_id.strip():
            raise CertificationError(f"{self.target_id}: adapter_id must not be empty.")
        if self.status not in ALLOWED_REFERENCE_RUN_STATUSES:
            raise CertificationError(f"{self.target_id}: unsupported reference run status '{self.status}'.")
        if self.certification_effect not in ALLOWED_CERTIFICATION_EFFECTS:
            raise CertificationError(
                f"{self.target_id}: unsupported certification_effect '{self.certification_effect}'."
            )
        if not self.command:
            raise CertificationError(f"{self.target_id}: command must not be empty.")
        if not self.executable.strip():
            raise CertificationError(f"{self.target_id}: executable must not be empty.")

        if self.status != "passed" and self.certification_effect != "none":
            raise CertificationError(
                f"{self.target_id}: non-passed reference runs cannot be certification evidence."
            )
        if self.status == "unavailable" and not self.skip_reason.strip():
            raise CertificationError(f"{self.target_id}: unavailable reference run requires skip_reason.")
        self._validate_artifact_hashes()
        if self.status == "passed":
            self._validate_passed_report()

    def _validate_artifact_hashes(self) -> None:
        missing_input_hashes = sorted(set(self.input_artifacts) - set(self.input_sha256))
        missing_output_hashes = sorted(set(self.output_artifacts) - set(self.output_sha256))
        if missing_input_hashes:
            raise CertificationError(
                f"{self.target_id}: missing input SHA-256 entries for {missing_input_hashes}."
            )
        if missing_output_hashes:
            raise CertificationError(
                f"{self.target_id}: missing output SHA-256 entries for {missing_output_hashes}."
            )
        for label, digest in {**self.input_sha256, **self.output_sha256}.items():
            if not _looks_like_sha256(digest):
                raise CertificationError(f"{self.target_id}: invalid SHA-256 for {label}.")

    def _validate_passed_report(self) -> None:
        if not self.executable_found:
            raise CertificationError(f"{self.target_id}: passed reference run requires executable_found=true.")
        if self.skip_reason.strip():
            raise CertificationError(f"{self.target_id}: passed reference run must not have skip_reason.")
        if not self.package_versions:
            raise CertificationError(f"{self.target_id}: passed reference run requires package_versions.")
        if not self.input_artifacts or not self.output_artifacts:
            raise CertificationError(
                f"{self.target_id}: passed reference run requires input and output artifacts."
            )
        if not self.tolerance.strip():
            raise CertificationError(f"{self.target_id}: passed reference run requires tolerance.")


def load_reference_targets(path: str | Path) -> list[ReferenceTarget]:
    """Load and validate the reference-target registry."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise CertificationError("reference target payload must contain a list named 'targets'.")

    targets = [ReferenceTarget.from_mapping(raw) for raw in raw_targets]
    ids = [target.id for target in targets]
    duplicates = sorted({target_id for target_id in ids if ids.count(target_id) > 1})
    if duplicates:
        raise CertificationError(f"Duplicate reference target ids: {duplicates}")
    return targets


def load_reference_run_report(path: str | Path) -> ReferenceRunReport:
    """Load and validate one external reference-adapter execution report."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return ReferenceRunReport.from_mapping(payload)


def load_reference_run_reports(path: str | Path) -> list[ReferenceRunReport]:
    """Load one report file or all TOML reports in a directory."""

    root = Path(path)
    if root.is_file():
        return [load_reference_run_report(root)]
    if not root.is_dir():
        raise CertificationError(f"reference run path does not exist: {root}")
    return [load_reference_run_report(report_path) for report_path in sorted(root.glob("*.toml"))]


def summarize_reference_run_reports(reports: list[ReferenceRunReport]) -> dict[str, int]:
    """Count external reference-adapter reports by execution status."""

    summary = {status: 0 for status in ALLOWED_REFERENCE_RUN_STATUSES}
    for report in reports:
        report.validate()
        summary[report.status] += 1
    return {status: count for status, count in sorted(summary.items()) if count}


def assert_reference_runs_target_known(
    targets: list[ReferenceTarget],
    reports: list[ReferenceRunReport],
) -> None:
    """Fail if a report points at a target absent from the reference-target registry."""

    target_ids = {target.id for target in targets}
    unknown = sorted({report.target_id for report in reports} - target_ids)
    if unknown:
        raise CertificationError(f"Reference run reports point at unknown targets: {unknown}")


def certification_candidate_artifacts(reports: list[ReferenceRunReport]) -> tuple[str, ...]:
    """Return output artifacts from passed reports only."""

    artifacts: list[str] = []
    for report in reports:
        if report.is_certification_evidence_candidate:
            artifacts.extend(report.output_artifacts)
    return tuple(artifacts)


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)


def summarize_reference_targets(targets: list[ReferenceTarget]) -> dict[str, int]:
    """Count targets by certification status."""

    summary = {status: 0 for status in ALLOWED_STATUSES}
    for target in targets:
        target.validate()
        summary[target.status] += 1
    return {status: count for status, count in sorted(summary.items()) if count}


def assert_no_unsupported_production_claims(targets: list[ReferenceTarget]) -> None:
    """Fail closed if a production target lacks all prerequisite evidence levels."""

    for target in targets:
        if target.status != "production_certified":
            continue
        evidence_text = "\n".join(target.evidence_artifacts).lower()
        missing = [
            status
            for status in sorted(PRODUCTION_REQUIRED_STATUSES)
            if status not in evidence_text
        ]
        if missing:
            raise CertificationError(
                f"{target.id}: production certification lacks evidence markers for {missing}."
            )
