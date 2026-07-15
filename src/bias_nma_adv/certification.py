"""Certification gates for reference matching and production claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
