"""Allowed evidence-source policy for real-world validation data."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import re
import tomllib
from urllib.parse import urlparse


class EvidenceSourceError(ValueError):
    """Raised when a source is outside the allowed evidence boundary."""


SOURCE_TYPE_POLICY_SCHEMA_VERSION = "source_type_policy/v1"


@dataclass(frozen=True)
class SourceTypePolicy:
    """Validation policy for one evidence source type.

    Source types are loaded from ``source_type_policy.toml`` so widening the
    boundary is a policy-data change rather than a rewrite of every validator.
    """

    id: str
    role: str
    category: str
    identifier_pattern: str | None
    identifier_error: str
    allowed_hosts: tuple[str, ...]
    host_error: str
    required_statement_all: tuple[str, ...]
    required_statement_any: tuple[str, ...]
    forbidden_statement_any: tuple[str, ...]
    statement_error: str
    forbidden_statement_error: str

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "SourceTypePolicy":
        return cls(
            id=str(raw["id"]),
            role=str(raw["role"]),
            category=str(raw["category"]),
            identifier_pattern=(
                str(raw["identifier_pattern"]).strip()
                if raw.get("identifier_pattern") is not None
                and str(raw["identifier_pattern"]).strip()
                else None
            ),
            identifier_error=str(raw.get("identifier_error") or "source identifier is malformed."),
            allowed_hosts=tuple(str(item).lower() for item in raw.get("allowed_hosts", [])),
            host_error=str(raw.get("host_error") or "source URL host is outside the allowed boundary."),
            required_statement_all=tuple(
                str(item).lower() for item in raw.get("required_statement_all", [])
            ),
            required_statement_any=tuple(
                str(item).lower() for item in raw.get("required_statement_any", [])
            ),
            forbidden_statement_any=tuple(
                str(item).lower() for item in raw.get("forbidden_statement_any", [])
            ),
            statement_error=str(raw.get("statement_error") or "access_statement is insufficient."),
            forbidden_statement_error=str(
                raw.get("forbidden_statement_error") or "access_statement contains a forbidden term."
            ),
        )


def load_source_type_policies() -> dict[str, SourceTypePolicy]:
    """Load source-type policies bundled with the package."""

    policy_text = resources.files(__package__).joinpath("source_type_policy.toml").read_text(
        encoding="utf-8"
    )
    raw = tomllib.loads(policy_text)
    if raw.get("schema_version") != SOURCE_TYPE_POLICY_SCHEMA_VERSION:
        raise EvidenceSourceError(
            f"source type policy schema_version must be {SOURCE_TYPE_POLICY_SCHEMA_VERSION}."
        )
    policies = {
        policy.id: policy
        for policy in (
            SourceTypePolicy.from_mapping(item) for item in raw.get("source_types", [])
        )
    }
    if not policies:
        raise EvidenceSourceError("source type policy must define at least one source type.")
    if len(policies) != len(raw.get("source_types", [])):
        raise EvidenceSourceError("source type policy contains duplicate source type ids.")
    return policies


SOURCE_TYPE_POLICIES = load_source_type_policies()
EFFECT_EVIDENCE_SOURCE_TYPES = {
    source_type
    for source_type, policy in SOURCE_TYPE_POLICIES.items()
    if policy.role == "effect"
}
REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES = {
    source_type
    for source_type, policy in SOURCE_TYPE_POLICIES.items()
    if policy.role == "effect" and policy.category == "trial_registry_results"
}
REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES = {
    source_type
    for source_type, policy in SOURCE_TYPE_POLICIES.items()
    if policy.role == "effect" and policy.category == "regulatory_review"
}
PROTOCOL_ONLY_SOURCE_TYPES = {
    source_type
    for source_type, policy in SOURCE_TYPE_POLICIES.items()
    if policy.role == "protocol_only"
}
ALLOWED_SOURCE_TYPES = EFFECT_EVIDENCE_SOURCE_TYPES | PROTOCOL_ONLY_SOURCE_TYPES


@dataclass(frozen=True)
class EvidenceSource:
    """Source metadata for one extracted clinical datum."""

    source_type: str
    identifier: str
    url: str
    access_statement: str

    def validate(self) -> None:
        policy = SOURCE_TYPE_POLICIES.get(self.source_type)
        if policy is None:
            raise EvidenceSourceError(
                f"source_type '{self.source_type}' is not allowed; use one of "
                f"{sorted(ALLOWED_SOURCE_TYPES)}."
            )
        if not self.identifier.strip():
            raise EvidenceSourceError("identifier must not be empty.")
        if not self.url.startswith(("https://", "http://")):
            raise EvidenceSourceError("url must be an absolute HTTP(S) URL.")
        if not self.access_statement.strip():
            raise EvidenceSourceError("access_statement must describe why the source is admissible.")

        _validate_identifier(self.identifier, policy)
        _validate_host(self.url, policy)
        _validate_statement(self.access_statement, policy)


def validate_sources(sources: list[EvidenceSource]) -> None:
    """Validate a non-empty source list."""

    if not sources:
        raise EvidenceSourceError("at least one evidence source is required.")
    for source in sources:
        source.validate()


def _validate_identifier(identifier: str, policy: SourceTypePolicy) -> None:
    if policy.identifier_pattern and re.match(policy.identifier_pattern, identifier) is None:
        raise EvidenceSourceError(policy.identifier_error)


def _validate_host(url: str, policy: SourceTypePolicy) -> None:
    if not policy.allowed_hosts:
        return
    host = (urlparse(url).hostname or "").lower()
    if not any(_host_matches(host, pattern) for pattern in policy.allowed_hosts):
        raise EvidenceSourceError(policy.host_error)


def _validate_statement(statement: str, policy: SourceTypePolicy) -> None:
    lowered = statement.lower()
    if any(token in lowered for token in policy.forbidden_statement_any):
        raise EvidenceSourceError(policy.forbidden_statement_error)
    if policy.required_statement_all and not all(
        token in lowered for token in policy.required_statement_all
    ):
        raise EvidenceSourceError(policy.statement_error)
    if policy.required_statement_any and not any(
        token in lowered for token in policy.required_statement_any
    ):
        raise EvidenceSourceError(policy.statement_error)


def _host_matches(host: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        suffix = pattern[1:]
        return host.endswith(suffix) and host != suffix.lstrip(".")
    return host == pattern


def _optional_string(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
