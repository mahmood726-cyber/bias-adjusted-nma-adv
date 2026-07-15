"""Registry checks for local source-backed benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import string
import tomllib
from typing import Any

from bias_nma_adv.ctgov_hr_network import (
    CTGOV_HR_NETWORK_VERIFICATION_SCHEMA_VERSION,
    CTGovHRNetworkVerificationReport,
    load_ctgov_hr_network_manifest,
    validate_ctgov_hr_network_source_bundle,
)
from bias_nma_adv.data import ValidationError
from bias_nma_adv.evidence_sources import EFFECT_EVIDENCE_SOURCE_TYPES
from bias_nma_adv.event_count_verification import (
    EVENT_COUNT_VERIFICATION_SCHEMA_VERSION,
    EventCountVerificationReport,
)
from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.source_verification import (
    SOURCE_VERIFICATION_SCHEMA_VERSION,
    SourceVerificationReport,
)
from bias_nma_adv.survival_benchmark import (
    SURVIVAL_HR_VERIFICATION_SCHEMA_VERSION,
    SurvivalHRVerificationReport,
    load_survival_hr_manifest,
    validate_survival_hr_identity_bundle,
    validate_survival_hr_source_bundle,
)


BENCHMARK_REGISTRY_SCHEMA_VERSION = "source_benchmark_registry/v1"
ALLOWED_SOURCE_POLICIES = {
    "clinicaltrials_gov + pubmed_abstract only",
    "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
}
SUPPORTED_SOURCE_CHECK_SCHEMA_VERSIONS = {
    SOURCE_VERIFICATION_SCHEMA_VERSION,
    EVENT_COUNT_VERIFICATION_SCHEMA_VERSION,
    SURVIVAL_HR_VERIFICATION_SCHEMA_VERSION,
    CTGOV_HR_NETWORK_VERIFICATION_SCHEMA_VERSION,
}


class BenchmarkRegistryError(ValueError):
    """Raised when the local benchmark registry is malformed or overclaims."""


@dataclass(frozen=True)
class SourceBenchmarkEntry:
    """One local source-backed benchmark registry entry."""

    id: str
    domain: str
    artifact_path: str
    artifact_sha256: str
    artifact_schema_version: str
    artifact_status: str
    source_policy: str
    evidence_mode: str
    certification_effect: str
    certification_scope: str
    n_studies: int
    source_manifests: tuple[str, ...]
    source_manifest_sha256: dict[str, str]
    source_checks: tuple[str, ...]
    source_check_sha256: dict[str, str]
    required_limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SourceBenchmarkEntry":
        required = {
            "id",
            "domain",
            "artifact_path",
            "artifact_sha256",
            "artifact_schema_version",
            "artifact_status",
            "source_policy",
            "evidence_mode",
            "certification_effect",
            "certification_scope",
            "n_studies",
            "source_manifests",
            "source_manifest_sha256",
            "source_checks",
            "source_check_sha256",
            "required_limitations",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise BenchmarkRegistryError(f"benchmark registry entry missing required keys: {missing}")
        entry = cls(
            id=str(raw["id"]),
            domain=str(raw["domain"]),
            artifact_path=str(raw["artifact_path"]),
            artifact_sha256=str(raw["artifact_sha256"]),
            artifact_schema_version=str(raw["artifact_schema_version"]),
            artifact_status=str(raw["artifact_status"]),
            source_policy=str(raw["source_policy"]),
            evidence_mode=str(raw["evidence_mode"]),
            certification_effect=str(raw["certification_effect"]),
            certification_scope=str(raw["certification_scope"]),
            n_studies=int(raw["n_studies"]),
            source_manifests=tuple(str(item) for item in raw["source_manifests"]),
            source_manifest_sha256={
                str(key): str(value) for key, value in raw["source_manifest_sha256"].items()
            },
            source_checks=tuple(str(item) for item in raw["source_checks"]),
            source_check_sha256={
                str(key): str(value) for key, value in raw["source_check_sha256"].items()
            },
            required_limitations=tuple(str(item) for item in raw["required_limitations"]),
        )
        entry.validate_metadata()
        return entry

    def validate_metadata(self) -> None:
        if not self.id.strip():
            raise BenchmarkRegistryError("benchmark registry id must not be empty.")
        if not self.domain.strip():
            raise BenchmarkRegistryError(f"{self.id}: domain must not be empty.")
        if not self.artifact_path.strip():
            raise BenchmarkRegistryError(f"{self.id}: artifact_path must not be empty.")
        if not _looks_like_sha256(self.artifact_sha256):
            raise BenchmarkRegistryError(f"{self.id}: artifact_sha256 is not a SHA-256 digest.")
        if self.source_policy not in ALLOWED_SOURCE_POLICIES:
            raise BenchmarkRegistryError(f"{self.id}: source_policy is outside the allowed evidence boundary.")
        if self.certification_effect != "none":
            raise BenchmarkRegistryError(f"{self.id}: local source benchmarks cannot certify model performance.")
        if self.certification_scope != "local_source_verified_only":
            raise BenchmarkRegistryError(f"{self.id}: certification_scope must be local_source_verified_only.")
        if self.n_studies <= 0:
            raise BenchmarkRegistryError(f"{self.id}: n_studies must be positive.")
        if not self.source_manifests:
            raise BenchmarkRegistryError(f"{self.id}: source_manifests must not be empty.")
        if not self.source_checks:
            raise BenchmarkRegistryError(f"{self.id}: source_checks must not be empty.")
        if not self.required_limitations:
            raise BenchmarkRegistryError(f"{self.id}: required_limitations must not be empty.")
        self._validate_hash_map("source_manifest_sha256", self.source_manifests, self.source_manifest_sha256)
        self._validate_hash_map("source_check_sha256", self.source_checks, self.source_check_sha256)

    def _validate_hash_map(self, label: str, paths: tuple[str, ...], digests: dict[str, str]) -> None:
        missing = sorted(set(paths) - set(digests))
        extra = sorted(set(digests) - set(paths))
        if missing:
            raise BenchmarkRegistryError(f"{self.id}: {label} missing entries for {missing}.")
        if extra:
            raise BenchmarkRegistryError(f"{self.id}: {label} has extra entries for {extra}.")
        for path, digest in digests.items():
            if not _looks_like_sha256(digest):
                raise BenchmarkRegistryError(f"{self.id}: {label} has invalid SHA-256 for {path}.")


@dataclass(frozen=True)
class SourceBenchmarkRegistry:
    """Machine-readable registry for local source-backed benchmark artifacts."""

    checked_at: str
    purpose: str
    allowed_evidence_sources: tuple[str, ...]
    certification_rule: str
    benchmarks: tuple[SourceBenchmarkEntry, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SourceBenchmarkRegistry":
        required = {
            "schema_version",
            "checked_at",
            "purpose",
            "allowed_evidence_sources",
            "certification_rule",
            "benchmarks",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise BenchmarkRegistryError(f"benchmark registry missing required keys: {missing}")
        if raw["schema_version"] != BENCHMARK_REGISTRY_SCHEMA_VERSION:
            raise BenchmarkRegistryError(
                f"benchmark registry schema_version must be {BENCHMARK_REGISTRY_SCHEMA_VERSION}."
            )
        entries = tuple(SourceBenchmarkEntry.from_mapping(item) for item in raw["benchmarks"])
        registry = cls(
            checked_at=str(raw["checked_at"]),
            purpose=str(raw["purpose"]),
            allowed_evidence_sources=tuple(str(item) for item in raw["allowed_evidence_sources"]),
            certification_rule=str(raw["certification_rule"]),
            benchmarks=entries,
        )
        registry.validate_metadata()
        return registry

    def validate_metadata(self) -> None:
        if set(self.allowed_evidence_sources) != EFFECT_EVIDENCE_SOURCE_TYPES:
            raise BenchmarkRegistryError("benchmark registry allowed_evidence_sources drifted.")
        if "certification_effect = none" not in self.certification_rule:
            raise BenchmarkRegistryError("benchmark registry must state the non-certification rule.")
        ids = [entry.id for entry in self.benchmarks]
        duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
        if duplicates:
            raise BenchmarkRegistryError(f"Duplicate benchmark registry ids: {duplicates}")
        artifact_paths = [entry.artifact_path for entry in self.benchmarks]
        duplicate_artifacts = sorted({path for path in artifact_paths if artifact_paths.count(path) > 1})
        if duplicate_artifacts:
            raise BenchmarkRegistryError(f"Duplicate benchmark artifacts in registry: {duplicate_artifacts}")


def load_source_benchmark_registry(path: str | Path) -> SourceBenchmarkRegistry:
    """Load and validate registry metadata without reading artifact files."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return SourceBenchmarkRegistry.from_mapping(payload)


def validate_source_benchmark_registry(path: str | Path, *, repo_root: str | Path | None = None) -> SourceBenchmarkRegistry:
    """Validate registry metadata plus artifact, source-manifest, and source-check files."""

    registry_path = Path(path)
    root = Path(repo_root) if repo_root is not None else registry_path.resolve().parents[1]
    registry = load_source_benchmark_registry(registry_path)
    for entry in registry.benchmarks:
        validate_source_benchmark_entry(entry, repo_root=root)
    return registry


def validate_source_benchmark_entry(entry: SourceBenchmarkEntry, *, repo_root: str | Path) -> None:
    """Validate one registry entry against files on disk."""

    root = Path(repo_root)
    artifact = _load_toml_with_hash(root, entry.artifact_path, entry.artifact_sha256)
    _validate_artifact_payload(entry, artifact)
    for manifest_path in entry.source_manifests:
        expected_sha = entry.source_manifest_sha256[manifest_path]
        _assert_file_hash(root / manifest_path, expected_sha)
    for source_check_path in entry.source_checks:
        expected_sha = entry.source_check_sha256[source_check_path]
        source_check = _load_json_with_hash(root, source_check_path, expected_sha)
        _validate_source_check_payload(entry, source_check, repo_root=root)


def discover_source_backed_benchmark_artifacts(repo_root: str | Path) -> tuple[str, ...]:
    """Find source-backed benchmark artifacts that must be present in the registry."""

    root = Path(repo_root)
    candidates = [
        *sorted((root / "validation" / "real_meta").glob("*_benchmark.toml")),
        *sorted((root / "validation" / "survival").glob("*_benchmark.toml")),
        *sorted((root / "validation" / "networks").glob("*_benchmark.toml")),
    ]
    return tuple(path.relative_to(root).as_posix() for path in candidates)


def assert_registry_covers_source_backed_artifacts(
    registry: SourceBenchmarkRegistry,
    *,
    repo_root: str | Path,
) -> None:
    """Fail if a source-backed local benchmark artifact is not registered."""

    discovered = set(discover_source_backed_benchmark_artifacts(repo_root))
    registered = {entry.artifact_path for entry in registry.benchmarks}
    missing = sorted(discovered - registered)
    extra = sorted(registered - discovered)
    if missing:
        raise BenchmarkRegistryError(f"Unregistered source-backed benchmark artifacts: {missing}")
    if extra:
        raise BenchmarkRegistryError(f"Registry points at missing source-backed benchmark artifacts: {extra}")


def _validate_artifact_payload(entry: SourceBenchmarkEntry, artifact: dict[str, Any]) -> None:
    expected = {
        "schema_version": entry.artifact_schema_version,
        "benchmark_id": entry.id,
        "status": entry.artifact_status,
        "certification_effect": entry.certification_effect,
        "source_policy": entry.source_policy,
    }
    expected["evidence_mode"] = entry.evidence_mode
    for key, value in expected.items():
        if str(artifact.get(key, "")) != value:
            raise BenchmarkRegistryError(
                f"{entry.id}: artifact {key} mismatch: expected {value!r}, got {artifact.get(key)!r}."
            )
    if int(artifact.get("n_studies", -1)) != entry.n_studies:
        raise BenchmarkRegistryError(f"{entry.id}: artifact n_studies does not match registry.")
    limitations = _flatten_text(artifact.get("limitations", []))
    for term in entry.required_limitations:
        if term.lower() not in limitations:
            raise BenchmarkRegistryError(f"{entry.id}: artifact limitations missing required term {term!r}.")


def _validate_source_check_payload(
    entry: SourceBenchmarkEntry,
    source_check: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    if str(source_check.get("benchmark_id", "")) != entry.id:
        raise BenchmarkRegistryError(f"{entry.id}: source-check benchmark_id mismatch.")
    if str(source_check.get("status", "")) != "verified":
        raise BenchmarkRegistryError(f"{entry.id}: source-check status must be verified.")
    if str(source_check.get("certification_effect", "")) != "none":
        raise BenchmarkRegistryError(f"{entry.id}: source-check certification_effect must be none.")
    schema_version = str(source_check.get("schema_version", ""))
    if schema_version not in SUPPORTED_SOURCE_CHECK_SCHEMA_VERSIONS:
        raise BenchmarkRegistryError(f"{entry.id}: unsupported source-check schema_version {schema_version!r}.")
    try:
        if schema_version == SOURCE_VERIFICATION_SCHEMA_VERSION:
            report = SourceVerificationReport.from_mapping(source_check)
            _validate_source_verification_reference(entry, report)
            _validate_identity_bundle_when_applicable(entry, report, repo_root=repo_root)
        elif schema_version == EVENT_COUNT_VERIFICATION_SCHEMA_VERSION:
            report = EventCountVerificationReport.from_mapping(source_check)
            _validate_event_count_reference(entry, report)
        elif schema_version == SURVIVAL_HR_VERIFICATION_SCHEMA_VERSION:
            report = SurvivalHRVerificationReport.from_mapping(source_check)
            _validate_survival_hr_reference(entry, report, repo_root=repo_root)
        elif schema_version == CTGOV_HR_NETWORK_VERIFICATION_SCHEMA_VERSION:
            report = CTGovHRNetworkVerificationReport.from_mapping(source_check)
            _validate_ctgov_hr_network_reference(entry, report, repo_root=repo_root)
    except ValidationError as exc:
        raise BenchmarkRegistryError(f"{entry.id}: specialized source-check validation failed: {exc}") from exc


def _validate_source_verification_reference(
    entry: SourceBenchmarkEntry,
    report: SourceVerificationReport,
) -> None:
    _assert_registered_source_manifest_sha(
        entry,
        report.source_manifest,
        report.source_manifest_sha256,
        label="source verification",
    )


def _validate_event_count_reference(
    entry: SourceBenchmarkEntry,
    report: EventCountVerificationReport,
) -> None:
    _assert_registered_source_manifest_sha(
        entry,
        report.source_manifest,
        report.source_manifest_sha256,
        label="event-count verification source manifest",
    )
    _assert_registered_source_manifest_sha(
        entry,
        report.dataset,
        report.dataset_sha256,
        label="event-count verification dataset",
    )


def _validate_identity_bundle_when_applicable(
    entry: SourceBenchmarkEntry,
    report: SourceVerificationReport,
    *,
    repo_root: Path,
) -> None:
    if entry.evidence_mode != "reported_hr_pubmed_abstract":
        return
    manifest = load_survival_hr_manifest(repo_root / report.source_manifest)
    validate_survival_hr_identity_bundle(manifest, report)


def _validate_survival_hr_reference(
    entry: SourceBenchmarkEntry,
    report: SurvivalHRVerificationReport,
    *,
    repo_root: Path,
) -> None:
    _assert_registered_source_manifest_sha(
        entry,
        report.manifest,
        report.manifest_sha256,
        label="survival HR verification manifest",
    )
    manifest = load_survival_hr_manifest(repo_root / report.manifest)
    validate_survival_hr_source_bundle(manifest, report)


def _validate_ctgov_hr_network_reference(
    entry: SourceBenchmarkEntry,
    report: CTGovHRNetworkVerificationReport,
    *,
    repo_root: Path,
) -> None:
    _assert_registered_source_manifest_sha(
        entry,
        report.manifest,
        report.manifest_sha256,
        label="CT.gov HR verification manifest",
    )
    manifest = load_ctgov_hr_network_manifest(repo_root / report.manifest)
    validate_ctgov_hr_network_source_bundle(manifest, report)


def _assert_registered_source_manifest_sha(
    entry: SourceBenchmarkEntry,
    rel_path: str,
    observed_sha: str,
    *,
    label: str,
) -> None:
    expected_sha = entry.source_manifest_sha256.get(rel_path)
    if expected_sha is None:
        raise BenchmarkRegistryError(f"{entry.id}: {label} references unregistered source artifact {rel_path!r}.")
    if observed_sha != expected_sha:
        raise BenchmarkRegistryError(
            f"{entry.id}: {label} SHA-256 mismatch for {rel_path}: expected {expected_sha}, got {observed_sha}."
        )


def _load_toml_with_hash(root: Path, rel_path: str, expected_sha: str) -> dict[str, Any]:
    path = root / rel_path
    _assert_file_hash(path, expected_sha)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json_with_hash(root: Path, rel_path: str, expected_sha: str) -> dict[str, Any]:
    path = root / rel_path
    _assert_file_hash(path, expected_sha)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_file_hash(path: Path, expected_sha: str) -> None:
    if not path.is_file():
        raise BenchmarkRegistryError(f"registered file does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise BenchmarkRegistryError(f"{path}: SHA-256 mismatch: expected {expected_sha}, got {observed}.")


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in string.hexdigits for char in value)


def _flatten_text(value: Any) -> str:
    if isinstance(value, list | tuple):
        return " ".join(str(item).lower() for item in value)
    return str(value).lower()
