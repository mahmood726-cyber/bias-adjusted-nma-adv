"""Portfolio reuse registry and local scan helpers.

This module records local repositories as implementation candidates only. It
does not import their code and does not turn their outputs into evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import subprocess
import tomllib
from typing import Any, Iterable


PORTFOLIO_REUSE_REGISTRY_SCHEMA_VERSION = "portfolio_reuse_sources/v1"
PORTFOLIO_REUSE_SCAN_SCHEMA_VERSION = "portfolio_reuse_scan/v1"

ALLOWED_SOURCE_POLICY = "local_reuse_registry_only_no_clinical_evidence"
ALLOWED_PRIORITIES = {"high", "medium", "low"}
REQUIRED_REVIEW_ROUNDS = {
    "source_boundary_review",
    "statistical_methods_review",
    "implementation_contract_review",
    "claims_governance_review",
}


class PortfolioReuseError(ValueError):
    """Raised when the portfolio reuse registry or scan is malformed."""


@dataclass(frozen=True)
class PortfolioReuseSource:
    """One local repository that may contribute reusable implementation ideas."""

    id: str
    repo_name: str
    priority: str
    domain: str
    required_assets: tuple[str, ...]
    reusable_assets: tuple[str, ...]
    import_boundary: str
    next_gate: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PortfolioReuseSource":
        missing = _missing_keys(
            raw,
            {
                "id",
                "repo_name",
                "priority",
                "domain",
                "required_assets",
                "reusable_assets",
                "import_boundary",
                "next_gate",
            },
        )
        if missing:
            raise PortfolioReuseError(f"portfolio source missing required keys: {missing}")
        source = cls(
            id=str(raw["id"]),
            repo_name=str(raw["repo_name"]),
            priority=str(raw["priority"]),
            domain=str(raw["domain"]),
            required_assets=tuple(str(item) for item in raw["required_assets"]),
            reusable_assets=tuple(str(item) for item in raw["reusable_assets"]),
            import_boundary=str(raw["import_boundary"]),
            next_gate=str(raw["next_gate"]),
        )
        source.validate()
        return source

    def validate(self) -> None:
        if not self.id.strip():
            raise PortfolioReuseError("portfolio source id must not be empty.")
        if self.priority not in ALLOWED_PRIORITIES:
            raise PortfolioReuseError(f"{self.id}: unsupported priority {self.priority!r}.")
        _assert_relative_name(self.repo_name, f"{self.id}: repo_name")
        if not self.domain.strip():
            raise PortfolioReuseError(f"{self.id}: domain must not be empty.")
        if not self.required_assets:
            raise PortfolioReuseError(f"{self.id}: required_assets must not be empty.")
        if not self.reusable_assets:
            raise PortfolioReuseError(f"{self.id}: reusable_assets must not be empty.")
        for asset in self.required_assets:
            _assert_relative_asset(asset, f"{self.id}: required asset")
        boundary = self.import_boundary.lower()
        boundary_terms = ("certif", "revalid", "valid", "verify", "evidence", "reference", "source")
        if not any(term in boundary for term in boundary_terms):
            raise PortfolioReuseError(
                f"{self.id}: import_boundary must state validation, evidence, or certification limits."
            )
        if not self.next_gate.strip():
            raise PortfolioReuseError(f"{self.id}: next_gate must not be empty.")


@dataclass(frozen=True)
class PortfolioReuseRegistry:
    """Machine-readable registry of local portfolio reuse candidates."""

    checked_at: str
    purpose: str
    source_policy: str
    certification_effect: str
    claim_limit: str
    required_review_rounds: tuple[str, ...]
    sources: tuple[PortfolioReuseSource, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PortfolioReuseRegistry":
        missing = _missing_keys(
            raw,
            {
                "schema_version",
                "checked_at",
                "purpose",
                "source_policy",
                "certification_effect",
                "claim_limit",
                "required_review_rounds",
                "sources",
            },
        )
        if missing:
            raise PortfolioReuseError(f"portfolio reuse registry missing required keys: {missing}")
        if raw["schema_version"] != PORTFOLIO_REUSE_REGISTRY_SCHEMA_VERSION:
            raise PortfolioReuseError(
                f"schema_version must be {PORTFOLIO_REUSE_REGISTRY_SCHEMA_VERSION}."
            )
        registry = cls(
            checked_at=str(raw["checked_at"]),
            purpose=str(raw["purpose"]),
            source_policy=str(raw["source_policy"]),
            certification_effect=str(raw["certification_effect"]),
            claim_limit=str(raw["claim_limit"]),
            required_review_rounds=tuple(str(item) for item in raw["required_review_rounds"]),
            sources=tuple(PortfolioReuseSource.from_mapping(item) for item in raw["sources"]),
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        if self.source_policy != ALLOWED_SOURCE_POLICY:
            raise PortfolioReuseError("portfolio reuse source_policy drifted.")
        if self.certification_effect != "none":
            raise PortfolioReuseError("portfolio reuse registry cannot certify methods.")
        review_rounds = set(self.required_review_rounds)
        missing_reviews = sorted(REQUIRED_REVIEW_ROUNDS - review_rounds)
        if missing_reviews:
            raise PortfolioReuseError(f"portfolio reuse registry missing review rounds: {missing_reviews}")
        if "clinical" not in self.claim_limit.lower() or "superiority" not in self.claim_limit.lower():
            raise PortfolioReuseError("claim_limit must state clinical and superiority claim boundaries.")
        if not self.sources:
            raise PortfolioReuseError("portfolio reuse registry must define at least one source.")
        _assert_unique("portfolio source ids", [source.id for source in self.sources])


def load_portfolio_reuse_registry(path: str | Path) -> PortfolioReuseRegistry:
    """Load and validate a portfolio reuse registry."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return PortfolioReuseRegistry.from_mapping(payload)


def summarize_portfolio_reuse_registry(registry: PortfolioReuseRegistry) -> dict[str, Any]:
    """Return a compact, non-certifying registry summary."""

    return {
        "checked_at": registry.checked_at,
        "n_sources": len(registry.sources),
        "priority_counts": _counts(source.priority for source in registry.sources),
        "domain_counts": _counts(source.domain for source in registry.sources),
        "required_review_rounds": list(registry.required_review_rounds),
        "certification_effect": registry.certification_effect,
    }


def build_portfolio_reuse_scan_report(
    *,
    registry_path: str | Path,
    roots: Iterable[str | Path],
    checked_at: str,
) -> dict[str, Any]:
    """Scan local roots for registered portfolio sources.

    The report is intentionally non-certifying. It only tells whether candidate
    repositories and required relative assets are present in the local workspace.
    """

    registry = load_portfolio_reuse_registry(registry_path)
    normalized_roots = tuple(_normalize_existing_roots(roots))
    rows = [
        _scan_source(source=source, roots=normalized_roots)
        for source in registry.sources
    ]
    status_counts = _counts(row["status"] for row in rows)
    priority_counts = _counts(row["priority"] for row in rows)

    return {
        "schema_version": PORTFOLIO_REUSE_SCAN_SCHEMA_VERSION,
        "checked_at": checked_at,
        "registry": str(Path(registry_path)),
        "roots_checked": [str(root) for root in normalized_roots],
        "source_policy": registry.source_policy,
        "certification_effect": registry.certification_effect,
        "claim_limit": registry.claim_limit,
        "required_review_rounds": list(registry.required_review_rounds),
        "summary": {
            "n_sources": len(rows),
            "status_counts": status_counts,
            "priority_counts": priority_counts,
        },
        "sources": rows,
        "limitations": [
            "Local portfolio repositories are implementation candidates only.",
            "Existing outputs from source repositories are not accepted as clinical evidence.",
            "Every imported asset requires source-policy, statistical, implementation, and claims-governance review.",
            "This scan does not establish tier-one parity, clinical validity, or method superiority.",
        ],
    }


def _scan_source(*, source: PortfolioReuseSource, roots: tuple[Path, ...]) -> dict[str, Any]:
    repo_path = _find_repo_path(source.repo_name, roots)
    if repo_path is None:
        return {
            "id": source.id,
            "repo_name": source.repo_name,
            "priority": source.priority,
            "domain": source.domain,
            "status": "missing_repo",
            "path": None,
            "worktree_state": "not_checked",
            "present_assets": [],
            "missing_assets": list(source.required_assets),
            "import_boundary": source.import_boundary,
            "next_gate": source.next_gate,
        }

    present_assets = []
    missing_assets = []
    for asset in source.required_assets:
        if (repo_path / Path(asset)).is_file():
            present_assets.append(asset)
        else:
            missing_assets.append(asset)

    worktree_state = _git_worktree_state(repo_path)
    if missing_assets:
        status = "missing_required_assets"
    elif worktree_state == "clean":
        status = "ready_for_porting_review"
    elif worktree_state == "dirty":
        status = "review_only_dirty_worktree"
    elif worktree_state == "not_git":
        status = "review_only_not_git"
    else:
        status = "review_only_git_unknown"

    return {
        "id": source.id,
        "repo_name": source.repo_name,
        "priority": source.priority,
        "domain": source.domain,
        "status": status,
        "path": str(repo_path),
        "worktree_state": worktree_state,
        "present_assets": present_assets,
        "missing_assets": missing_assets,
        "import_boundary": source.import_boundary,
        "next_gate": source.next_gate,
    }


def _find_repo_path(repo_name: str, roots: tuple[Path, ...]) -> Path | None:
    wanted = repo_name.lower()
    for root in roots:
        direct = root / repo_name
        if direct.is_dir():
            return direct.resolve()
        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.lower() == wanted:
                    return child.resolve()
        except OSError:
            continue
    return None


def _git_worktree_state(repo_path: Path) -> str:
    if not (repo_path / ".git").exists():
        return "not_git"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--short", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return "dirty" if completed.stdout.strip() else "clean"


def _normalize_existing_roots(roots: Iterable[str | Path]) -> list[Path]:
    normalized = []
    seen = set()
    for root in roots:
        path = Path(root).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(resolved)
    return normalized


def _assert_relative_name(value: str, label: str) -> None:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        not value.strip()
        or path.is_absolute()
        or windows_path.drive
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise PortfolioReuseError(f"{label} must be a plain relative repository name.")


def _assert_relative_asset(value: str, label: str) -> None:
    path = Path(value)
    windows_path = PureWindowsPath(value)
    parts = Path(value).parts
    if (
        not value.strip()
        or path.is_absolute()
        or windows_path.drive
        or "\\" in value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise PortfolioReuseError(f"{label} must be a relative POSIX-style file path.")


def _missing_keys(raw: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required - set(raw))


def _assert_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise PortfolioReuseError(f"{label} must be unique; duplicates: {duplicates}")


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
