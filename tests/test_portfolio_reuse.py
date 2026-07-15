import copy
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.portfolio_reuse import (
    PortfolioReuseError,
    PortfolioReuseRegistry,
    build_portfolio_reuse_scan_report,
    load_portfolio_reuse_registry,
    summarize_portfolio_reuse_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "validation" / "portfolio_reuse_sources.toml"


def test_portfolio_reuse_registry_is_non_certifying_and_path_safe():
    registry = load_portfolio_reuse_registry(REGISTRY)

    assert registry.certification_effect == "none"
    assert registry.source_policy == "local_reuse_registry_only_no_clinical_evidence"
    assert {source.repo_name for source in registry.sources} >= {
        "wasserstein",
        "advanced-nma-pooling",
        "aact-kit",
        "rct-extractor-v2",
    }

    for source in registry.sources:
        assert not Path(source.repo_name).is_absolute()
        assert "/" not in source.repo_name
        assert "\\" not in source.repo_name
        for asset in source.required_assets:
            assert not Path(asset).is_absolute()
            assert "\\" not in asset
            assert ".." not in Path(asset).parts

    summary = summarize_portfolio_reuse_registry(registry)
    assert summary["n_sources"] == 8
    assert summary["priority_counts"] == {"high": 4, "medium": 4}
    assert summary["certification_effect"] == "none"


def test_portfolio_reuse_scan_reports_present_missing_and_review_status(tmp_path):
    root = tmp_path / "portfolio"
    root.mkdir()

    repo = root / "wasserstein"
    (repo / "benchmark" / "tests").mkdir(parents=True)
    for asset in [
        "README.md",
        "benchmark/README.md",
        "faithful_guyot.py",
        "improved_hr_estimation.py",
        "improved_guyot_algorithm.py",
        "benchmark/km_metrics.py",
        "benchmark/tests/test_km_metrics.py",
    ]:
        path = repo / asset
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder for scan test\n", encoding="utf-8")

    report = build_portfolio_reuse_scan_report(
        registry_path=REGISTRY,
        roots=[root],
        checked_at="2026-07-15T00:00:00Z",
    )

    assert report["certification_effect"] == "none"
    assert report["summary"]["n_sources"] == 8
    assert report["summary"]["status_counts"]["review_only_not_git"] == 1
    assert report["summary"]["status_counts"]["missing_repo"] == 7

    wasserstein = next(row for row in report["sources"] if row["repo_name"] == "wasserstein")
    assert wasserstein["status"] == "review_only_not_git"
    assert wasserstein["worktree_state"] == "not_git"
    assert wasserstein["missing_assets"] == []
    assert len(wasserstein["present_assets"]) == 7


def test_portfolio_reuse_scan_reports_missing_required_assets(tmp_path):
    root = tmp_path / "portfolio"
    (root / "aact-kit").mkdir(parents=True)

    report = build_portfolio_reuse_scan_report(
        registry_path=REGISTRY,
        roots=[root],
        checked_at="2026-07-15T00:00:00Z",
    )

    aact = next(row for row in report["sources"] if row["repo_name"] == "aact-kit")
    assert aact["status"] == "missing_required_assets"
    assert "src/aact_kit/schema.py" in aact["missing_assets"]


def test_portfolio_reuse_registry_rejects_certification_effect():
    payload = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["certification_effect"] = "production_certified"

    with pytest.raises(PortfolioReuseError, match="cannot certify"):
        PortfolioReuseRegistry.from_mapping(payload)


def test_portfolio_reuse_registry_rejects_absolute_repo_or_asset_paths():
    payload = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))

    bad_repo = copy.deepcopy(payload)
    bad_repo["sources"][0]["repo_name"] = "C:/Projects/wasserstein"
    with pytest.raises(PortfolioReuseError, match="plain relative repository name"):
        PortfolioReuseRegistry.from_mapping(bad_repo)

    bad_asset = copy.deepcopy(payload)
    bad_asset["sources"][0]["required_assets"][0] = "C:/tmp/README.md"
    with pytest.raises(PortfolioReuseError, match="relative POSIX-style"):
        PortfolioReuseRegistry.from_mapping(bad_asset)


def test_scan_portfolio_reuse_cli_writes_json(tmp_path):
    output = tmp_path / "portfolio_scan.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/scan_portfolio_reuse.py",
            "--root",
            str(tmp_path),
            "--checked-at",
            "2026-07-15T00:00:00Z",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "portfolio reuse scan written" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "portfolio_reuse_scan/v1"
    assert payload["certification_effect"] == "none"
    assert payload["summary"]["status_counts"] == {"missing_repo": 8}
