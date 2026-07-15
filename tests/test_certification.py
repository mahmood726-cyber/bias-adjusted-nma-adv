import copy
import hashlib
from pathlib import Path

import pytest

from bias_nma_adv.certification import (
    CertificationError,
    ReferenceRunReport,
    ReferenceTarget,
    assert_no_unsupported_production_claims,
    assert_reference_runs_target_known,
    certification_candidate_artifacts,
    load_reference_run_reports,
    load_reference_targets,
    summarize_reference_run_reports,
    summarize_reference_targets,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "validation" / "reference_targets.toml"
REFERENCE_RUNS_PATH = ROOT / "validation" / "reference_runs"
PAIRWISE_R_ADAPTER = ROOT / "external" / "r" / "pairwise_metafor_meta.R"
PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_reference_adapters.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_reference_targets_registry_is_valid():
    targets = load_reference_targets(TARGETS_PATH)

    assert len(targets) >= 8
    assert {target.id for target in targets} >= {
        "frequentist_nma_netmeta",
        "bayesian_nma_multinma_cmdstan",
        "mlnmr_multinma",
        "dose_response_mbnmadose",
        "cross_design_crossnma",
        "component_nma_netmeta_cnma",
        "certainty_cinema_robmen",
        "pairwise_metafor_meta",
    }

    summary = summarize_reference_targets(targets)
    assert summary == {"planned": len(targets)}
    assert_no_unsupported_production_claims(targets)


def test_reference_target_rejects_certification_without_evidence():
    raw = {
        "id": "bad_claim",
        "domain": "frequentist_nma",
        "module": "src/bias_nma_adv/model.py",
        "reference_method": "netmeta",
        "status": "reference_matched",
        "acceptance_criteria": ["match published examples"],
        "evidence_artifacts": [],
    }

    with pytest.raises(CertificationError, match="requires evidence_artifacts"):
        ReferenceTarget.from_mapping(raw)


def test_production_claim_requires_prerequisite_evidence_markers():
    raw = {
        "id": "premature_production",
        "domain": "frequentist_nma",
        "module": "src/bias_nma_adv/model.py",
        "reference_method": "netmeta",
        "status": "production_certified",
        "acceptance_criteria": ["match published examples"],
        "evidence_artifacts": ["reports/netmeta_reference_matched.json"],
    }
    target = ReferenceTarget.from_mapping(copy.deepcopy(raw))

    with pytest.raises(CertificationError, match="lacks evidence markers"):
        assert_no_unsupported_production_claims([target])

    raw["evidence_artifacts"] = [
        "reports/netmeta_reference_matched.json",
        "reports/netmeta_simulation_validated.json",
        "reports/netmeta_externally_reproduced.json",
    ]
    target = ReferenceTarget.from_mapping(raw)
    assert_no_unsupported_production_claims([target])


def test_reference_run_reports_are_fail_closed_and_targeted():
    targets = load_reference_targets(TARGETS_PATH)
    reports = load_reference_run_reports(REFERENCE_RUNS_PATH)

    assert_reference_runs_target_known(targets, reports)
    assert summarize_reference_run_reports(reports) == {"unavailable": 1}
    report = reports[0]
    assert report.target_id == "pairwise_metafor_meta"
    assert report.adapter_id == "r_metafor_meta_pairwise_preflight"
    assert report.certification_effect == "none"
    assert report.is_certification_evidence_candidate is False
    assert "Rscript" in report.command
    assert "external/r/pairwise_metafor_meta.R" in report.command
    assert report.executable_found is False
    assert "Rscript is not available" in report.skip_reason
    assert PAIRWISE_R_ADAPTER.is_file()
    assert PREFLIGHT_SCRIPT.is_file()
    for artifact, expected_sha in report.input_sha256.items():
        assert sha256_file(ROOT / artifact) == expected_sha
    assert certification_candidate_artifacts(reports) == ()


def test_reference_run_rejects_nonpassed_certification_evidence():
    raw = {
        "schema_version": "reference_run/v1",
        "target_id": "pairwise_metafor_meta",
        "adapter_id": "bad_unavailable_evidence",
        "reference_method": "metafor and meta",
        "status": "unavailable",
        "certification_effect": "evidence_candidate",
        "command": ["Rscript", "--vanilla", "external/r/pairwise_metafor_meta.R"],
        "executable": "Rscript",
        "executable_found": False,
        "package_versions": {},
        "input_artifacts": [],
        "input_sha256": {},
        "output_artifacts": [],
        "output_sha256": {},
        "tolerance": "",
        "skip_reason": "Rscript is not available.",
    }

    with pytest.raises(CertificationError, match="non-passed reference runs"):
        ReferenceRunReport.from_mapping(raw)


def test_passed_reference_run_requires_hashes_versions_and_tolerance():
    raw = {
        "schema_version": "reference_run/v1",
        "target_id": "pairwise_metafor_meta",
        "adapter_id": "bad_passed_report",
        "reference_method": "metafor and meta",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "command": ["Rscript", "--vanilla", "external/r/pairwise_metafor_meta.R"],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {"R": "4.6.0"},
        "input_artifacts": ["validation/real_meta/sglt2_hf_primary_events.csv"],
        "input_sha256": {},
        "output_artifacts": ["validation/reference_runs/pairwise_metafor_meta_output.json"],
        "output_sha256": {},
        "tolerance": "absolute <= 1e-6",
        "skip_reason": "",
    }

    with pytest.raises(CertificationError, match="missing input SHA-256"):
        ReferenceRunReport.from_mapping(raw)

    raw["input_sha256"] = {
        "validation/real_meta/sglt2_hf_primary_events.csv": "a" * 64,
    }
    with pytest.raises(CertificationError, match="missing output SHA-256"):
        ReferenceRunReport.from_mapping(raw)

    raw["output_sha256"] = {
        "validation/reference_runs/pairwise_metafor_meta_output.json": "b" * 64,
    }
    report = ReferenceRunReport.from_mapping(raw)
    assert report.is_certification_evidence_candidate is True
    assert certification_candidate_artifacts([report]) == (
        "validation/reference_runs/pairwise_metafor_meta_output.json",
    )
