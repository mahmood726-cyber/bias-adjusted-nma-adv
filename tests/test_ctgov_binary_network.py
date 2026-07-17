import copy
import json
from pathlib import Path

import pytest

from bias_nma_adv.ctgov_binary_network import (
    CTGovBinaryNetworkVerificationReport,
    contrast_rows_from_manifest,
    load_ctgov_binary_network_manifest,
    load_ctgov_binary_network_verification_report,
    run_ctgov_binary_network_benchmark,
    validate_ctgov_binary_network_source_bundle,
)
from bias_nma_adv.data import ValidationError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "networks" / "psoriasis_pasi90_ctgov_binary_network.toml"
SOURCE_CHECK = ROOT / "validation" / "source_checks" / "psoriasis_pasi90_ctgov_binary_network_check.json"
ARTIFACT = ROOT / "validation" / "networks" / "psoriasis_pasi90_ctgov_binary_network_benchmark.toml"


def test_ctgov_binary_network_manifest_is_closed_loop_and_source_bounded():
    manifest = load_ctgov_binary_network_manifest(MANIFEST)

    assert manifest.benchmark_id == "psoriasis_pasi90_ctgov_binary_network"
    assert manifest.certification_effect == "none"
    assert manifest.reference_treatment == "placebo"
    assert {study.nct_id for study in manifest.studies} == {"NCT01358578", "NCT01646177"}
    assert {study.pmid for study in manifest.studies} == {"25007392", "26072109"}
    assert sum(len(study.arms) for study in manifest.studies) == 8

    rows = contrast_rows_from_manifest(manifest)
    assert len(rows) == 12
    assert {row.study for row in rows} == {"FIXTURE", "UNCOVER-3"}
    assert all(row.se > 0.0 for row in rows)


def test_ctgov_binary_network_source_bundle_verifies_ctgov_counts_and_pubmed_identity():
    manifest = load_ctgov_binary_network_manifest(MANIFEST)
    report = load_ctgov_binary_network_verification_report(SOURCE_CHECK)

    validate_ctgov_binary_network_source_bundle(manifest, report)
    assert report.status == "verified"
    assert {record.source_type for record in report.records} == {
        "clinicaltrials_gov",
        "pubmed_abstract",
    }


def test_ctgov_binary_network_artifact_regenerates_expected_summary():
    artifact = run_ctgov_binary_network_benchmark(
        MANIFEST,
        verification_report_path=SOURCE_CHECK,
    )

    assert artifact["schema_version"] == "ctgov_binary_network_benchmark/v1"
    assert artifact["status"] == "local_pass"
    assert artifact["certification_effect"] == "none"
    assert artifact["n_studies"] == 2
    assert artifact["n_arms"] == 8
    assert artifact["n_contrast_rows"] == 12
    assert artifact["closed_loop_cycle_rank"] == 6
    assert artifact["candidate"]["random_effect"]["tau2"] == pytest.approx(0.0, abs=1e-12)
    assert artifact["candidate"]["random_effect"]["q"] == pytest.approx(0.4746923481840964)
    assert artifact["candidate"]["node_splitting"]["n_estimable"] == 0
    assert "not broad inconsistency performance" in " ".join(artifact["limitations"])


def test_ctgov_binary_network_source_report_rejects_unverified_record():
    raw = json.loads(SOURCE_CHECK.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["records"][0]["verified"] = False

    with pytest.raises(ValidationError, match="not verified"):
        CTGovBinaryNetworkVerificationReport.from_mapping(raw)


def test_ctgov_binary_network_artifact_file_is_noncertifying():
    text = ARTIFACT.read_text(encoding="utf-8")

    assert 'certification_effect = "none"' in text
    assert "does not certify model performance" in text
    assert "clinical guidance" not in text.lower()
