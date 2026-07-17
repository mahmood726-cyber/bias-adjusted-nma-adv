import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bias_nma_adv.data import ValidationError
from bias_nma_adv.dta_benchmark import (
    DTA_BENCHMARK_SCHEMA_VERSION,
    DTA_MANIFEST_SCHEMA_VERSION,
    DTA_VERIFICATION_SCHEMA_VERSION,
    load_dta_manifest,
    load_dta_verification_report,
    run_dta_benchmark,
    validate_dta_source_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "dta" / "midkine_elisa_cancer_dta.toml"
SOURCE_CHECK = ROOT / "validation" / "source_checks" / "midkine_elisa_cancer_dta_check.json"
BENCHMARK = ROOT / "validation" / "dta" / "midkine_elisa_cancer_dta_benchmark.toml"
CSV = ROOT / "validation" / "dta" / "midkine_elisa_cancer_dta_2x2.csv"
WRITE_SCRIPT = ROOT / "scripts" / "write_dta_benchmark.py"


def test_dta_open_access_manifest_and_source_bundle_are_verified():
    manifest = load_dta_manifest(MANIFEST)
    report = load_dta_verification_report(SOURCE_CHECK)

    assert manifest.benchmark_id == "midkine_elisa_cancer_dta"
    assert manifest.article_doi == "10.1371/journal.pone.0180511"
    assert manifest.table_doi == "10.1371/journal.pone.0180511.t001"
    assert manifest.index_test == "serum Midkine ELISA"
    assert manifest.reference_standard == "cancer diagnosis as classified in source study"
    assert len(manifest.records) == 11
    assert report.table_sha256 == "380281e17104442b1c3560d74286bb79c29eb432027855012291cee903c9edaf"
    assert {record.source_type for record in manifest.records} == {"open_access_paper"}
    assert validate_dta_source_bundle(manifest, report) == {
        "benchmark_id": "midkine_elisa_cancer_dta",
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": "verified",
        "certification_effect": "none",
        "n_studies": 11,
        "source_counts": {"open_access_paper": 11},
    }


def test_dta_benchmark_recomputes_from_source_manifest():
    artifact = run_dta_benchmark(MANIFEST, verification_report_path=SOURCE_CHECK)

    assert artifact["schema_version"] == DTA_BENCHMARK_SCHEMA_VERSION
    assert artifact["benchmark_id"] == "midkine_elisa_cancer_dta"
    assert artifact["status"] == "local_pass"
    assert artifact["certification_effect"] == "none"
    assert artifact["n_studies"] == 11
    assert artifact["source_bundle"]["source_counts"] == {"open_access_paper": 11}
    fit = artifact["candidate"]["bivariate_logitnormal_reml"]
    assert fit["n_studies"] == 11
    assert fit["converged"] is True
    assert fit["pooled_sensitivity"] == pytest.approx(0.7464719686937061)
    assert fit["pooled_specificity"] == pytest.approx(0.8147795241065229)
    assert fit["log_diagnostic_odds_ratio"] == pytest.approx(2.561254348894983)
    assert "does not certify model performance" in artifact["limitations"]


def test_dta_csv_matches_manifest_counts():
    manifest = load_dta_manifest(MANIFEST)
    expected = {record.study_id: record for record in manifest.records}
    with CSV.open("r", encoding="utf-8", newline="") as handle:
        observed = {row["study_id"]: row for row in csv.DictReader(handle)}

    assert set(observed) == set(expected)
    for study_id, record in expected.items():
        row = observed[study_id]
        assert int(row["tp"]) == record.tp
        assert int(row["fp"]) == record.fp
        assert int(row["fn"]) == record.fn
        assert int(row["tn"]) == record.tn
        assert row["source_doi"] == record.source_doi
        assert row["table_doi"] == record.table_doi


def test_write_dta_benchmark_script_regenerates_artifact(tmp_path):
    output = tmp_path / "dta_benchmark.toml"
    subprocess.run(
        [
            sys.executable,
            str(WRITE_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--source-check",
            str(SOURCE_CHECK),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.read_text(encoding="utf-8") == BENCHMARK.read_text(encoding="utf-8")


def test_dta_source_bundle_rejects_count_drift(tmp_path):
    manifest = load_dta_manifest(MANIFEST)
    payload = json.loads(SOURCE_CHECK.read_text(encoding="utf-8"))
    payload["records"][0]["tp"] += 1
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    report = load_dta_verification_report(drifted)

    with pytest.raises(ValidationError, match="counts mismatch"):
        validate_dta_source_bundle(manifest, report)


def test_dta_manifest_schema_constant_is_current():
    assert DTA_MANIFEST_SCHEMA_VERSION == "dta_open_access_manifest/v1"
    assert DTA_VERIFICATION_SCHEMA_VERSION == "dta_source_verification/v1"
