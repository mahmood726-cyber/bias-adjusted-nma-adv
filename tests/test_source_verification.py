import copy
from pathlib import Path
import tomllib

import pytest

from bias_nma_adv.data import ValidationError
from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.source_verification import (
    SourceVerificationReport,
    load_source_verification_report,
    summarize_source_verification,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "validation" / "source_checks" / "sglt2_hf_primary_source_check.json"
SGLT2_SOURCES = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_sources.toml"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_real_meta_sources.py"


def test_source_verification_snapshot_matches_manifest_sources():
    report = load_source_verification_report(SOURCE_REPORT)
    manifest = tomllib.loads(SGLT2_SOURCES.read_text(encoding="utf-8"))

    assert report.status == "verified"
    assert report.benchmark_id == manifest["benchmark_id"]
    assert report.source_manifest == "validation/real_meta/sglt2_hf_primary_sources.toml"
    assert report.source_manifest_sha256 == sha256_file(SGLT2_SOURCES)
    assert summarize_source_verification(report) == {
        "clinicaltrials_gov": 4,
        "pubmed_abstract": 4,
    }
    assert VERIFY_SCRIPT.is_file()

    expected = {
        (study["study_id"], source["source_type"], source["identifier"])
        for study in manifest["studies"]
        for source in study["sources"]
    }
    observed = {
        (record.study_id, record.source_type, record.identifier)
        for record in report.records
    }
    assert observed == expected

    for record in report.records:
        assert record.http_status == 200
        assert record.identity_verified is True
        assert record.evidence_scope == "identity_and_reachability"
        assert len(record.response_sha256) == 64
        if record.source_type == "clinicaltrials_gov":
            assert record.details["nct_id"] == record.identifier
            assert record.details["overall_status"] == "COMPLETED"
            assert "clinicaltrials.gov/api/v2/studies/" in record.api_url
        if record.source_type == "pubmed_abstract":
            assert record.details["pmid"] == record.identifier
            assert record.details["abstract_present"] is True
            assert record.details["journal"] == "N Engl J Med"
            assert len(record.details["abstract_sha256"]) == 64


def test_source_verification_rejects_failed_identity_in_verified_report():
    raw = copy.deepcopy(load_source_verification_report(SOURCE_REPORT).__dict__)
    raw["schema_version"] = "source_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["identity_verified"] = False

    with pytest.raises(ValidationError, match="source identity was not verified"):
        SourceVerificationReport.from_mapping(raw)


def test_source_verification_rejects_scope_creep():
    raw = copy.deepcopy(load_source_verification_report(SOURCE_REPORT).__dict__)
    raw["schema_version"] = "source_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["evidence_scope"] = "event_count_extraction"

    with pytest.raises(ValidationError, match="unsupported evidence_scope"):
        SourceVerificationReport.from_mapping(raw)
