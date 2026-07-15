import copy
from pathlib import Path
import tomllib

import pytest

from bias_nma_adv.data import ValidationError
from bias_nma_adv.event_count_verification import (
    EventCountVerificationReport,
    load_event_count_verification_report,
)
from bias_nma_adv.real_meta import load_arm_event_rows, sha256_file


ROOT = Path(__file__).resolve().parents[1]
EVENT_COUNT_REPORT = ROOT / "validation" / "source_checks" / "sglt2_hf_primary_event_counts.json"
SGLT2_EVENTS = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_events.csv"
SGLT2_SOURCES = ROOT / "validation" / "real_meta" / "sglt2_hf_primary_sources.toml"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_pubmed_event_counts.py"


def test_event_count_snapshot_matches_pubmed_abstract_terms_and_counts():
    report = load_event_count_verification_report(EVENT_COUNT_REPORT)
    manifest = tomllib.loads(SGLT2_SOURCES.read_text(encoding="utf-8"))
    rows = load_arm_event_rows(SGLT2_EVENTS)

    assert report.status == "verified"
    assert report.benchmark_id == manifest["benchmark_id"]
    assert report.dataset == "validation/real_meta/sglt2_hf_primary_events.csv"
    assert report.dataset_sha256 == sha256_file(SGLT2_EVENTS)
    assert report.source_manifest == "validation/real_meta/sglt2_hf_primary_sources.toml"
    assert report.source_manifest_sha256 == sha256_file(SGLT2_SOURCES)
    assert report.certification_effect == "none"
    assert VERIFY_SCRIPT.is_file()
    assert len(report.records) == 4

    expected_sources = {
        (
            study["study_id"],
            study["pmid"],
            tuple(study["active_source_terms"]),
            tuple(study["control_source_terms"]),
        )
        for study in manifest["studies"]
    }
    observed_sources = {
        (
            record.study_id,
            record.pmid,
            record.active_source_terms,
            record.control_source_terms,
        )
        for record in report.records
    }
    assert observed_sources == expected_sources

    rows_by_study = {}
    for row in rows:
        rows_by_study.setdefault(row.study_id, {})[row.arm_role] = row

    for record in report.records:
        active = rows_by_study[record.study_id]["active"]
        control = rows_by_study[record.study_id]["control"]
        assert record.evidence_scope == "pubmed_abstract_event_count_tokens"
        assert record.verified is True
        assert record.active_count_token_found is True
        assert record.control_count_token_found is True
        assert record.active_term_near_count is True
        assert record.control_term_near_count is True
        assert len(record.abstract_sha256) == 64
        assert record.outcome_id == active.outcome_id == control.outcome_id
        assert record.outcome_label == active.outcome_label == control.outcome_label
        assert record.active_events == active.events
        assert record.active_n == active.n
        assert record.control_events == control.events
        assert record.control_n == control.n
        assert record.active_count_token == f"{active.events} of {active.n}"
        assert record.control_count_token == f"{control.events} of {control.n}"


def test_event_count_report_rejects_unverified_record_marked_verified():
    raw = copy.deepcopy(load_event_count_verification_report(EVENT_COUNT_REPORT).__dict__)
    raw["schema_version"] = "event_count_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["active_count_token_found"] = False

    with pytest.raises(ValidationError, match="verified record is missing count or treatment evidence"):
        EventCountVerificationReport.from_mapping(raw)


def test_event_count_report_rejects_scope_creep():
    raw = copy.deepcopy(load_event_count_verification_report(EVENT_COUNT_REPORT).__dict__)
    raw["schema_version"] = "event_count_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["evidence_scope"] = "full_text_event_extraction"

    with pytest.raises(ValidationError, match="unsupported evidence_scope"):
        EventCountVerificationReport.from_mapping(raw)


def test_event_count_report_rejects_scalar_source_terms():
    raw = copy.deepcopy(load_event_count_verification_report(EVENT_COUNT_REPORT).__dict__)
    raw["schema_version"] = "event_count_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["active_source_terms"] = "dapagliflozin"

    with pytest.raises(ValidationError, match="source terms must be lists"):
        EventCountVerificationReport.from_mapping(raw)


def test_event_count_report_rejects_count_token_mismatch():
    raw = copy.deepcopy(load_event_count_verification_report(EVENT_COUNT_REPORT).__dict__)
    raw["schema_version"] = "event_count_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["active_count_token"] = "386/2373"

    with pytest.raises(ValidationError, match="active_count_token does not match active counts"):
        EventCountVerificationReport.from_mapping(raw)


def test_event_count_report_rejects_certification_effect():
    raw = copy.deepcopy(load_event_count_verification_report(EVENT_COUNT_REPORT).__dict__)
    raw["schema_version"] = "event_count_verification/v1"
    raw["certification_effect"] = "reference_matched"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]

    with pytest.raises(ValidationError, match="cannot certify model performance"):
        EventCountVerificationReport.from_mapping(raw)
