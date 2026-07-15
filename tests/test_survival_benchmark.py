import copy
from pathlib import Path

import pytest

from bias_nma_adv.data import ValidationError
from bias_nma_adv.real_meta import sha256_file
from bias_nma_adv.survival_benchmark import (
    SurvivalHRManifest,
    SurvivalHRVerificationReport,
    load_survival_hr_manifest,
    load_survival_hr_verification_report,
    validate_survival_hr_identity_bundle,
    validate_survival_hr_source_bundle,
)
from bias_nma_adv.source_verification import (
    SourceVerificationReport,
    load_source_verification_report,
    summarize_source_verification,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "survival" / "sglt2_hf_reported_hrs.toml"
REPORT = ROOT / "validation" / "source_checks" / "sglt2_hf_reported_hr_tokens.json"
IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "sglt2_hf_reported_hr_source_check.json"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_pubmed_survival_hrs.py"
IDENTITY_VERIFY_SCRIPT = ROOT / "scripts" / "verify_survival_sources.py"


def test_survival_hr_manifest_is_source_bounded_and_pubmed_only():
    manifest = load_survival_hr_manifest(MANIFEST)

    assert manifest.benchmark_id == "sglt2_hf_reported_hr"
    assert manifest.evidence_mode == "reported_hr_pubmed_abstract"
    assert manifest.status == "candidate_source_verified"
    assert manifest.certification_effect == "none"
    assert manifest.manifest_sha256 == sha256_file(MANIFEST)
    assert len(manifest.studies) == 4
    assert {study.study_id for study in manifest.studies} == {
        "DAPA-HF",
        "EMPEROR-Reduced",
        "DELIVER",
        "EMPEROR-Preserved",
    }
    for study in manifest.studies:
        assert study.source_type == "pubmed_abstract"
        assert study.km_reconstruction_status == "not_digitized"
        assert study.reuse_origin == "wasserstein_method_pattern_only"
        assert study.effect_direction == "active_vs_control"
        assert study.source_url.startswith("https://pubmed.ncbi.nlm.nih.gov/")


def test_survival_hr_manifest_rejects_uncertified_km_import(tmp_path):
    bad = tmp_path / "bad_survival_manifest.toml"
    bad.write_text(
        MANIFEST.read_text(encoding="utf-8").replace(
            'km_reconstruction_status = "not_digitized"',
            'km_reconstruction_status = "complete"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="KM reconstruction cannot be marked complete"):
        load_survival_hr_manifest(bad)


def test_survival_hr_manifest_rejects_ci_that_excludes_hr():
    raw = copy.deepcopy(load_survival_hr_manifest(MANIFEST).__dict__)
    raw["schema_version"] = "survival_hr_manifest/v1"
    raw["studies"] = [copy.deepcopy(study.__dict__) for study in raw["studies"]]
    raw["studies"][0]["ci_lower"] = "0.80"

    with pytest.raises(ValidationError, match="reported HR is not contained"):
        SurvivalHRManifest.from_mapping(raw)


def test_survival_hr_manifest_rejects_pubmed_url_identifier_mismatch(tmp_path):
    bad = tmp_path / "bad_survival_manifest.toml"
    bad.write_text(
        MANIFEST.read_text(encoding="utf-8").replace(
            "https://pubmed.ncbi.nlm.nih.gov/31535829/",
            "https://pubmed.ncbi.nlm.nih.gov/32865377/",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="PubMed source URL"):
        load_survival_hr_manifest(bad)


def test_survival_hr_manifest_rejects_scalar_source_terms():
    raw = copy.deepcopy(load_survival_hr_manifest(MANIFEST).__dict__)
    raw["schema_version"] = "survival_hr_manifest/v1"
    raw["studies"] = [copy.deepcopy(study.__dict__) for study in raw["studies"]]
    raw["studies"][0]["source_terms"] = "dapagliflozin"

    with pytest.raises(ValidationError, match="source_terms must be a list"):
        SurvivalHRManifest.from_mapping(raw)


def test_survival_hr_verification_snapshot_matches_manifest():
    manifest = load_survival_hr_manifest(MANIFEST)
    report = load_survival_hr_verification_report(REPORT)

    assert report.status == "verified"
    assert report.certification_effect == "none"
    assert report.benchmark_id == manifest.benchmark_id
    assert report.manifest == "validation/survival/sglt2_hf_reported_hrs.toml"
    assert report.manifest_sha256 == sha256_file(MANIFEST)
    assert VERIFY_SCRIPT.is_file()
    assert len(report.records) == len(manifest.studies)

    expected = {
        (study.study_id, study.pmid, study.outcome_id, study.reported_hr, study.ci_lower, study.ci_upper)
        for study in manifest.studies
    }
    observed = {
        (record.study_id, record.pmid, record.outcome_id, record.reported_hr, record.ci_lower, record.ci_upper)
        for record in report.records
    }
    assert observed == expected

    for record in report.records:
        assert record.evidence_scope == "pubmed_abstract_reported_hr_tokens"
        assert len(record.abstract_sha256) == 64
        assert record.hr_token_found is True
        assert record.ci_lower_token_found is True
        assert record.ci_upper_token_found is True
        assert record.hazard_ratio_anchor_found is True
        assert record.confidence_interval_anchor_found is True
        assert record.tokens_near_hazard_ratio_anchor is True
        assert record.source_terms_near_hazard_ratio_anchor is True
        assert record.verified is True


def test_survival_hr_identity_snapshot_matches_manifest():
    manifest = load_survival_hr_manifest(MANIFEST)
    report = load_source_verification_report(IDENTITY_REPORT)

    assert report.status == "verified"
    assert report.certification_effect == "none"
    assert report.benchmark_id == manifest.benchmark_id
    assert report.source_manifest == "validation/survival/sglt2_hf_reported_hrs.toml"
    assert report.source_manifest_sha256 == sha256_file(MANIFEST)
    assert summarize_source_verification(report) == {
        "clinicaltrials_gov": 4,
        "pubmed_abstract": 4,
    }
    assert IDENTITY_VERIFY_SCRIPT.is_file()

    expected = {
        (study.study_id, "clinicaltrials_gov", study.nct_id)
        for study in manifest.studies
    } | {
        (study.study_id, "pubmed_abstract", study.pmid)
        for study in manifest.studies
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
        if record.source_type == "pubmed_abstract":
            assert record.details["pmid"] == record.identifier
            assert record.details["abstract_present"] is True
            assert len(record.details["abstract_sha256"]) == 64

    bundle = validate_survival_hr_identity_bundle(manifest, report)
    assert bundle["source_counts"] == {"clinicaltrials_gov": 4, "pubmed_abstract": 4}


def test_survival_hr_report_rejects_unverified_record_marked_verified():
    raw = copy.deepcopy(load_survival_hr_verification_report(REPORT).__dict__)
    raw["schema_version"] = "survival_hr_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["tokens_near_hazard_ratio_anchor"] = False

    with pytest.raises(ValidationError, match="verified HR record is missing abstract token evidence"):
        SurvivalHRVerificationReport.from_mapping(raw)


def test_survival_hr_report_rejects_scalar_source_terms():
    raw = copy.deepcopy(load_survival_hr_verification_report(REPORT).__dict__)
    raw["schema_version"] = "survival_hr_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["source_terms"] = "dapagliflozin"

    with pytest.raises(ValidationError, match="source_terms must be a list"):
        SurvivalHRVerificationReport.from_mapping(raw)


def test_survival_hr_manifest_rejects_certification_effect():
    raw = copy.deepcopy(load_survival_hr_manifest(MANIFEST).__dict__)
    raw["schema_version"] = "survival_hr_manifest/v1"
    raw["certification_effect"] = "reference_matched"
    raw["studies"] = [copy.deepcopy(study.__dict__) for study in raw["studies"]]

    with pytest.raises(ValidationError, match="cannot certify model performance"):
        SurvivalHRManifest.from_mapping(raw)


def test_survival_hr_report_rejects_certification_effect():
    raw = copy.deepcopy(load_survival_hr_verification_report(REPORT).__dict__)
    raw["schema_version"] = "survival_hr_verification/v1"
    raw["certification_effect"] = "reference_matched"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]

    with pytest.raises(ValidationError, match="cannot certify model performance"):
        SurvivalHRVerificationReport.from_mapping(raw)


def test_survival_hr_source_bundle_rejects_manifest_hash_drift():
    manifest = load_survival_hr_manifest(MANIFEST)
    raw = copy.deepcopy(load_survival_hr_verification_report(REPORT).__dict__)
    raw["schema_version"] = "survival_hr_verification/v1"
    raw["manifest_sha256"] = "a" * 64
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    report = SurvivalHRVerificationReport.from_mapping(raw)

    with pytest.raises(ValidationError, match="manifest_sha256"):
        validate_survival_hr_source_bundle(manifest, report)


def test_survival_hr_identity_bundle_rejects_record_drift():
    manifest = load_survival_hr_manifest(MANIFEST)
    raw = copy.deepcopy(load_source_verification_report(IDENTITY_REPORT).__dict__)
    raw["schema_version"] = "source_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["identifier"] = "NCT00000000"
    report = SourceVerificationReport.from_mapping(raw)

    with pytest.raises(ValidationError, match="identity records do not match"):
        validate_survival_hr_identity_bundle(manifest, report)
