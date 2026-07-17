import copy
import importlib.util
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
PCSK9_MANIFEST = ROOT / "validation" / "survival" / "pcsk9_mace_reported_hrs.toml"
PCSK9_REPORT = ROOT / "validation" / "source_checks" / "pcsk9_mace_reported_hr_tokens.json"
PCSK9_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "pcsk9_mace_reported_hr_source_check.json"
CKD_MANIFEST = ROOT / "validation" / "survival" / "sglt2_ckd_reported_hrs.toml"
CKD_REPORT = ROOT / "validation" / "source_checks" / "sglt2_ckd_reported_hr_tokens.json"
CKD_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "sglt2_ckd_reported_hr_source_check.json"
GLP1_MANIFEST = ROOT / "validation" / "survival" / "glp1_mace_reported_hrs.toml"
GLP1_REPORT = ROOT / "validation" / "source_checks" / "glp1_mace_reported_hr_tokens.json"
GLP1_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "glp1_mace_reported_hr_source_check.json"
PARP_FIRSTLINE_MANIFEST = ROOT / "validation" / "survival" / "parp_firstline_ovarian_pfs_reported_hrs.toml"
PARP_FIRSTLINE_REPORT = ROOT / "validation" / "source_checks" / "parp_firstline_ovarian_pfs_reported_hr_tokens.json"
PARP_FIRSTLINE_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "parp_firstline_ovarian_pfs_reported_hr_source_check.json"
PARP_RECURRENT_MANIFEST = ROOT / "validation" / "survival" / "parp_recurrent_ovarian_pfs_reported_hrs.toml"
PARP_RECURRENT_REPORT = ROOT / "validation" / "source_checks" / "parp_recurrent_ovarian_pfs_reported_hr_tokens.json"
PARP_RECURRENT_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "parp_recurrent_ovarian_pfs_reported_hr_source_check.json"
CDK46_MANIFEST = ROOT / "validation" / "survival" / "cdk46_breast_pfs_reported_hrs.toml"
CDK46_REPORT = ROOT / "validation" / "source_checks" / "cdk46_breast_pfs_reported_hr_tokens.json"
CDK46_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "cdk46_breast_pfs_reported_hr_source_check.json"
RCC_MANIFEST = ROOT / "validation" / "survival" / "rcc_firstline_pfs_reported_hrs.toml"
RCC_REPORT = ROOT / "validation" / "source_checks" / "rcc_firstline_pfs_reported_hr_tokens.json"
RCC_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "rcc_firstline_pfs_reported_hr_source_check.json"
NSCLC_MANIFEST = ROOT / "validation" / "survival" / "nsclc_firstline_pfs_reported_hrs.toml"
NSCLC_REPORT = ROOT / "validation" / "source_checks" / "nsclc_firstline_pfs_reported_hr_tokens.json"
NSCLC_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "nsclc_firstline_pfs_reported_hr_source_check.json"
HFREF_MANIFEST = ROOT / "validation" / "survival" / "hfref_therapies_primary_reported_hrs.toml"
HFREF_REPORT = ROOT / "validation" / "source_checks" / "hfref_therapies_primary_reported_hr_tokens.json"
HFREF_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "hfref_therapies_primary_reported_hr_source_check.json"
DOAC_MANIFEST = ROOT / "validation" / "survival" / "doac_af_primary_reported_hrs.toml"
DOAC_REPORT = ROOT / "validation" / "source_checks" / "doac_af_primary_reported_hr_tokens.json"
DOAC_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "doac_af_primary_reported_hr_source_check.json"
TAVI_MANIFEST = ROOT / "validation" / "survival" / "tavi_savr_primary_reported_hrs.toml"
TAVI_REPORT = ROOT / "validation" / "source_checks" / "tavi_savr_primary_reported_hr_tokens.json"
TAVI_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "tavi_savr_primary_reported_hr_source_check.json"
MELANOMA_MANIFEST = ROOT / "validation" / "survival" / "melanoma_pfs_reported_hrs.toml"
MELANOMA_REPORT = ROOT / "validation" / "source_checks" / "melanoma_pfs_reported_hr_tokens.json"
MELANOMA_IDENTITY_REPORT = ROOT / "validation" / "source_checks" / "melanoma_pfs_reported_hr_source_check.json"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_pubmed_survival_hrs.py"
IDENTITY_VERIFY_SCRIPT = ROOT / "scripts" / "verify_survival_sources.py"


@pytest.mark.parametrize(
    ("manifest_path", "benchmark_id", "expected_study_ids"),
    [
        (
            MANIFEST,
            "sglt2_hf_reported_hr",
            {
                "DAPA-HF",
                "EMPEROR-Reduced",
                "DELIVER",
                "EMPEROR-Preserved",
            },
        ),
        (
            PCSK9_MANIFEST,
            "pcsk9_mace_reported_hr",
            {
                "FOURIER",
                "ODYSSEY-Outcomes",
            },
        ),
        (
            CKD_MANIFEST,
            "sglt2_ckd_reported_hr",
            {
                "CREDENCE",
                "DAPA-CKD",
                "EMPA-KIDNEY",
            },
        ),
        (
            GLP1_MANIFEST,
            "glp1_mace_reported_hr",
            {
                "ELIXA",
                "LEADER",
                "SUSTAIN-6",
                "EXSCEL",
                "HARMONY-Outcomes",
                "REWIND",
                "PIONEER-6",
                "AMPLITUDE-O",
            },
        ),
        (
            PARP_FIRSTLINE_MANIFEST,
            "parp_firstline_ovarian_pfs_reported_hr",
            {
                "SOLO1",
                "PAOLA-1",
                "PRIMA",
                "VELIA",
            },
        ),
        (
            PARP_RECURRENT_MANIFEST,
            "parp_recurrent_ovarian_pfs_reported_hr",
            {
                "NOVA",
                "ARIEL3",
                "SOLO2",
                "Study19",
            },
        ),
        (
            CDK46_MANIFEST,
            "cdk46_breast_pfs_reported_hr",
            {
                "PALOMA-2",
                "PALOMA-3",
                "MONALEESA-2",
                "MONARCH-3",
                "MONALEESA-7",
                "MONALEESA-3",
            },
        ),
        (
            RCC_MANIFEST,
            "rcc_firstline_pfs_reported_hr",
            {
                "KEYNOTE-426",
                "JAVELIN-Renal-101",
                "CLEAR",
                "CheckMate-9ER",
                "CABOSUN",
                "COMPARZ",
            },
        ),
        (
            NSCLC_MANIFEST,
            "nsclc_firstline_pfs_reported_hr",
            {
                "KEYNOTE-024",
                "KEYNOTE-189",
                "KEYNOTE-407",
                "IMpower150",
                "IMpower130",
            },
        ),
        (
            HFREF_MANIFEST,
            "hfref_therapies_primary_reported_hr",
            {
                "PARADIGM-HF",
                "EMPHASIS-HF",
                "VICTORIA",
                "GALACTIC-HF",
            },
        ),
        (
            DOAC_MANIFEST,
            "doac_af_primary_reported_hr",
            {
                "ARISTOTLE",
                "ROCKET-AF",
                "AVERROES",
            },
        ),
        (
            TAVI_MANIFEST,
            "tavi_savr_primary_reported_hr",
            {
                "PARTNER-3",
                "PARTNER-2A",
            },
        ),
        (
            MELANOMA_MANIFEST,
            "melanoma_pfs_reported_hr",
            {
                "KEYNOTE-006",
                "CheckMate-066",
                "COMBI-d",
            },
        ),
    ],
)
def test_survival_hr_manifest_is_source_bounded_and_pubmed_only(
    manifest_path, benchmark_id, expected_study_ids
):
    manifest = load_survival_hr_manifest(manifest_path)

    assert manifest.benchmark_id == benchmark_id
    assert manifest.evidence_mode == "reported_hr_pubmed_abstract"
    assert manifest.status == "candidate_source_verified"
    assert manifest.certification_effect == "none"
    assert manifest.manifest_sha256 == sha256_file(manifest_path)
    assert {study.study_id for study in manifest.studies} == expected_study_ids
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


def test_pubmed_survival_hr_verifier_normalizes_lancet_decimal_typography():
    spec = importlib.util.spec_from_file_location(
        "verify_pubmed_survival_hrs", VERIFY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    text = "hazard ratio [HR] 0·88, 95% CI 0·79-0·99"
    normalised = module.normalise_text(text)
    assert "0.88" in normalised
    assert "0.79-0.99" in normalised
    actual_middle_dot = module.normalise_text("hazard ratio 0\u00b755, 95% CI 0\u00b744-0\u00b769")
    assert "0.55" in actual_middle_dot
    assert "0.44-0.69" in actual_middle_dot

    abstract = (
        "The margin used the upper boundary of the confidence interval for "
        "the hazard ratio. Results favored treatment (hazard ratio [HR] "
        "0·88, 95% CI 0·79-0·99)."
    )
    anchor_found, tokens_near = module.tokens_near_anchor(
        abstract,
        "hazard ratio",
        ["0.88", "0.79", "0.99"],
        window=80,
    )
    assert anchor_found is True
    assert tokens_near is True

    ci_found, ci_tokens_near = module.tokens_near_anchor(
        abstract,
        "95% ci",
        ["0.79", "0.99"],
        window=80,
    )
    assert ci_found is True
    assert ci_tokens_near is True


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


@pytest.mark.parametrize(
    ("manifest_path", "report_path", "manifest_relpath"),
    [
        (
            MANIFEST,
            REPORT,
            "validation/survival/sglt2_hf_reported_hrs.toml",
        ),
        (
            PCSK9_MANIFEST,
            PCSK9_REPORT,
            "validation/survival/pcsk9_mace_reported_hrs.toml",
        ),
        (
            CKD_MANIFEST,
            CKD_REPORT,
            "validation/survival/sglt2_ckd_reported_hrs.toml",
        ),
        (
            GLP1_MANIFEST,
            GLP1_REPORT,
            "validation/survival/glp1_mace_reported_hrs.toml",
        ),
        (
            PARP_FIRSTLINE_MANIFEST,
            PARP_FIRSTLINE_REPORT,
            "validation/survival/parp_firstline_ovarian_pfs_reported_hrs.toml",
        ),
        (
            PARP_RECURRENT_MANIFEST,
            PARP_RECURRENT_REPORT,
            "validation/survival/parp_recurrent_ovarian_pfs_reported_hrs.toml",
        ),
        (
            CDK46_MANIFEST,
            CDK46_REPORT,
            "validation/survival/cdk46_breast_pfs_reported_hrs.toml",
        ),
        (
            RCC_MANIFEST,
            RCC_REPORT,
            "validation/survival/rcc_firstline_pfs_reported_hrs.toml",
        ),
        (
            NSCLC_MANIFEST,
            NSCLC_REPORT,
            "validation/survival/nsclc_firstline_pfs_reported_hrs.toml",
        ),
        (
            HFREF_MANIFEST,
            HFREF_REPORT,
            "validation/survival/hfref_therapies_primary_reported_hrs.toml",
        ),
        (
            DOAC_MANIFEST,
            DOAC_REPORT,
            "validation/survival/doac_af_primary_reported_hrs.toml",
        ),
        (
            TAVI_MANIFEST,
            TAVI_REPORT,
            "validation/survival/tavi_savr_primary_reported_hrs.toml",
        ),
        (
            MELANOMA_MANIFEST,
            MELANOMA_REPORT,
            "validation/survival/melanoma_pfs_reported_hrs.toml",
        ),
    ],
)
def test_survival_hr_verification_snapshot_matches_manifest(
    manifest_path, report_path, manifest_relpath
):
    manifest = load_survival_hr_manifest(manifest_path)
    report = load_survival_hr_verification_report(report_path)

    assert report.status == "verified"
    assert report.certification_effect == "none"
    assert report.benchmark_id == manifest.benchmark_id
    assert report.manifest == manifest_relpath
    assert report.manifest_sha256 == sha256_file(manifest_path)
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


@pytest.mark.parametrize(
    ("manifest_path", "identity_report_path", "manifest_relpath", "expected_counts"),
    [
        (
            MANIFEST,
            IDENTITY_REPORT,
            "validation/survival/sglt2_hf_reported_hrs.toml",
            {"clinicaltrials_gov": 4, "pubmed_abstract": 4},
        ),
        (
            PCSK9_MANIFEST,
            PCSK9_IDENTITY_REPORT,
            "validation/survival/pcsk9_mace_reported_hrs.toml",
            {"clinicaltrials_gov": 2, "pubmed_abstract": 2},
        ),
        (
            CKD_MANIFEST,
            CKD_IDENTITY_REPORT,
            "validation/survival/sglt2_ckd_reported_hrs.toml",
            {"clinicaltrials_gov": 3, "pubmed_abstract": 3},
        ),
        (
            GLP1_MANIFEST,
            GLP1_IDENTITY_REPORT,
            "validation/survival/glp1_mace_reported_hrs.toml",
            {"clinicaltrials_gov": 8, "pubmed_abstract": 8},
        ),
        (
            PARP_FIRSTLINE_MANIFEST,
            PARP_FIRSTLINE_IDENTITY_REPORT,
            "validation/survival/parp_firstline_ovarian_pfs_reported_hrs.toml",
            {"clinicaltrials_gov": 4, "pubmed_abstract": 4},
        ),
        (
            PARP_RECURRENT_MANIFEST,
            PARP_RECURRENT_IDENTITY_REPORT,
            "validation/survival/parp_recurrent_ovarian_pfs_reported_hrs.toml",
            {"clinicaltrials_gov": 4, "pubmed_abstract": 4},
        ),
        (
            CDK46_MANIFEST,
            CDK46_IDENTITY_REPORT,
            "validation/survival/cdk46_breast_pfs_reported_hrs.toml",
            {"clinicaltrials_gov": 6, "pubmed_abstract": 6},
        ),
        (
            RCC_MANIFEST,
            RCC_IDENTITY_REPORT,
            "validation/survival/rcc_firstline_pfs_reported_hrs.toml",
            {"clinicaltrials_gov": 6, "pubmed_abstract": 6},
        ),
        (
            NSCLC_MANIFEST,
            NSCLC_IDENTITY_REPORT,
            "validation/survival/nsclc_firstline_pfs_reported_hrs.toml",
            {"clinicaltrials_gov": 5, "pubmed_abstract": 5},
        ),
        (
            HFREF_MANIFEST,
            HFREF_IDENTITY_REPORT,
            "validation/survival/hfref_therapies_primary_reported_hrs.toml",
            {"clinicaltrials_gov": 4, "pubmed_abstract": 4},
        ),
        (
            DOAC_MANIFEST,
            DOAC_IDENTITY_REPORT,
            "validation/survival/doac_af_primary_reported_hrs.toml",
            {"clinicaltrials_gov": 3, "pubmed_abstract": 3},
        ),
        (
            TAVI_MANIFEST,
            TAVI_IDENTITY_REPORT,
            "validation/survival/tavi_savr_primary_reported_hrs.toml",
            {"clinicaltrials_gov": 2, "pubmed_abstract": 2},
        ),
        (
            MELANOMA_MANIFEST,
            MELANOMA_IDENTITY_REPORT,
            "validation/survival/melanoma_pfs_reported_hrs.toml",
            {"clinicaltrials_gov": 3, "pubmed_abstract": 3},
        ),
    ],
)
def test_survival_hr_identity_snapshot_matches_manifest(
    manifest_path, identity_report_path, manifest_relpath, expected_counts
):
    manifest = load_survival_hr_manifest(manifest_path)
    report = load_source_verification_report(identity_report_path)

    assert report.status == "verified"
    assert report.certification_effect == "none"
    assert report.benchmark_id == manifest.benchmark_id
    assert report.source_manifest == manifest_relpath
    assert report.source_manifest_sha256 == sha256_file(manifest_path)
    assert summarize_source_verification(report) == expected_counts
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
            assert record.details["overall_status"] in {
                "ACTIVE_NOT_RECRUITING",
                "COMPLETED",
                "TERMINATED",
            }
        if record.source_type == "pubmed_abstract":
            assert record.details["pmid"] == record.identifier
            assert record.details["abstract_present"] is True
            assert len(record.details["abstract_sha256"]) == 64

    bundle = validate_survival_hr_identity_bundle(manifest, report)
    assert bundle["source_counts"] == expected_counts


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
