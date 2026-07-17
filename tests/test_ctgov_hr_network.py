import csv
import copy
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.ctgov_hr_network import (
    CTGovHRNetworkManifest,
    CTGovHRNetworkVerificationReport,
    ctgov_hr_network_log_effects,
    load_ctgov_hr_network_manifest,
    load_ctgov_hr_network_verification_report,
    run_ctgov_hr_network_benchmark,
    sparse_design_bias_diagnostics,
    summarize_sparse_design_bias_diagnostics,
    validate_ctgov_hr_network_source_bundle,
)
from bias_nma_adv.data import ValidationError
from bias_nma_adv.real_meta import sha256_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "networks" / "t2d_mace_ctgov_hrs.toml"
REPORT = ROOT / "validation" / "source_checks" / "t2d_mace_ctgov_hr_network_check.json"
ARTIFACT = ROOT / "validation" / "networks" / "t2d_mace_ctgov_hr_network_benchmark.toml"
EFFECTS_CSV = ROOT / "validation" / "networks" / "t2d_mace_ctgov_hr_network_effects.csv"
WRITER_SCRIPT = ROOT / "scripts" / "write_ctgov_hr_network_benchmark.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_ctgov_hr_network.py"


def test_ctgov_hr_network_manifest_is_source_bounded():
    manifest = load_ctgov_hr_network_manifest(MANIFEST)

    assert manifest.benchmark_id == "t2d_mace_ctgov_hr_network"
    assert manifest.evidence_mode == "reported_hr_clinicaltrials_gov_results"
    assert manifest.status == "candidate_source_verified"
    assert manifest.certification_effect == "none"
    assert manifest.reference_treatment == "placebo"
    assert manifest.network_type == "star_class_network"
    assert manifest.reuse_origin == "complex_evidence_synthesis_map_pattern_only"
    assert manifest.manifest_sha256 == sha256_file(MANIFEST)
    assert len(manifest.studies) == 10
    assert {study.analysis_treatment for study in manifest.studies} == {
        "DPP-4 inhibitor",
        "GLP-1 RA",
        "SGLT2 inhibitor",
    }
    assert {study.nct_id for study in manifest.studies} == {
        "NCT00790205",
        "NCT01032629",
        "NCT01131676",
        "NCT01144338",
        "NCT01394952",
        "NCT01897532",
        "NCT01986881",
        "NCT02065791",
        "NCT02465515",
        "NCT03914326",
    }
    for study in manifest.studies:
        assert study.source_type == "clinicaltrials_gov"
        assert study.source_url.startswith("https://clinicaltrials.gov/study/")
        assert study.effect_direction == "active_vs_control"
        assert study.control_treatment == "placebo"
        assert study.reported_hr
        assert study.ci_lower
        assert study.ci_upper
        assert study.active_drug in study.source_terms
        assert "placebo" in study.source_terms


def test_ctgov_hr_network_verification_snapshot_matches_manifest():
    manifest = load_ctgov_hr_network_manifest(MANIFEST)
    report = load_ctgov_hr_network_verification_report(REPORT)

    assert VERIFY_SCRIPT.is_file()
    assert report.status == "verified"
    assert report.certification_effect == "none"
    assert report.benchmark_id == manifest.benchmark_id
    assert report.manifest == "validation/networks/t2d_mace_ctgov_hrs.toml"
    assert report.manifest_sha256 == sha256_file(MANIFEST)
    assert len(report.records) == len(manifest.studies)

    expected = {
        (
            study.study_id,
            study.nct_id,
            study.outcome_id,
            study.reported_hr,
            study.ci_lower,
            study.ci_upper,
            study.source_terms,
            study.outcome_search_terms,
        )
        for study in manifest.studies
    }
    observed = {
        (
            record.study_id,
            record.nct_id,
            record.outcome_id,
            record.reported_hr,
            record.ci_lower,
            record.ci_upper,
            record.source_terms,
            record.outcome_search_terms,
        )
        for record in report.records
    }
    assert observed == expected

    for record in report.records:
        assert record.evidence_scope == "clinicaltrials_gov_reported_hr_analysis"
        assert len(record.response_sha256) == 64
        assert record.nct_id_found is True
        assert record.status_completed is True
        assert record.hazard_ratio_analysis_found is True
        assert record.ci_tokens_found is True
        assert record.outcome_terms_found is True
        assert record.source_terms_found is True
        assert record.matched_param_type == "Hazard Ratio (HR)"
        assert record.matched_outcome_title
        assert record.verified is True

    bundle = validate_ctgov_hr_network_source_bundle(manifest, report)
    assert bundle == {
        "benchmark_id": manifest.benchmark_id,
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": "verified",
        "certification_effect": "none",
        "n_studies": 10,
        "source_counts": {"clinicaltrials_gov": 10},
    }


def test_ctgov_hr_network_artifact_recomputes_from_verified_sources():
    with ARTIFACT.open("rb") as handle:
        artifact = tomllib.load(handle)

    assert artifact["schema_version"] == "ctgov_hr_network_benchmark/v1"
    assert artifact["status"] == "local_pass"
    assert artifact["certification_effect"] == "none"
    assert artifact["source_policy"] == "clinicaltrials_gov + pubmed_abstract + open_access_paper only"
    assert artifact["evidence_mode"] == "reported_hr_clinicaltrials_gov_results"
    assert artifact["network_type"] == "star_class_network"
    assert artifact["effect_scale"] == "log_hr"
    assert artifact["reference_treatment"] == "placebo"
    assert artifact["n_studies"] == 10
    assert artifact["n_treatments"] == 4
    assert "closed-loop inconsistency cannot be assessed" in " ".join(artifact["limitations"])

    manifest_path = ROOT / artifact["source_manifest"]
    report_path = ROOT / artifact["source_verification_report"]
    assert sha256_file(manifest_path) == artifact["source_manifest_sha256"]
    assert sha256_file(report_path) == artifact["source_verification_report_sha256"]

    manifest = load_ctgov_hr_network_manifest(manifest_path)
    report = load_ctgov_hr_network_verification_report(report_path)
    assert validate_ctgov_hr_network_source_bundle(manifest, report) == artifact["source_bundle"]

    recomputed = run_ctgov_hr_network_benchmark(manifest_path, verification_report_path=report_path)
    assert recomputed["source_manifest_sha256"] == artifact["source_manifest_sha256"]
    assert recomputed["source_verification_report_sha256"] == artifact["source_verification_report_sha256"]
    assert recomputed["source_bundle"] == artifact["source_bundle"]
    assert recomputed["model_config"] == artifact["model_config"]

    expected_effects = {effect.study_id: effect for effect in ctgov_hr_network_log_effects(manifest)}
    artifact_effects = {effect["study_id"]: effect for effect in artifact["study_effects"]}
    assert set(artifact_effects) == set(expected_effects)
    for study_id, expected in expected_effects.items():
        observed = artifact_effects[study_id]
        for field in (
            "study_id",
            "trial",
            "nct_id",
            "outcome_id",
            "outcome_label",
            "active_drug",
            "analysis_treatment",
            "control_treatment",
            "effect_direction",
            "effect_scale",
            "variance_source",
        ):
            assert observed[field] == getattr(expected, field)
        for field in ("reported_hr", "ci_lower", "ci_upper", "estimate", "variance", "se"):
            assert abs(observed[field] - getattr(expected, field)) < 1e-14

    for key in ("fixed_gls", "random_gls"):
        observed = artifact["candidate"][key]
        expected = recomputed["candidate"][key]
        for field in (
            "model",
            "reference_treatment",
            "treatments",
            "nonreference_treatments",
            "df",
            "multi_arm_studies",
            "warnings",
        ):
            assert observed[field] == expected[field]
        for field in ("tau2", "q"):
            if expected[field] == "":
                assert observed[field] == ""
            else:
                assert abs(observed[field] - expected[field]) < 1e-14
        for observed_effect, expected_effect in zip(observed["effects"], expected["effects"], strict=True):
            for field in ("treatment", "reference_treatment"):
                assert observed_effect[field] == expected_effect[field]
            for field in (
                "estimate",
                "se",
                "ci_low",
                "ci_high",
                "hr",
                "hr_ci_low",
                "hr_ci_high",
            ):
                assert abs(observed_effect[field] - expected_effect[field]) < 1e-14


def test_ctgov_hr_network_effects_csv_matches_source_backed_artifact():
    with ARTIFACT.open("rb") as handle:
        artifact = tomllib.load(handle)
    with EFFECTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected = {row["study_id"]: row for row in artifact["study_effects"]}
    observed = {row["study_id"]: row for row in rows}
    assert set(observed) == set(expected)
    for study_id, expected_row in expected.items():
        observed_row = observed[study_id]
        assert observed_row["nct_id"] == expected_row["nct_id"]
        assert observed_row["analysis_treatment"] == expected_row["analysis_treatment"]
        assert observed_row["control_treatment"] == expected_row["control_treatment"]
        assert float(observed_row["estimate"]) == pytest.approx(expected_row["estimate"])
        assert float(observed_row["se"]) == pytest.approx(expected_row["se"])
        assert float(observed_row["variance"]) == pytest.approx(expected_row["variance"])


def test_ctgov_hr_network_writer_regenerates_same_artifact(tmp_path):
    output = tmp_path / "ctgov_network.toml"
    completed = subprocess.run(
        [
            sys.executable,
            str(WRITER_SCRIPT),
            "--manifest",
            "validation/networks/t2d_mace_ctgov_hrs.toml",
            "--source-check",
            "validation/source_checks/t2d_mace_ctgov_hr_network_check.json",
            "--output",
            str(output),
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    with ARTIFACT.open("rb") as handle:
        expected = tomllib.load(handle)
    with output.open("rb") as handle:
        regenerated = tomllib.load(handle)
    assert regenerated == expected


def test_ctgov_hr_network_manifest_rejects_certification_effect():
    raw = copy.deepcopy(load_ctgov_hr_network_manifest(MANIFEST).__dict__)
    raw["schema_version"] = "ctgov_hr_network_manifest/v1"
    raw["certification_effect"] = "reference_matched"
    raw["studies"] = [copy.deepcopy(study.__dict__) for study in raw["studies"]]

    with pytest.raises(ValidationError, match="cannot certify model performance"):
        CTGovHRNetworkManifest.from_mapping(raw)


def test_ctgov_hr_network_report_rejects_unverified_record_marked_verified():
    raw = copy.deepcopy(load_ctgov_hr_network_verification_report(REPORT).__dict__)
    raw["schema_version"] = "ctgov_hr_network_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["hazard_ratio_analysis_found"] = False

    with pytest.raises(ValidationError, match="missing source evidence"):
        CTGovHRNetworkVerificationReport.from_mapping(raw)


def test_ctgov_hr_network_source_bundle_rejects_record_drift():
    manifest = load_ctgov_hr_network_manifest(MANIFEST)
    raw = copy.deepcopy(load_ctgov_hr_network_verification_report(REPORT).__dict__)
    raw["schema_version"] = "ctgov_hr_network_verification/v1"
    raw["records"] = [copy.deepcopy(record.__dict__) for record in raw["records"]]
    raw["records"][0]["reported_hr"] = "0.99"
    report = CTGovHRNetworkVerificationReport.from_mapping(raw)

    with pytest.raises(ValidationError, match="does not match manifest"):
        validate_ctgov_hr_network_source_bundle(manifest, report)


def test_sparse_design_bias_diagnostic_blocks_underidentified_cross_design_borrowing():
    rows = [
        {"analysis_treatment": "A", "design": "rct"},
        {"analysis_treatment": "A", "design": "nrs"},
        {"analysis_treatment": "B", "design": "rct"},
        {"analysis_treatment": "B", "design": "rct"},
        {"analysis_treatment": "B", "design": "nrs"},
        {"analysis_treatment": "B", "design": "nrs"},
        {"analysis_treatment": "C", "design": "rct"},
        {"analysis_treatment": "C", "design": "rct"},
    ]

    diagnostics = sparse_design_bias_diagnostics(rows, min_per_design=2)
    by_treatment = {item.treatment: item for item in diagnostics}

    assert by_treatment["A"].status == "underidentified_sparse_design_bias"
    assert by_treatment["A"].design_counts == {"nrs": 1, "rct": 1}
    assert by_treatment["A"].certification_effect == "none"
    assert "hierarchical shrinkage" in by_treatment["A"].warnings[0]
    assert by_treatment["B"].status == "sufficient_local_design_replication"
    assert by_treatment["C"].status == "single_design_no_cross_design_bias_contrast"

    assert summarize_sparse_design_bias_diagnostics(diagnostics) == {
        "schema_version": "sparse_design_bias_diagnostic/v1",
        "certification_effect": "none",
        "n_treatments": 3,
        "status_counts": {
            "single_design_no_cross_design_bias_contrast": 1,
            "sufficient_local_design_replication": 1,
            "underidentified_sparse_design_bias": 1,
        },
        "blocked_cross_design_borrowing_treatments": ["A", "C"],
    }


def test_sparse_design_bias_diagnostic_rejects_missing_design_fields():
    with pytest.raises(ValidationError, match="missing key 'design'"):
        sparse_design_bias_diagnostics([{"analysis_treatment": "A"}])
