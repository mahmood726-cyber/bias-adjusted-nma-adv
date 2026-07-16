import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.data import ValidationError
from bias_nma_adv.dose_response_benchmark import (
    DOSE_RESPONSE_BENCHMARK_SCHEMA_VERSION,
    DoseResponseManifest,
    DoseResponseVerificationReport,
    dose_response_effects,
    load_dose_response_manifest,
    load_dose_response_verification_report,
    run_dose_response_benchmark,
    validate_dose_response_source_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "dose_response" / "semaglutide_obesity_dose_response.toml"
SOURCE_CHECK = ROOT / "validation" / "source_checks" / "semaglutide_obesity_dose_response_check.json"
ARTIFACT = ROOT / "validation" / "dose_response" / "semaglutide_obesity_dose_response_benchmark.toml"
WRITE_SCRIPT = ROOT / "scripts" / "write_dose_response_benchmark.py"


def test_dose_response_manifest_source_check_and_effects_validate():
    manifest = load_dose_response_manifest(MANIFEST)
    report = load_dose_response_verification_report(SOURCE_CHECK)

    source_bundle = validate_dose_response_source_bundle(manifest, report)
    assert source_bundle == {
        "benchmark_id": "semaglutide_obesity_dose_response",
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": "verified",
        "certification_effect": "none",
        "n_records": 2,
        "source_counts": {"clinicaltrials_gov": 1, "pubmed_abstract": 1},
    }

    effects = dose_response_effects(manifest)
    assert len(effects) == 5
    assert effects[0].nct_id == "NCT02453711"
    assert effects[0].pmid == "30122305"
    assert effects[0].estimate == pytest.approx(-3.70)
    assert effects[-1].dose == pytest.approx(0.4)
    assert effects[-1].estimate == pytest.approx(-11.55)
    assert all(effect.variance > 0 for effect in effects)
    assert {
        "shared placebo covariance not modeled" in effect.variance_source
        for effect in effects
    } == {True}


def test_dose_response_benchmark_artifact_recomputes_from_verified_sources():
    expected = tomllib.loads(ARTIFACT.read_text(encoding="utf-8"))
    observed = run_dose_response_benchmark(
        MANIFEST,
        verification_report_path=SOURCE_CHECK,
    )

    assert observed["schema_version"] == DOSE_RESPONSE_BENCHMARK_SCHEMA_VERSION
    assert observed["certification_effect"] == "none"
    assert observed["source_policy"] == (
        "clinicaltrials_gov + pubmed_abstract + open_access_paper only"
    )
    assert observed["n_studies"] == 1
    assert observed["n_dose_effects"] == 5
    assert observed["study_effects"] == expected["study_effects"]
    assert observed["candidate"]["weighted_quadratic"]["q"] == pytest.approx(
        expected["candidate"]["weighted_quadratic"]["q"]
    )
    assert "not MBNMAdose reference matched" in observed["limitations"]
    assert "does not certify model performance" in observed["limitations"]


def test_write_dose_response_benchmark_script_regenerates_artifact(tmp_path):
    output = tmp_path / "dose_response_benchmark.toml"
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

    assert tomllib.loads(output.read_text(encoding="utf-8")) == tomllib.loads(
        ARTIFACT.read_text(encoding="utf-8")
    )


def test_dose_response_sources_reject_certification_or_unverified_reports():
    manifest_payload = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_payload["certification_effect"] = "production_certified"
    with pytest.raises(ValidationError, match="cannot certify"):
        DoseResponseManifest.from_mapping(manifest_payload)

    report_payload = json.loads(SOURCE_CHECK.read_text(encoding="utf-8"))
    report_payload["records"][0]["verified"] = False
    with pytest.raises(ValidationError, match="not verified"):
        DoseResponseVerificationReport.from_mapping(report_payload)
