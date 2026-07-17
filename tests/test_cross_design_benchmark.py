import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.cross_design_benchmark import (
    CROSS_DESIGN_HR_BENCHMARK_SCHEMA_VERSION,
    CrossDesignHRManifest,
    CrossDesignHRVerificationReport,
    cross_design_log_effects,
    load_cross_design_manifest,
    load_cross_design_verification_report,
    run_cross_design_benchmark,
    validate_cross_design_source_bundle,
)
from bias_nma_adv.data import ValidationError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "cross_design" / "sglt2_rct_nrs_cross_design.toml"
SOURCE_CHECK = ROOT / "validation" / "source_checks" / "sglt2_rct_nrs_cross_design_check.json"
ARTIFACT = ROOT / "validation" / "cross_design" / "sglt2_rct_nrs_cross_design_benchmark.toml"
WRITE_SCRIPT = ROOT / "scripts" / "write_cross_design_benchmark.py"


def test_cross_design_manifest_source_check_and_effects_validate():
    manifest = load_cross_design_manifest(MANIFEST)
    report = load_cross_design_verification_report(SOURCE_CHECK)

    source_bundle = validate_cross_design_source_bundle(manifest, report)
    assert source_bundle == {
        "benchmark_id": "sglt2_rct_nrs_cross_design",
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": "verified",
        "certification_effect": "none",
        "n_studies": 4,
        "design_counts": {"nrs": 2, "rct": 2},
        "source_counts": {"pubmed_abstract": 4},
    }

    effects = cross_design_log_effects(manifest)
    assert {effect.design for effect in effects} == {"rct", "nrs"}
    assert effects[0].study_id == "DAPA-HF"
    assert effects[0].estimate == pytest.approx(-0.3011050928)
    assert effects[-1].study_id == "CVD-REAL-2"
    assert effects[-1].reported_hr == pytest.approx(0.60)
    assert all(effect.variance > 0 for effect in effects)


def test_cross_design_benchmark_artifact_recomputes_from_verified_sources():
    expected = tomllib.loads(ARTIFACT.read_text(encoding="utf-8"))
    observed = run_cross_design_benchmark(
        MANIFEST,
        verification_report_path=SOURCE_CHECK,
    )

    assert observed["schema_version"] == CROSS_DESIGN_HR_BENCHMARK_SCHEMA_VERSION
    assert observed["certification_effect"] == "none"
    assert observed["n_studies"] == 4
    assert observed["study_effects"] == expected["study_effects"]
    assert observed["compatibility"]["status"] == "separated_only_estimand_mismatch"
    assert observed["compatibility"]["combined_borrowing_allowed"] is False
    assert "control_treatment" in observed["compatibility"]["mismatched_fields"]
    assert observed["candidate"]["separated_by_design"]["rct"]["hr"] == pytest.approx(
        expected["candidate"]["separated_by_design"]["rct"]["hr"]
    )
    assert observed["candidate"]["separated_by_design"]["nrs"]["hr"] == pytest.approx(
        expected["candidate"]["separated_by_design"]["nrs"]["hr"]
    )
    assert any("not crossnma reference matching" in item for item in observed["limitations"])
    assert "does not certify model performance" in observed["limitations"]


def test_write_cross_design_benchmark_script_regenerates_artifact(tmp_path):
    output = tmp_path / "cross_design_benchmark.toml"
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


def test_cross_design_sources_reject_certification_or_unverified_reports():
    manifest_payload = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_payload["certification_effect"] = "production_certified"
    with pytest.raises(ValidationError, match="cannot certify"):
        CrossDesignHRManifest.from_mapping(manifest_payload)

    report_payload = json.loads(SOURCE_CHECK.read_text(encoding="utf-8"))
    report_payload["records"][0]["verified"] = False
    with pytest.raises(ValidationError, match="unverified"):
        CrossDesignHRVerificationReport.from_mapping(report_payload)
