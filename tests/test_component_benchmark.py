import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.component_benchmark import (
    COMPONENT_BENCHMARK_SCHEMA_VERSION,
    ComponentManifest,
    ComponentVerificationReport,
    component_contrasts,
    load_component_manifest,
    load_component_verification_report,
    run_component_benchmark,
    validate_component_source_bundle,
)
from bias_nma_adv.data import ValidationError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "component" / "sitagliptin_pioglitazone_component.toml"
SOURCE_CHECK = ROOT / "validation" / "source_checks" / "sitagliptin_pioglitazone_component_check.json"
ARTIFACT = ROOT / "validation" / "component" / "sitagliptin_pioglitazone_component_source_benchmark.toml"
WRITE_SCRIPT = ROOT / "scripts" / "write_component_benchmark.py"


def test_component_manifest_source_check_and_contrasts_validate():
    manifest = load_component_manifest(MANIFEST)
    report = load_component_verification_report(SOURCE_CHECK)

    source_bundle = validate_component_source_bundle(manifest, report)
    assert source_bundle == {
        "benchmark_id": "sitagliptin_pioglitazone_component",
        "manifest_sha256": manifest.manifest_sha256,
        "verification_status": "verified",
        "certification_effect": "none",
        "n_records": 2,
        "source_counts": {"clinicaltrials_gov": 1, "pubmed_abstract": 1},
    }

    contrasts = component_contrasts(manifest)
    assert len(contrasts) == 21
    assert contrasts[0]["nct_id"] == "NCT00722371"
    assert contrasts[0]["pmid"] == "23909985"
    assert contrasts[0]["treat1"] == "pioglitazone_15"
    assert contrasts[0]["treat2"] == "sitagliptin_100"
    assert contrasts[0]["estimate"] == pytest.approx(0.21)
    assert contrasts[-1]["treat1"] == "sitagliptin_100 + pioglitazone_45"
    assert contrasts[-1]["treat2"] == "sitagliptin_100 + pioglitazone_30"
    assert all(item["variance"] > 0 for item in contrasts)
    assert {
        "same-trial covariance not modeled" in item["variance_source"]
        for item in contrasts
    } == {True}


def test_component_benchmark_artifact_recomputes_from_verified_sources():
    expected = tomllib.loads(ARTIFACT.read_text(encoding="utf-8"))
    observed = run_component_benchmark(
        MANIFEST,
        verification_report_path=SOURCE_CHECK,
    )

    assert observed["schema_version"] == COMPONENT_BENCHMARK_SCHEMA_VERSION
    assert observed["certification_effect"] == "none"
    assert observed["source_policy"] == (
        "clinicaltrials_gov + pubmed_abstract + open_access_paper only"
    )
    assert observed["n_studies"] == 1
    assert observed["n_component_contrasts"] == 21
    assert observed["study_effects"] == expected["study_effects"]
    assert observed["candidate"]["additive_component_wls"]["rank"] == 4
    assert observed["candidate"]["additive_component_wls"]["rank"] == expected[
        "candidate"
    ]["additive_component_wls"]["rank"]
    assert "not broad netmeta CNMA parity" in observed["limitations"]
    assert "does not certify model performance" in observed["limitations"]


def test_write_component_benchmark_script_regenerates_artifact(tmp_path):
    output = tmp_path / "component_benchmark.toml"
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


def test_component_sources_reject_certification_or_unverified_reports():
    manifest_payload = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_payload["certification_effect"] = "production_certified"
    with pytest.raises(ValidationError, match="cannot certify"):
        ComponentManifest.from_mapping(manifest_payload)

    report_payload = json.loads(SOURCE_CHECK.read_text(encoding="utf-8"))
    report_payload["records"][0]["verified"] = False
    with pytest.raises(ValidationError, match="not verified"):
        ComponentVerificationReport.from_mapping(report_payload)
