import json
from pathlib import Path

import pytest

from bias_nma_adv.proof_effect_bundle import (
    PROOF_EFFECT_BUNDLE_SCHEMA_VERSION,
    ProofEffectBundleError,
    load_proof_effect_bundle,
    validate_proof_effect_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "validation" / "ingestion" / "sglt2_hf_reported_hr_proof_effects.json"


def test_validates_source_backed_sglt2_reported_hr_proof_effect_bundle():
    summary = validate_proof_effect_bundle(BUNDLE, repo_root=ROOT)

    assert summary == {
        "schema_version": PROOF_EFFECT_BUNDLE_SCHEMA_VERSION,
        "bundle_id": "sglt2_hf_reported_hr_proof_effects",
        "status": "local_pass",
        "n_records": 4,
        "effect_type_counts": {"HR": 4},
        "source_type_counts": {"pubmed_abstract": 4},
        "certification_effect": "none",
    }


def test_bundle_records_are_non_certifying_and_carry_minimal_hr_ci_snippets():
    payload = load_proof_effect_bundle(BUNDLE)

    assert payload["certification_effect"] == "none"
    assert len(payload["records"]) == 4
    for record in payload["records"]:
        snippet = record["provenance"]["source_text"]
        assert record["certification_effect"] == "none"
        assert record["effect_type"] == "HR"
        assert "hazard ratio" in snippet
        assert str(record["point_estimate"]) in snippet
        assert str(record["ci_lower"]) in snippet
        assert f"{record['ci_upper']:.2f}" in snippet or str(record["ci_upper"]) in snippet
        assert len(snippet.split()) <= 20


def test_bundle_rejects_any_attempt_to_claim_certification(tmp_path):
    mutated = _write_mutated_bundle(tmp_path, certification_effect="production_certified")

    with pytest.raises(ProofEffectBundleError, match="cannot certify"):
        validate_proof_effect_bundle(mutated, repo_root=ROOT)


def test_bundle_rejects_source_report_hash_drift(tmp_path):
    mutated = _write_mutated_bundle(tmp_path, source_manifest_sha256="0" * 64)

    with pytest.raises(ProofEffectBundleError, match="source manifest SHA-256 mismatch"):
        validate_proof_effect_bundle(mutated, repo_root=ROOT)


def test_bundle_rejects_snippet_without_reported_hr_token(tmp_path):
    payload = load_proof_effect_bundle(BUNDLE)
    payload["records"][0]["provenance"]["source_text"] = payload["records"][0]["provenance"][
        "source_text"
    ].replace("0.74", "0.xx")
    mutated = tmp_path / "mutated_bundle.json"
    mutated.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ProofEffectBundleError, match="missing token 0.74"):
        validate_proof_effect_bundle(mutated, repo_root=ROOT)


def _write_mutated_bundle(tmp_path: Path, **overrides) -> Path:
    payload = load_proof_effect_bundle(BUNDLE)
    payload.update(overrides)
    mutated = tmp_path / "mutated_bundle.json"
    mutated.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return mutated
