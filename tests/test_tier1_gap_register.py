import copy
from pathlib import Path

import pytest

from bias_nma_adv.tier1_gap_register import (
    REQUIRED_BLOCKED_CLAIMS,
    REQUIRED_GAP_IDS,
    TIER1_GAP_REGISTER_SCHEMA_VERSION,
    Tier1GapRegister,
    Tier1GapRegisterError,
    load_tier1_gap_register,
    summarize_tier1_gap_register,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "validation" / "tier1_gap_register.toml"


def test_tier1_gap_register_keeps_current_shortcomings_blocking():
    register = load_tier1_gap_register(REGISTER)

    assert register.certification_effect == "none"
    assert set(register.blocked_claims) >= REQUIRED_BLOCKED_CLAIMS
    assert {gap.id for gap in register.gaps} == REQUIRED_GAP_IDS
    assert {gap.status for gap in register.gaps} == {"blocking"}

    by_id = {gap.id: gap for gap in register.gaps}
    assert "node_splitting" in by_id["feature_completeness"].missing_capabilities
    assert (
        "multiarm_gls_influence_leverage_diagnostics"
        in by_id["feature_completeness"].implemented_capabilities
    )
    assert "optimizer_stress_matrix" in by_id["numerical_stability"].missing_capabilities
    assert "cmdstan_backend" in by_id["bayesian_ecosystem_integration"].missing_capabilities
    assert "multinma" in by_id["bayesian_ecosystem_integration"].tier_one_references


def test_tier1_gap_register_summary_is_validation_status_ready():
    summary = summarize_tier1_gap_register(load_tier1_gap_register(REGISTER))

    assert summary == {
        "schema_version": TIER1_GAP_REGISTER_SCHEMA_VERSION,
        "checked_at": "2026-07-15",
        "n_gaps": 3,
        "gap_ids": [
            "feature_completeness",
            "numerical_stability",
            "bayesian_ecosystem_integration",
        ],
        "status_counts": {"blocking": 3},
        "implemented_capabilities": {
            "feature_completeness": ["multiarm_gls_influence_leverage_diagnostics"]
        },
        "blocked_claims": [
            "tier_one_parity",
            "tier_one_superiority",
            "production_certification",
            "clinical_reporting",
            "hta_reporting",
        ],
        "certification_effect": "none",
    }


def test_tier1_gap_register_rejects_removed_gap_or_softened_status():
    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["gaps"] = [gap for gap in raw["gaps"] if gap["id"] != "feature_completeness"]

    with pytest.raises(Tier1GapRegisterError, match="feature_completeness"):
        Tier1GapRegister.from_mapping(raw)

    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["gaps"][0]["status"] = "resolved"

    with pytest.raises(Tier1GapRegisterError, match="resolved gaps require"):
        Tier1GapRegister.from_mapping(raw)


def test_tier1_gap_register_rejects_certification_or_missing_blocked_claim():
    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["certification_effect"] = "production_certified"

    with pytest.raises(Tier1GapRegisterError, match="cannot certify"):
        Tier1GapRegister.from_mapping(raw)

    raw = _register_to_mapping(load_tier1_gap_register(REGISTER))
    raw["blocked_claims"] = [
        claim for claim in raw["blocked_claims"] if claim != "tier_one_superiority"
    ]

    with pytest.raises(Tier1GapRegisterError, match="tier_one_superiority"):
        Tier1GapRegister.from_mapping(raw)


def _register_to_mapping(register: Tier1GapRegister) -> dict[str, object]:
    return {
        "schema_version": TIER1_GAP_REGISTER_SCHEMA_VERSION,
        "checked_at": register.checked_at,
        "certification_effect": register.certification_effect,
        "purpose": register.purpose,
        "source_boundary": register.source_boundary,
        "superiority_claim_rule": register.superiority_claim_rule,
        "blocked_claims": list(register.blocked_claims),
        "gaps": [
            {
                "id": gap.id,
                "status": gap.status,
                "review_source": gap.review_source,
                "summary": gap.summary,
                "tier_one_references": list(gap.tier_one_references),
                "missing_capabilities": list(gap.missing_capabilities),
                "implemented_capabilities": list(gap.implemented_capabilities),
                "required_evidence_artifacts": list(gap.required_evidence_artifacts),
                "claim_limit": gap.claim_limit,
                "certification_effect": gap.certification_effect,
            }
            for gap in copy.deepcopy(register.gaps)
        ],
    }
