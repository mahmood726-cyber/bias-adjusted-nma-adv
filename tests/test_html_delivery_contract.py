import copy
from pathlib import Path

import pytest

from bias_nma_adv.html_delivery_contract import (
    HTML_DELIVERY_CONTRACT_SCHEMA_VERSION,
    HTML_ONLY_BLOCKED_IDS,
    REQUIRED_CAPABILITY_IDS,
    HTMLDeliveryContract,
    HTMLDeliveryContractError,
    load_html_delivery_contract,
    summarize_html_delivery_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "validation" / "html_delivery_contract.toml"


def test_html_delivery_contract_separates_dashboards_from_engines():
    contract = load_html_delivery_contract(CONTRACT)

    assert contract.certification_effect == "none"
    by_id = {capability.id: capability for capability in contract.capabilities}
    assert set(by_id) == REQUIRED_CAPABILITY_IDS
    assert by_id["static_validation_dashboard"].delivery_mode == "static_html_allowed"
    assert by_id["interactive_artifact_explorer"].status == "allowed"

    for capability_id in HTML_ONLY_BLOCKED_IDS:
        capability = by_id[capability_id]
        assert capability.delivery_mode == "backend_required"
        assert capability.status == "blocked_for_html_only"
        assert capability.certification_effect == "none"

    assert "javascript_only_reimplementation_without_numeric_parity_tests" in by_id[
        "statistical_estimation_engine"
    ].forbidden_replacements
    assert "html_declares_adapter_passed_without_run_report" in by_id[
        "reference_software_adapters"
    ].forbidden_replacements


def test_html_delivery_contract_summary_is_validation_status_ready():
    summary = summarize_html_delivery_contract(load_html_delivery_contract(CONTRACT))

    assert summary == {
        "schema_version": HTML_DELIVERY_CONTRACT_SCHEMA_VERSION,
        "checked_at": "2026-07-15",
        "n_capabilities": 6,
        "delivery_mode_counts": {
            "backend_required": 4,
            "static_html_allowed": 2,
        },
        "status_counts": {
            "allowed": 2,
            "blocked_for_html_only": 4,
        },
        "html_only_blocked_ids": [
            "ci_and_certification_gates",
            "live_source_verification",
            "reference_software_adapters",
            "statistical_estimation_engine",
        ],
        "certification_effect": "none",
    }


def test_html_delivery_contract_rejects_html_only_engine_downgrade():
    raw = _contract_to_mapping(load_html_delivery_contract(CONTRACT))
    for item in raw["capabilities"]:
        if item["id"] == "statistical_estimation_engine":
            item["delivery_mode"] = "static_html_allowed"
            item["status"] = "allowed"

    with pytest.raises(HTMLDeliveryContractError, match="backend_required"):
        HTMLDeliveryContract.from_mapping(raw)


def test_html_delivery_contract_rejects_certification_or_missing_capability():
    raw = _contract_to_mapping(load_html_delivery_contract(CONTRACT))
    raw["certification_effect"] = "production_certified"

    with pytest.raises(HTMLDeliveryContractError, match="cannot certify"):
        HTMLDeliveryContract.from_mapping(raw)

    raw = _contract_to_mapping(load_html_delivery_contract(CONTRACT))
    raw["capabilities"] = [
        item for item in raw["capabilities"] if item["id"] != "reference_software_adapters"
    ]

    with pytest.raises(HTMLDeliveryContractError, match="reference_software_adapters"):
        HTMLDeliveryContract.from_mapping(raw)


def _contract_to_mapping(contract: HTMLDeliveryContract) -> dict[str, object]:
    return {
        "schema_version": HTML_DELIVERY_CONTRACT_SCHEMA_VERSION,
        "checked_at": contract.checked_at,
        "certification_effect": contract.certification_effect,
        "purpose": contract.purpose,
        "global_rule": contract.global_rule,
        "capabilities": [
            {
                "id": capability.id,
                "delivery_mode": capability.delivery_mode,
                "status": capability.status,
                "summary": capability.summary,
                "required_backing_artifacts": list(capability.required_backing_artifacts),
                "forbidden_replacements": list(capability.forbidden_replacements),
                "regression_risk": capability.regression_risk,
                "certification_effect": capability.certification_effect,
            }
            for capability in copy.deepcopy(contract.capabilities)
        ],
    }
