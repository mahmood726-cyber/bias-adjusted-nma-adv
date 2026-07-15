"""HTML delivery contract for dashboards and validation surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


HTML_DELIVERY_CONTRACT_SCHEMA_VERSION = "html_delivery_contract/v1"
ALLOWED_DELIVERY_MODES = {"static_html_allowed", "backend_required"}
ALLOWED_STATUSES = {"allowed", "blocked_for_html_only"}
REQUIRED_CAPABILITY_IDS = {
    "static_validation_dashboard",
    "interactive_artifact_explorer",
    "statistical_estimation_engine",
    "live_source_verification",
    "reference_software_adapters",
    "ci_and_certification_gates",
}
HTML_ONLY_BLOCKED_IDS = {
    "statistical_estimation_engine",
    "live_source_verification",
    "reference_software_adapters",
    "ci_and_certification_gates",
}


class HTMLDeliveryContractError(ValueError):
    """Raised when HTML delivery rules weaken validation behavior."""


@dataclass(frozen=True)
class HTMLCapability:
    """One function and whether HTML can deliver it safely."""

    id: str
    delivery_mode: str
    status: str
    summary: str
    required_backing_artifacts: tuple[str, ...]
    forbidden_replacements: tuple[str, ...]
    regression_risk: str
    certification_effect: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "HTMLCapability":
        required = {
            "id",
            "delivery_mode",
            "status",
            "summary",
            "required_backing_artifacts",
            "forbidden_replacements",
            "regression_risk",
            "certification_effect",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise HTMLDeliveryContractError(f"HTML capability missing required keys: {missing}")
        capability = cls(
            id=str(raw["id"]),
            delivery_mode=str(raw["delivery_mode"]),
            status=str(raw["status"]),
            summary=str(raw["summary"]),
            required_backing_artifacts=tuple(str(item) for item in raw["required_backing_artifacts"]),
            forbidden_replacements=tuple(str(item) for item in raw["forbidden_replacements"]),
            regression_risk=str(raw["regression_risk"]),
            certification_effect=str(raw["certification_effect"]),
        )
        capability.validate()
        return capability

    def validate(self) -> None:
        if self.id not in REQUIRED_CAPABILITY_IDS:
            raise HTMLDeliveryContractError(f"unsupported HTML capability id: {self.id}")
        if self.delivery_mode not in ALLOWED_DELIVERY_MODES:
            raise HTMLDeliveryContractError(f"{self.id}: unsupported delivery mode.")
        if self.status not in ALLOWED_STATUSES:
            raise HTMLDeliveryContractError(f"{self.id}: unsupported status.")
        if self.certification_effect != "none":
            raise HTMLDeliveryContractError(f"{self.id}: HTML delivery cannot certify model performance.")
        if not self.required_backing_artifacts:
            raise HTMLDeliveryContractError(f"{self.id}: required_backing_artifacts must not be empty.")
        if not self.forbidden_replacements:
            raise HTMLDeliveryContractError(f"{self.id}: forbidden_replacements must not be empty.")
        if self.id in HTML_ONLY_BLOCKED_IDS:
            if self.delivery_mode != "backend_required" or self.status != "blocked_for_html_only":
                raise HTMLDeliveryContractError(f"{self.id}: must remain backend_required.")
            if "high" not in self.regression_risk.lower():
                raise HTMLDeliveryContractError(f"{self.id}: regression risk must remain high.")
        else:
            if self.delivery_mode != "static_html_allowed" or self.status != "allowed":
                raise HTMLDeliveryContractError(f"{self.id}: read-only artifact views should remain HTML-allowed.")


@dataclass(frozen=True)
class HTMLDeliveryContract:
    """Complete HTML delivery contract."""

    checked_at: str
    certification_effect: str
    purpose: str
    global_rule: str
    capabilities: tuple[HTMLCapability, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "HTMLDeliveryContract":
        required = {
            "schema_version",
            "checked_at",
            "certification_effect",
            "purpose",
            "global_rule",
            "capabilities",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise HTMLDeliveryContractError(f"HTML delivery contract missing required keys: {missing}")
        if raw["schema_version"] != HTML_DELIVERY_CONTRACT_SCHEMA_VERSION:
            raise HTMLDeliveryContractError(
                f"schema_version must be {HTML_DELIVERY_CONTRACT_SCHEMA_VERSION}."
            )
        contract = cls(
            checked_at=str(raw["checked_at"]),
            certification_effect=str(raw["certification_effect"]),
            purpose=str(raw["purpose"]),
            global_rule=str(raw["global_rule"]),
            capabilities=tuple(HTMLCapability.from_mapping(item) for item in raw["capabilities"]),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if not self.checked_at.strip():
            raise HTMLDeliveryContractError("HTML delivery contract checked_at must not be empty.")
        if self.certification_effect != "none":
            raise HTMLDeliveryContractError("HTML delivery contract cannot certify model performance.")
        rule = self.global_rule.lower()
        if "html-only" not in rule or "cannot replace" not in rule:
            raise HTMLDeliveryContractError("global_rule must block HTML-only replacement of engines.")
        ids = [capability.id for capability in self.capabilities]
        missing = sorted(REQUIRED_CAPABILITY_IDS - set(ids))
        if missing:
            raise HTMLDeliveryContractError(f"HTML delivery contract missing capabilities: {missing}")
        duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
        if duplicates:
            raise HTMLDeliveryContractError(f"duplicate HTML capabilities: {duplicates}")


def load_html_delivery_contract(path: str | Path) -> HTMLDeliveryContract:
    """Load and validate the HTML delivery contract."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    return HTMLDeliveryContract.from_mapping(payload)


def summarize_html_delivery_contract(contract: HTMLDeliveryContract) -> dict[str, Any]:
    """Return compact validation-status fields."""

    mode_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for capability in contract.capabilities:
        mode_counts[capability.delivery_mode] = mode_counts.get(capability.delivery_mode, 0) + 1
        status_counts[capability.status] = status_counts.get(capability.status, 0) + 1
    return {
        "schema_version": HTML_DELIVERY_CONTRACT_SCHEMA_VERSION,
        "checked_at": contract.checked_at,
        "n_capabilities": len(contract.capabilities),
        "delivery_mode_counts": dict(sorted(mode_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "html_only_blocked_ids": sorted(HTML_ONLY_BLOCKED_IDS),
        "certification_effect": contract.certification_effect,
    }
