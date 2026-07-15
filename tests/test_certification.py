import copy
from pathlib import Path

import pytest

from bias_nma_adv.certification import (
    CertificationError,
    ReferenceTarget,
    assert_no_unsupported_production_claims,
    load_reference_targets,
    summarize_reference_targets,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "validation" / "reference_targets.toml"


def test_reference_targets_registry_is_valid():
    targets = load_reference_targets(TARGETS_PATH)

    assert len(targets) >= 8
    assert {target.id for target in targets} >= {
        "frequentist_nma_netmeta",
        "bayesian_nma_multinma_cmdstan",
        "mlnmr_multinma",
        "dose_response_mbnmadose",
        "cross_design_crossnma",
        "component_nma_netmeta_cnma",
        "certainty_cinema_robmen",
        "pairwise_metafor_meta",
    }

    summary = summarize_reference_targets(targets)
    assert summary == {"planned": len(targets)}
    assert_no_unsupported_production_claims(targets)


def test_reference_target_rejects_certification_without_evidence():
    raw = {
        "id": "bad_claim",
        "domain": "frequentist_nma",
        "module": "src/bias_nma_adv/model.py",
        "reference_method": "netmeta",
        "status": "reference_matched",
        "acceptance_criteria": ["match published examples"],
        "evidence_artifacts": [],
    }

    with pytest.raises(CertificationError, match="requires evidence_artifacts"):
        ReferenceTarget.from_mapping(raw)


def test_production_claim_requires_prerequisite_evidence_markers():
    raw = {
        "id": "premature_production",
        "domain": "frequentist_nma",
        "module": "src/bias_nma_adv/model.py",
        "reference_method": "netmeta",
        "status": "production_certified",
        "acceptance_criteria": ["match published examples"],
        "evidence_artifacts": ["reports/netmeta_reference_matched.json"],
    }
    target = ReferenceTarget.from_mapping(copy.deepcopy(raw))

    with pytest.raises(CertificationError, match="lacks evidence markers"):
        assert_no_unsupported_production_claims([target])

    raw["evidence_artifacts"] = [
        "reports/netmeta_reference_matched.json",
        "reports/netmeta_simulation_validated.json",
        "reports/netmeta_externally_reproduced.json",
    ]
    target = ReferenceTarget.from_mapping(raw)
    assert_no_unsupported_production_claims([target])
