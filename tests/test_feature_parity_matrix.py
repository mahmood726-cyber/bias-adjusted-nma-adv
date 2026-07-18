import copy
from pathlib import Path

import pytest

from bias_nma_adv.feature_parity_matrix import (
    FEATURE_PARITY_MATRIX_SCHEMA_VERSION,
    REQUIRED_FEATURE_IDS,
    FeatureParityMatrix,
    FeatureParityMatrixError,
    load_feature_parity_matrix,
    summarize_feature_parity_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "validation" / "feature_parity_matrix.toml"


def test_feature_parity_matrix_keeps_broad_parity_incomplete():
    matrix = load_feature_parity_matrix(MATRIX)

    assert matrix.certification_effect == "none"
    assert matrix.global_feature_parity_complete is False
    assert {item.id for item in matrix.items} == REQUIRED_FEATURE_IDS

    by_id = {item.id: item for item in matrix.items}
    assert by_id["pairwise_metafor_meta"].status == "reference_candidate"
    assert "sglt2_hf_metafor_gosh_reference.toml" in "\n".join(
        by_id["pairwise_metafor_meta"].evidence_artifacts
    )
    assert "metafor_tau2_crosscheck_survival_reference.toml" in "\n".join(
        by_id["pairwise_metafor_meta"].evidence_artifacts
    )
    assert "psoriasis_sparse_binary_metafor_reference.toml" in "\n".join(
        by_id["pairwise_metafor_meta"].evidence_artifacts
    )
    assert "breast_adjuvant_idfs_prediction_interval_metafor_reference.toml" in "\n".join(
        by_id["pairwise_metafor_meta"].evidence_artifacts
    )
    assert "semaglutide_step_continuous_metafor_reference.toml" in "\n".join(
        by_id["pairwise_metafor_meta"].evidence_artifacts
    )
    assert "inclisiran_orion_continuous_metafor_reference.toml" in "\n".join(
        by_id["pairwise_metafor_meta"].evidence_artifacts
    )
    assert "sparse_binary_reference_cases" not in (
        by_id["pairwise_metafor_meta"].required_next_artifacts
    )
    assert "prediction_interval_reference_cases" not in (
        by_id["pairwise_metafor_meta"].required_next_artifacts
    )
    assert "continuous_outcome_reference_cases" not in (
        by_id["pairwise_metafor_meta"].required_next_artifacts
    )
    assert "continuous_edge_cases_zero_tau_and_positive_tau_across_more_domains" not in (
        by_id["pairwise_metafor_meta"].required_next_artifacts
    )
    assert "continuous_zero_tau_edge_case_across_additional_domain" in (
        by_id["pairwise_metafor_meta"].required_next_artifacts
    )
    assert "additional_continuous_domains_and_reference_versions" in (
        by_id["pairwise_metafor_meta"].required_next_artifacts
    )
    assert "psoriasis_pasi90_ctgov_binary_network_netmeta_reference.toml" in "\n".join(
        by_id["multiarm_netmeta_gls"].evidence_artifacts
    )
    assert by_id["stan_nuts_multinma_bayesian_nma"].status == "reference_candidate"
    assert by_id["node_splitting_inconsistency"].status == "reference_candidate"
    assert "psoriasis_pasi90_ctgov_binary_network_netsplit_reference.toml" in "\n".join(
        by_id["node_splitting_inconsistency"].evidence_artifacts
    )
    assert by_id["netheat_contribution_visualization"].status == "local_implemented"
    assert "psoriasis_pasi90_ctgov_binary_network_benchmark.toml" in "\n".join(
        by_id["netheat_contribution_visualization"].evidence_artifacts
    )
    assert by_id["dose_response_mbnmadose"].status == "reference_candidate"
    assert "mbnmadose_semaglutide_polynomial_reference.toml" in "\n".join(
        by_id["dose_response_mbnmadose"].evidence_artifacts
    )
    assert by_id["publication_bias_adjustments"].status == "reference_candidate"
    assert "publication_bias_t2d_ctgov_regtest_reference.toml" in "\n".join(
        by_id["publication_bias_adjustments"].evidence_artifacts
    )
    assert "publication_bias_glp1_metafor_trimfill_reference.toml" in "\n".join(
        by_id["publication_bias_adjustments"].evidence_artifacts
    )
    assert by_id["component_nma_netmeta"].status == "reference_candidate"
    assert by_id["cross_design_crossnma"].status == "local_implemented"
    assert "crossnma_sglt2_compatibility_preflight.toml" in "\n".join(
        by_id["cross_design_crossnma"].evidence_artifacts
    )
    assert "compatible_crossnma_reference_run" in (
        by_id["cross_design_crossnma"].required_next_artifacts
    )
    assert by_id["large_scale_validation"].status == "blocking"
    assert "stan_nuts_cmdstan_preflight.toml" in "\n".join(
        by_id["stan_nuts_multinma_bayesian_nma"].evidence_artifacts
    )
    assert "stan_nuts_cmdstan_output.json" in "\n".join(
        by_id["stan_nuts_multinma_bayesian_nma"].evidence_artifacts
    )

    summary = summarize_feature_parity_matrix(matrix)
    assert summary["schema_version"] == FEATURE_PARITY_MATRIX_SCHEMA_VERSION
    assert summary["global_feature_parity_complete"] is False
    assert summary["status_counts"] == {
        "blocking": 1,
        "local_implemented": 2,
        "planned": 1,
        "reference_candidate": 8,
    }
    assert summary["reference_matched_ids"] == []
    assert "stan_nuts_multinma_bayesian_nma" in summary["blocking_ids"]


def test_feature_parity_matrix_rejects_missing_required_feature_or_premature_completion():
    raw = _matrix_to_mapping(load_feature_parity_matrix(MATRIX))
    raw["features"] = [
        item for item in raw["features"] if item["id"] != "large_scale_validation"
    ]

    with pytest.raises(FeatureParityMatrixError, match="large_scale_validation"):
        FeatureParityMatrix.from_mapping(raw)

    raw = _matrix_to_mapping(load_feature_parity_matrix(MATRIX))
    raw["global_feature_parity_complete"] = True

    with pytest.raises(FeatureParityMatrixError, match="every item"):
        FeatureParityMatrix.from_mapping(raw)


def test_feature_parity_matrix_rejects_reference_match_without_evidence():
    raw = _matrix_to_mapping(load_feature_parity_matrix(MATRIX))
    raw["features"][0]["status"] = "reference_matched"
    raw["features"][0]["evidence_artifacts"] = []

    with pytest.raises(FeatureParityMatrixError, match="requires evidence"):
        FeatureParityMatrix.from_mapping(raw)


def _matrix_to_mapping(matrix: FeatureParityMatrix) -> dict[str, object]:
    return {
        "schema_version": FEATURE_PARITY_MATRIX_SCHEMA_VERSION,
        "checked_at": matrix.checked_at,
        "purpose": matrix.purpose,
        "certification_effect": matrix.certification_effect,
        "global_feature_parity_complete": matrix.global_feature_parity_complete,
        "source_boundary": matrix.source_boundary,
        "features": [
            {
                "id": item.id,
                "domain": item.domain,
                "reference_methods": list(item.reference_methods),
                "status": item.status,
                "evidence_artifacts": list(item.evidence_artifacts),
                "required_next_artifacts": list(item.required_next_artifacts),
                "claim_limit": item.claim_limit,
                "certification_effect": item.certification_effect,
            }
            for item in copy.deepcopy(matrix.items)
        ],
    }
