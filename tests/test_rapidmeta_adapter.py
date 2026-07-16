import json

import pytest

from bias_nma_adv.rapidmeta_adapter import (
    RAPIDMETA_APP_INDEX_SCHEMA_VERSION,
    RapidMetaAdapterError,
    evidence_dataset_from_rapidmeta_app_index,
)


def test_rapidmeta_app_index_imports_strict_binary_arm_level_contract(tmp_path):
    path = tmp_path / "app_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": RAPIDMETA_APP_INDEX_SCHEMA_VERSION,
                "analyses": [
                    {
                        "analysis_id": "mace",
                        "outcome_id": "primary_mace",
                        "measure_type": "binary",
                        "studies": [
                            {
                                "study_id": "NCT00000001",
                                "design": "rct",
                                "source_type": "clinicaltrials_gov",
                                "arms": [
                                    {
                                        "arm_id": "control",
                                        "treatment_id": "Placebo",
                                        "events": 20,
                                        "n": 100,
                                    },
                                    {
                                        "arm_id": "active",
                                        "treatment_id": "Drug",
                                        "events": 12,
                                        "n": 100,
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = evidence_dataset_from_rapidmeta_app_index(path)

    assert set(dataset.studies) == {"NCT00000001"}
    assert dataset.studies["NCT00000001"].source_type == "clinicaltrials_gov"
    assert len(dataset.arms) == 2
    assert len(dataset.outcomes_ad) == 2
    assert {outcome.source_type for outcome in dataset.outcomes_ad} == {
        "clinicaltrials_gov"
    }


def test_rapidmeta_app_index_requires_analysis_id_when_ambiguous(tmp_path):
    path = tmp_path / "app_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": RAPIDMETA_APP_INDEX_SCHEMA_VERSION,
                "analyses": [
                    {"analysis_id": "a", "outcome_id": "O1", "measure_type": "binary", "studies": []},
                    {"analysis_id": "b", "outcome_id": "O1", "measure_type": "binary", "studies": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RapidMetaAdapterError, match="analysis_id is required"):
        evidence_dataset_from_rapidmeta_app_index(path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload.update({"schema_version": "rapidmeta_app_index/v0"}),
            "schema_version",
        ),
        (
            lambda payload: payload["analyses"][0].update({"measure_type": "continuous"}),
            "only binary",
        ),
        (
            lambda payload: payload["analyses"][0]["studies"][0].update(
                {"source_type": "who_ictrp_protocol"}
            ),
            "cannot supply model-ready effects",
        ),
        (
            lambda payload: payload["analyses"][0]["studies"][0]["arms"][0].update(
                {"events": 101}
            ),
            "events cannot exceed",
        ),
    ],
)
def test_rapidmeta_app_index_fails_closed_on_invalid_payloads(tmp_path, mutator, message):
    payload = {
        "schema_version": RAPIDMETA_APP_INDEX_SCHEMA_VERSION,
        "analyses": [
            {
                "analysis_id": "mace",
                "outcome_id": "primary_mace",
                "measure_type": "binary",
                "studies": [
                    {
                        "study_id": "NCT00000001",
                        "design": "rct",
                        "source_type": "pubmed_abstract",
                        "arms": [
                            {
                                "arm_id": "control",
                                "treatment_id": "Placebo",
                                "events": 20,
                                "n": 100,
                            },
                            {
                                "arm_id": "active",
                                "treatment_id": "Drug",
                                "events": 12,
                                "n": 100,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    mutator(payload)
    path = tmp_path / "app_index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RapidMetaAdapterError, match=message):
        evidence_dataset_from_rapidmeta_app_index(path, analysis_id="mace")
