import pytest

from bias_nma_adv.component_nma import (
    COMPONENT_NMA_SCHEMA_VERSION,
    ComponentNMAError,
    fit_additive_component_nma,
)


FIXTURE = [
    {"study_id": "S1", "treat1": "A", "treat2": "Placebo", "estimate": 0.20, "se": 0.20},
    {"study_id": "S2", "treat1": "B", "treat2": "Placebo", "estimate": 0.30, "se": 0.25},
    {"study_id": "S3", "treat1": "A + B", "treat2": "Placebo", "estimate": 0.55, "se": 0.22},
    {"study_id": "S4", "treat1": "A + B", "treat2": "A", "estimate": 0.35, "se": 0.21},
    {"study_id": "S5", "treat1": "A + B", "treat2": "B", "estimate": 0.25, "se": 0.23},
]


def test_additive_component_nma_matches_hand_wls_fixture():
    fit = fit_additive_component_nma(FIXTURE)

    assert fit.schema_version == COMPONENT_NMA_SCHEMA_VERSION
    assert fit.components == ("A", "B")
    assert fit.rank == 2
    assert fit.df == 3
    assert fit.q == pytest.approx(0.05037015054702015, abs=1e-12)
    assert fit.warnings == ()

    components = {item.name: item for item in fit.component_effects}
    assert components["A"].estimate == pytest.approx(0.22133047331351394, abs=1e-12)
    assert components["A"].se == pytest.approx(0.13202483126466402, abs=1e-12)
    assert components["A"].estimable is True
    assert components["B"].estimate == pytest.approx(0.3290903740456773, abs=1e-12)
    assert components["B"].se == pytest.approx(0.13771807671957265, abs=1e-12)
    assert components["B"].estimable is True

    treatments = {item.name: item for item in fit.treatment_effects}
    assert treatments["Placebo"].estimate == pytest.approx(0.0, abs=1e-12)
    assert treatments["Placebo"].se == pytest.approx(0.0, abs=1e-12)
    assert treatments["A + B"].estimate == pytest.approx(
        components["A"].estimate + components["B"].estimate,
        abs=1e-12,
    )
    assert treatments["A + B"].se == pytest.approx(0.15575029947390334, abs=1e-12)
    assert treatments["A + B"].estimable is True


def test_additive_component_nma_flags_rank_deficiency():
    fit = fit_additive_component_nma(
        [
            {
                "study_id": "S1",
                "treat1": "A + B",
                "treat2": "Placebo",
                "estimate": 0.50,
                "se": 0.20,
            },
            {
                "study_id": "S2",
                "treat1": "A + B",
                "treat2": "Placebo",
                "estimate": 0.60,
                "se": 0.25,
            },
        ]
    )

    assert fit.rank == 1
    assert any("rank deficient" in warning for warning in fit.warnings)
    assert {item.name: item.estimable for item in fit.component_effects} == {
        "A": False,
        "B": False,
    }
    assert {item.name: item.estimable for item in fit.treatment_effects}["A + B"] is True


def test_additive_component_nma_fails_closed_for_bad_rows():
    with pytest.raises(ComponentNMAError, match="se must be positive"):
        fit_additive_component_nma(
            [
                {
                    "study_id": "bad",
                    "treat1": "A",
                    "treat2": "Placebo",
                    "estimate": 0.1,
                    "se": 0,
                }
            ]
        )
