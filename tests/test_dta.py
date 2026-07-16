import csv
from pathlib import Path

import pytest

from bias_nma_adv.dta import (
    DTA_MODEL_SCHEMA_VERSION,
    DTAError,
    DTAStudy,
    fit_bivariate_dta_reml,
    transform_dta_studies,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation" / "dta" / "dta_algorithmic_fixture.csv"


def test_dta_transform_applies_zero_cell_correction_only_when_needed():
    rows = [
        {"study_id": "zero_cell", "tp": 10, "fp": 0, "fn": 2, "tn": 30},
        {"study_id": "complete", "tp": 20, "fp": 4, "fn": 5, "tn": 40},
    ]

    transformed = transform_dta_studies(rows)

    assert transformed[0].continuity_correction == 0.5
    assert transformed[1].continuity_correction == 0.0
    assert transformed[0].specificity < 1.0
    assert transformed[1].sensitivity == pytest.approx(0.8)


def test_dta_fit_matches_mada_algorithmic_fixture_reference():
    rows = _fixture_rows()

    fit = fit_bivariate_dta_reml(rows)

    assert fit.schema_version == DTA_MODEL_SCHEMA_VERSION
    assert fit.method == "bivariate_logitnormal_reml_prototype"
    assert fit.n_studies == 8
    assert fit.converged is True
    assert fit.pooled_sensitivity == pytest.approx(0.7928540254492356, abs=1e-3)
    assert fit.pooled_specificity == pytest.approx(0.8517320362656781, abs=1e-3)
    assert fit.log_diagnostic_odds_ratio == pytest.approx(3.090466153506991, abs=2e-3)
    assert fit.tau2_sensitivity == pytest.approx(0.01802732174827957, abs=3e-3)
    assert fit.tau2_fpr == pytest.approx(0.02350613376869652, abs=3e-3)
    assert fit.rho_sensitivity_fpr == pytest.approx(1.0, abs=6e-3)
    assert fit.auc_trapezoid == pytest.approx(0.8885038267508946, abs=2e-3)
    assert any("not source-backed clinical evidence" in warning for warning in fit.warnings)


def test_dta_fit_fails_closed_for_too_few_studies():
    rows = _fixture_rows()[:4]

    with pytest.raises(DTAError, match="at least 5 studies"):
        fit_bivariate_dta_reml(rows)


def test_dta_study_rejects_invalid_counts_and_duplicate_ids():
    with pytest.raises(DTAError, match="non-diseased denominator"):
        DTAStudy.from_mapping({"study_id": "bad", "tp": 1, "fp": 0, "fn": 1, "tn": 0})

    rows = _fixture_rows()
    rows[1]["study_id"] = rows[0]["study_id"]
    with pytest.raises(DTAError, match="duplicate"):
        fit_bivariate_dta_reml(rows)


def _fixture_rows() -> list[dict[str, str]]:
    with FIXTURE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
