import copy
import json
from pathlib import Path

import pytest

from bias_nma_adv.stan_reference_validation import (
    StanReferenceValidationError,
    load_stan_reference_output,
    validate_stan_nuts_reference_output,
)


ROOT = Path(__file__).resolve().parents[1]
STAN_OUTPUT = ROOT / "validation" / "reference_runs" / "stan_nuts_cmdstan_output.json"


def test_stan_nuts_reference_output_matches_source_backed_artifact():
    summary = validate_stan_nuts_reference_output(STAN_OUTPUT, repo_root=ROOT)

    assert summary["schema_version"] == "stan_reference_validation/v1"
    assert summary["target_id"] == "bayesian_nma_multinma_cmdstan"
    assert summary["status"] == "passed"
    assert summary["certification_effect"] == "evidence_candidate"
    assert summary["max_abs_difference"] < 0.001
    assert summary["r_hat"] <= 1.01
    assert summary["ess_bulk"] >= 400
    assert summary["ess_tail"] >= 400
    assert summary["mcse_mean"] <= 0.005
    assert "not broad feature parity" in summary["claim_limit"]


def test_stan_nuts_reference_rejects_diagnostic_drift(tmp_path):
    payload = copy.deepcopy(load_stan_reference_output(STAN_OUTPUT))
    payload["diagnostics"]["divergent_transitions"] = 1
    mutated = tmp_path / "stan_divergence.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StanReferenceValidationError, match="divergent"):
        validate_stan_nuts_reference_output(mutated, repo_root=ROOT)


def test_stan_nuts_reference_rejects_numeric_drift(tmp_path):
    payload = copy.deepcopy(load_stan_reference_output(STAN_OUTPUT))
    payload["posterior"]["mean"] = 0.0
    payload["reference_comparison"]["posterior_mean"] = 0.0
    payload["reference_comparison"]["absolute_difference"] = abs(
        payload["reference_comparison"]["reference_estimate"]
    )
    mutated = tmp_path / "stan_numeric_drift.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StanReferenceValidationError, match="exceeding tolerance"):
        validate_stan_nuts_reference_output(mutated, repo_root=ROOT)
