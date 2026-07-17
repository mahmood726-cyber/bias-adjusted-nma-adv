import numpy as np
import pytest
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.bayesian import (
    BayesianNMAMCMCSampler,
    compute_mcmc_diagnostics,
    posterior_predictive_check,
    posterior_treatment_ranking,
    prior_predictive_check,
)

def test_bayesian_mcmc_sampler():
    dataset = EvidenceDataset()
    
    # Study 1 (RCT): A vs B
    dataset.add_study("S1", "rct")
    dataset.add_arm("S1", "arm1", "A", 100)
    dataset.add_arm("S1", "arm2", "B", 100)
    dataset.add_outcome_ad("S1", "arm1", "O1", "binary", 10)
    dataset.add_outcome_ad("S1", "arm2", "O1", "binary", 20)

    # Study 2 (RCT): A vs B
    dataset.add_study("S2", "rct")
    dataset.add_arm("S2", "arm1", "A", 100)
    dataset.add_arm("S2", "arm2", "B", 100)
    dataset.add_outcome_ad("S2", "arm1", "O1", "binary", 12)
    dataset.add_outcome_ad("S2", "arm2", "O1", "binary", 22)

    pooler = AdvancedBiasAdjustedNMAPooler(random_effects=True)
    
    # Run the setup steps manually or extract them to get vectors
    blocks, n_studies_dropped = pooler._build_study_blocks(dataset, "O1", "binary")
    assert n_studies_dropped == 0
    unique_designs = ["rct"]
    design_to_idx = {"rct": 0}
    param_names = pooler._build_parameter_names(("B",), (), [], False, ())
    
    y, x, v = pooler._assemble_design(
        blocks, param_names, reference_treatment="A", reference_design="rct", cov_names=[], study_specific_bias=False
    )
    
    sampler = BayesianNMAMCMCSampler(n_samples=500, burn_in=100, thinning=1)
    fit = sampler.fit(
        y=y,
        x=x,
        v=v,
        param_names=param_names,
        blocks=blocks,
        unique_designs=unique_designs,
        design_to_idx=design_to_idx,
        bias_prior_sd=1.0,
        treatment_shrinkage_lambda=0.0,
        treatment_centralities={"B": 1.0, "A": 1.0}
    )
    
    # Assertions
    assert "trt_B" in fit.posterior_means
    assert "tau_rct" in fit.posterior_means
    assert fit.chains.shape == (500, 2)
    assert fit.acceptance_rate >= 0.0
    assert set(fit.diagnostics) == {"trt_B", "tau_rct"}
    assert any("R-hat unavailable" in warning for warning in fit.diagnostic_warnings)
    assert fit.diagnostics["trt_B"].n_chains == 1
    assert fit.diagnostics["trt_B"].r_hat is None
    assert fit.diagnostics["trt_B"].ess_bulk > 0.0
    assert fit.diagnostics["trt_B"].mcse_mean >= 0.0
    
    # Credible interval for trt_B
    ci = fit.credible_intervals["trt_B"]
    assert ci[0] < ci[1]
    
    # SD should be positive
    assert fit.posterior_sds["trt_B"] > 0.0

    posterior_check = posterior_predictive_check(
        fit.chains,
        fit.parameter_names,
        y,
        x,
        v,
        seed=101,
    )
    assert posterior_check.check_type == "posterior_predictive"
    assert posterior_check.n_draws == 500
    assert posterior_check.n_observations == len(y)
    assert posterior_check.observed_mean == pytest.approx(float(np.mean(y)))
    assert posterior_check.predictive_interval[0] < posterior_check.predictive_interval[1]
    assert 0.0 <= posterior_check.bayesian_p_value <= 1.0
    assert any("not Stan" in warning for warning in posterior_check.warnings)

    ranking = posterior_treatment_ranking(
        fit.chains,
        fit.parameter_names,
        reference_treatment="A",
        beneficial_direction="lower",
    )
    assert ranking.treatments == ("A", "B")
    assert ranking.n_draws == 500
    assert set(ranking.rank_probabilities) == {"A", "B"}
    assert all(
        sum(probabilities) == pytest.approx(1.0, abs=1e-12)
        for probabilities in ranking.rank_probabilities.values()
    )
    assert all(1.0 <= rank <= 2.0 for rank in ranking.mean_ranks.values())
    assert any("not SUCRA" in warning for warning in ranking.warnings)


def test_compute_mcmc_diagnostics_exports_rhat_for_multiple_chains():
    rng = np.random.default_rng(123)
    draws = rng.normal(0.0, 1.0, size=(2, 500, 1))

    diagnostics = compute_mcmc_diagnostics(draws, ("theta",))
    diagnostic = diagnostics["theta"]

    assert diagnostic.n_chains == 2
    assert diagnostic.n_draws == 500
    assert diagnostic.r_hat is not None
    assert diagnostic.r_hat < 1.05
    assert diagnostic.ess_bulk > 0.0
    assert diagnostic.ess_tail > 0.0
    assert diagnostic.mcse_mean >= 0.0


def test_prior_predictive_check_simulates_from_declared_prior_scale():
    x = np.array([[1.0], [0.5], [-0.5]])
    v = np.diag([0.04, 0.05, 0.06])

    check = prior_predictive_check(x, v, n_draws=100, beta_sd=0.5, seed=7)

    assert check.check_type == "prior_predictive"
    assert check.n_draws == 100
    assert check.n_observations == 3
    assert check.observed_mean is None
    assert check.predictive_interval[0] < check.predictive_interval[1]
    assert check.bayesian_p_value is None
    assert any("not Stan" in warning for warning in check.warnings)


def test_bayesian_predictive_and_ranking_helpers_reject_invalid_inputs():
    draws = np.ones((3, 1))
    with pytest.raises(ValueError, match="at least four draws"):
        posterior_treatment_ranking(
            draws,
            ("trt_B",),
            reference_treatment="A",
        )

    with pytest.raises(ValueError, match="2D design matrix"):
        prior_predictive_check(np.array([1.0, 2.0]), np.eye(2))

    with pytest.raises(ValueError, match="parameter_names length"):
        posterior_predictive_check(
            np.ones((4, 2)),
            ("trt_B",),
            np.array([0.0]),
            np.ones((1, 1)),
            np.eye(1),
        )
