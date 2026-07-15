import numpy as np
from bias_nma_adv.data import EvidenceDataset
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.bayesian import BayesianNMAMCMCSampler

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
    blocks = pooler._build_study_blocks(dataset, "O1", "binary")
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
    
    # Credible interval for trt_B
    ci = fit.credible_intervals["trt_B"]
    assert ci[0] < ci[1]
    
    # SD should be positive
    assert fit.posterior_sds["trt_B"] > 0.0
