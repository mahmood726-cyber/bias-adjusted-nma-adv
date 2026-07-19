from __future__ import annotations

import numpy as np
import pytest
from bias_nma_adv.copula import ClaytonCopulaJointEstimator

def test_clayton_copula():
    copula = ClaytonCopulaJointEstimator(theta=2.0)
    u = np.array([0.9, 0.8, 0.7])
    v = np.array([0.95, 0.9, 0.85])
    
    cdf = copula.joint_cdf(u, v)
    assert len(cdf) == 3
    assert np.all(cdf >= 0.0)
    assert np.all(cdf <= 1.0)
    
    density = copula.joint_density(u, v)
    assert np.all(density >= 0.0)
    
    # Check optimizer fitting
    theta = copula.fit(u, v)
    assert theta > 0.0


def test_registry_publication_bias():
    from bias_nma_adv.publication_bias import RegistryPublicationBiasAuditor
    auditor = RegistryPublicationBiasAuditor()
    
    # 1. Register trials
    auditor.register_trial_protocol("NCT123", "mace", "mace", "completed", ["DrugX"]) # No switching, published
    auditor.register_trial_protocol("NCT456", "mortality", "mace", "completed", ["DrugX"]) # Outcome switching!
    auditor.register_trial_protocol("NCT789", "mace", "mace", "completed", ["DrugX"]) # Unpublished trial
    
    # 2. Check outcome switching
    switching_scores = auditor.audit_outcome_switching(["NCT123", "NCT456", "NCT999"])
    assert switching_scores["NCT123"] == 0.0
    assert switching_scores["NCT456"] == 1.0
    assert switching_scores["NCT999"] == 1.0 # High risk (no registry protocol)
    
    # 3. Check unpublished ratio
    utr = auditor.calculate_unpublished_ratio("DrugX", ["NCT123", "NCT456"])
    # 1 out of 3 trials is unpublished (NCT789)
    assert np.isclose(utr, 1.0 / 3.0)
    
    # 4. Legacy automatic shrinkage is quarantined; use explicit sensitivity instead.
    pooled_effect = -0.30 # log-HR
    import pytest
    with pytest.raises(NotImplementedError, match="quarantined"):
        auditor.apply_bias_shrinkage(pooled_effect, utr)

def test_symbolic_regression():
    from bias_nma_adv.symbolic import SymbolicHazardRegressor
    reg = SymbolicHazardRegressor()
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # true hazard: h(t) = 0.5 * t + 0.2 * log(t)
    hazards = 0.5 * times + 0.2 * np.log(times)
    
    formula, coefs, mse = reg.fit_best_formula(times, hazards)
    assert formula != "None"
    assert len(coefs) == 2
    assert mse < 1.0

def test_conditional_gan():
    from bias_nma_adv.gan import ConditionalGAN
    gan = ConditionalGAN(noise_dim=2, cond_dim=1, out_dim=2)
    
    cond = np.array([[1.0], [0.0], [1.0], [0.0]])
    real_x = np.random.normal(0.0, 1.0, size=(4, 2))
    
    # Check shape of generation
    fake = gan.generate(cond)
    assert fake.shape == (4, 2)
    
    # Train one step and check loss
    loss = gan.train_step(real_x, cond, lr=0.01)
    assert isinstance(loss, float)

def test_bayesian_model_averaging():
    from bias_nma_adv.bma import BayesianModelAverager
    bma = BayesianModelAverager()
    
    effects = np.array([-0.30, -0.25, -0.15])
    variances = np.array([0.01, 0.012, 0.015])
    bics = np.array([120.0, 122.0, 128.0]) # Model 1 is strongly preferred
    
    probs = bma.calculate_posterior_probabilities(bics)
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]
    
    avg_eff, avg_var = bma.average_effects(effects, variances, bics)
    assert avg_eff < 0.0
    assert avg_var > 0.0

def test_multinomial_glmm():
    from bias_nma_adv.multinomial import MultinomialGLMMSolver
    solver = MultinomialGLMMSolver()
    
    x = np.array([[1.0, 0.5], [1.0, -0.5], [1.0, 1.2], [1.0, -1.2]])
    # 3 classes: one-hot encoded
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    
    solver.fit(x, y)
    probs = solver.predict_proba(x)
    assert probs.shape == (4, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_collaborative_tmle():
    from bias_nma_adv.ctmle import CollaborativeTMLE
    ctmle = CollaborativeTMLE()
    
    rng = np.random.default_rng(42)
    n = 150
    w = rng.normal(0.0, 1.0, size=(n, 2))
    ps = 1.0 / (1.0 + np.exp(-w[:, 1]))
    a = rng.binomial(1, ps)
    y_prob = 1.0 / (1.0 + np.exp(-(0.5 * a + w[:, 0])))
    y = rng.binomial(1, y_prob)
    
    rd = ctmle.estimate_risk_difference(w, a, y)
    assert -1.0 <= rd <= 1.0

def test_no_u_turn_sampler():
    from bias_nma_adv.nuts import NoUTurnSampler
    sampler = NoUTurnSampler(step_size=0.1, seed=42)
    
    def log_post(theta):
        return -0.5 * theta[0]**2
        
    def grad_log_post(theta):
        return -theta
        
    initial_pos = np.array([1.0])
    samples = sampler.sample(initial_pos, log_post, grad_log_post, n_samples=15)
    assert samples.shape == (15, 1)
    assert np.mean(np.abs(samples)) < 2.0

def test_registry_sponsor_auditor():
    from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor
    import pytest
    auditor = RegistrySponsorAuditor()

    with pytest.raises(ValueError, match="randomized"):
        auditor.register_trial_flow("NCT_BAD_N", "other", randomized=0, lost_to_follow_up=0)
    with pytest.raises(ValueError, match="lost_to_follow_up"):
        auditor.register_trial_flow("NCT_BAD_LTFU", "other", randomized=100, lost_to_follow_up=101)
    
    # 1. Register trials
    # NIH funded, no attrition
    auditor.register_trial_flow("NCT111", "nih", randomized=1000, lost_to_follow_up=10)
    # Industry funded, high attrition (10%)
    auditor.register_trial_flow("NCT222", "industry", randomized=1000, lost_to_follow_up=100)
    
    # 2. Check attrition ratio
    lar1 = auditor.calculate_attrition_ratio("NCT111")
    lar2 = auditor.calculate_attrition_ratio("NCT222")
    lar_missing = auditor.calculate_attrition_ratio("NCT_MISSING")
    assert np.isclose(lar1, 0.01)
    assert np.isclose(lar2, 0.10)
    assert np.isclose(lar_missing, 1.0)
    
    # 3. Check sponsor bias score
    sbs1 = auditor.calculate_sponsorship_bias_score("NCT111")
    sbs2 = auditor.calculate_sponsorship_bias_score("NCT222")
    assert sbs1 == 0.0
    assert sbs2 == 1.0
    
    # 4. Check quality weight adjustment
    q1 = auditor.adjust_doi_welton_quality("NCT111", 1.0)
    q2 = auditor.adjust_doi_welton_quality("NCT222", 1.0)
    q_missing = auditor.adjust_doi_welton_quality("NCT_MISSING", 1.0)
    
    # NCT111 should remain close to 1.0 (no penalty)
    assert np.isclose(q1, 1.0)
    # NCT222 should be heavily penalized (industry + attrition)
    assert q2 < 0.80
    # Missing metadata must not be treated as perfect follow-up.
    assert np.isclose(q_missing, 0.56)




def test_sponsor_class_fails_closed_on_unrecognised_class():
    """An ambiguous registry class must not score as clean provenance.

    Previously sponsor_class was matched by exact equality against "industry",
    so AACT's OTHER/NETWORK/UNKNOWN/AMBIG, a typo, or "" all fell through to
    0.0 -- the BEST score. That made a trial registered with untidy metadata
    score better than one never registered at all (which already scored 1.0).
    """
    from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor, SponsorClassError

    auditor = RegistrySponsorAuditor()
    auditor.register_trial_flow("NCT_IND", "industry", randomized=100, lost_to_follow_up=0)
    auditor.register_trial_flow("NCT_NIH", "nih", randomized=100, lost_to_follow_up=0)
    for nct, raw in (
        ("NCT_OTHER", "other"),
        ("NCT_UNKNOWN", "unknown"),
        ("NCT_AMBIG", "ambig"),
        ("NCT_NETWORK", "network"),
        ("NCT_EMPTY", ""),
        ("NCT_TYPO", "industy"),
    ):
        auditor.register_trial_flow(nct, raw, randomized=100, lost_to_follow_up=0)

    assert auditor.sponsor_class_status("NCT_IND") == "industry"
    assert auditor.sponsor_class_status("NCT_NIH") == "non_industry"
    assert auditor.sponsor_class_status("NCT_MISSING") == "unregistered"

    # Only an affirmative non-industry class earns the clean score.
    assert auditor.calculate_sponsorship_bias_score("NCT_NIH") == 0.0
    assert auditor.calculate_sponsorship_bias_score("NCT_IND") == 1.0

    for nct in ("NCT_OTHER", "NCT_UNKNOWN", "NCT_AMBIG", "NCT_NETWORK", "NCT_EMPTY", "NCT_TYPO"):
        assert auditor.sponsor_class_status(nct) == "unrecognised", nct
        # THE FIX: these used to be 0.0.
        assert auditor.calculate_sponsorship_bias_score(nct) == 1.0, nct
        # ...and the substitution is visible, not silent.
        assert nct in auditor.unrecognised_sponsor_classes, nct

    # The perverse incentive is gone: garbage metadata is no longer better than
    # no registration at all.
    assert auditor.calculate_sponsorship_bias_score("NCT_OTHER") == (
        auditor.calculate_sponsorship_bias_score("NCT_MISSING")
    )


def test_sponsor_class_strict_mode_raises_on_unrecognised_class():
    from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor, SponsorClassError

    strict = RegistrySponsorAuditor(strict=True)
    strict.register_trial_flow("NCT_OK", "nih", randomized=100, lost_to_follow_up=0)
    with pytest.raises(SponsorClassError, match="not a recognised class"):
        strict.register_trial_flow("NCT_BAD", "other", randomized=100, lost_to_follow_up=0)


def test_unrecognised_sponsor_class_downweights_quality():
    """The fail-closed score must actually reach the quality adjustment."""
    from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor

    auditor = RegistrySponsorAuditor()
    auditor.register_trial_flow("NCT_NIH", "nih", randomized=1000, lost_to_follow_up=0)
    auditor.register_trial_flow("NCT_OTHER", "other", randomized=1000, lost_to_follow_up=0)

    clean = auditor.adjust_doi_welton_quality("NCT_NIH", 1.0)
    ambiguous = auditor.adjust_doi_welton_quality("NCT_OTHER", 1.0)

    assert clean == pytest.approx(1.0)
    # Used to be equal to `clean`; ambiguous provenance now costs the same 20%
    # as declared industry funding.
    assert ambiguous == pytest.approx(0.80)
    assert ambiguous < clean
