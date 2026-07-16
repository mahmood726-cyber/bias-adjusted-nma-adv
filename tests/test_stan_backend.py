from bias_nma_adv.stan_backend import (
    REQUIRED_STAN_DIAGNOSTICS,
    cmdstan_preflight,
    summarize_cmdstan_preflight,
)


def test_cmdstan_preflight_is_non_certifying_and_reports_required_diagnostics():
    report = cmdstan_preflight()

    assert report.certification_effect == "none"
    assert report.status in {"available", "unavailable", "failed"}
    assert set(REQUIRED_STAN_DIAGNOSTICS) == {
        "r_hat",
        "ess_bulk",
        "ess_tail",
        "divergent_transitions",
        "treedepth_saturation",
        "mcse",
        "prior_predictive_checks",
        "posterior_predictive_checks",
    }
    assert report.required_diagnostics == REQUIRED_STAN_DIAGNOSTICS
    if report.status != "available":
        assert report.missing_components
        assert "available" in report.message.lower() or "installed" in report.message.lower()


def test_cmdstan_preflight_summary_is_json_ready():
    report = cmdstan_preflight()
    summary = summarize_cmdstan_preflight(report)

    assert summary["status"] == report.status
    assert summary["certification_effect"] == "none"
    assert summary["required_diagnostics"] == list(REQUIRED_STAN_DIAGNOSTICS)
    assert isinstance(summary["missing_components"], list)
