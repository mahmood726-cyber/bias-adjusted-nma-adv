import copy
from pathlib import Path

import pytest

from bias_nma_adv.certification import (
    CertificationError,
    ReferenceRunReport,
    ReferenceTarget,
    assert_no_unsupported_production_claims,
    assert_reference_runs_target_known,
    certification_candidate_artifacts,
    load_reference_run_reports,
    load_reference_targets,
    summarize_reference_run_reports,
    summarize_reference_targets,
)
from bias_nma_adv.real_meta import sha256_file


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "validation" / "reference_targets.toml"
REFERENCE_RUNS_PATH = ROOT / "validation" / "reference_runs"
PAIRWISE_R_ADAPTER = ROOT / "external" / "r" / "pairwise_metafor_meta.R"
PAIRWISE_PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_reference_adapters.py"
MULTIARM_R_ADAPTER = ROOT / "external" / "r" / "multiarm_netmeta_fixture.R"
MULTIARM_PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_multiarm_netmeta_adapter.py"
DTA_R_ADAPTER = ROOT / "external" / "r" / "dta_mada_reitsma_fixture.R"
DTA_SOURCE_R_ADAPTER = ROOT / "external" / "r" / "dta_mada_reitsma_source_table.R"
DTA_PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_dta_mada_adapter.py"
DOSE_RESPONSE_R_ADAPTER = ROOT / "external" / "r" / "dose_response_metafor_polynomial.R"
SURVIVAL_HR_R_ADAPTER = ROOT / "external" / "r" / "survival_hr_metafor_pairwise.R"
CTGOV_HR_NETWORK_R_ADAPTER = ROOT / "external" / "r" / "ctgov_hr_network_netmeta.R"
COMPONENT_CNMA_R_ADAPTER = ROOT / "external" / "r" / "component_netmeta_cnma_fixture.R"
STAN_MODEL = ROOT / "external" / "stan" / "standard_binary_nma.stan"
STAN_PREFLIGHT_SCRIPT = ROOT / "scripts" / "preflight_stan_nuts_adapter.py"
STAN_REFERENCE_SCRIPT = ROOT / "scripts" / "run_stan_nuts_reference.py"

def test_reference_targets_registry_is_valid():
    targets = load_reference_targets(TARGETS_PATH)

    assert len(targets) >= 8
    assert {target.id for target in targets} >= {
        "frequentist_nma_netmeta",
        "bayesian_nma_multinma_cmdstan",
        "mlnmr_multinma",
        "dose_response_mbnmadose",
        "dose_response_metafor_polynomial_smoke",
        "cross_design_crossnma",
        "component_nma_netmeta_cnma",
        "certainty_cinema_robmen",
        "dta_bivariate_hsroc_reference",
        "dta_source_table_mada_reitsma_smoke",
        "pairwise_metafor_meta",
        "reported_hr_survival_metafor_pairwise",
        "ctgov_hr_network_netmeta_star",
        "ctgov_binary_network_netmeta_closed_loop",
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


def test_reference_run_reports_are_fail_closed_and_targeted():
    targets = load_reference_targets(TARGETS_PATH)
    reports = load_reference_run_reports(REFERENCE_RUNS_PATH)

    assert_reference_runs_target_known(targets, reports)
    summary = summarize_reference_run_reports(reports)
    assert summary == {"failed": 4, "passed": 11}

    by_adapter = {report.adapter_id: report for report in reports}
    assert set(by_adapter) == {
        "r_metafor_meta_pairwise_preflight",
        "r_metafor_meta_pairwise_output_validation",
        "r_netmeta_multiarm_preflight",
        "r_netmeta_multiarm_output_validation",
        "r_mada_dta_reitsma_preflight",
        "r_mada_dta_reitsma_output_validation",
        "r_mada_dta_midkine_source_output_validation",
        "r_metafor_dose_response_polynomial_output_validation",
        "r_metafor_sglt2_survival_hr_output_validation",
        "r_metafor_pcsk9_survival_hr_output_validation",
        "r_netmeta_t2d_ctgov_hr_network_output_validation",
        "r_netmeta_psoriasis_ctgov_binary_network_output_validation",
        "r_netmeta_component_cnma_output_validation",
        "python_cmdstan_nuts_preflight",
        "python_cmdstan_nuts_output_validation",
    }

    report = by_adapter["r_metafor_meta_pairwise_preflight"]
    assert report.adapter_id == "r_metafor_meta_pairwise_preflight"
    assert report.certification_effect == "none"
    assert report.is_certification_evidence_candidate is False
    assert "Rscript" in report.command
    assert "external/r/pairwise_metafor_meta.R" in report.command
    if report.executable_found is False:
        assert "Rscript is not available" in report.skip_reason
    assert PAIRWISE_R_ADAPTER.is_file()
    assert PAIRWISE_PREFLIGHT_SCRIPT.is_file()

    report = by_adapter["r_netmeta_multiarm_preflight"]
    assert report.adapter_id == "r_netmeta_multiarm_preflight"
    assert report.reference_method == "netmeta"
    assert report.certification_effect == "none"
    assert report.is_certification_evidence_candidate is False
    assert "Rscript" in report.command
    assert "external/r/multiarm_netmeta_fixture.R" in report.command
    if report.executable_found is False:
        assert "Rscript is not available" in report.skip_reason
    assert MULTIARM_R_ADAPTER.is_file()
    assert MULTIARM_PREFLIGHT_SCRIPT.is_file()

    report = by_adapter["r_mada_dta_reitsma_preflight"]
    assert report.adapter_id == "r_mada_dta_reitsma_preflight"
    assert report.reference_method == "mada::reitsma"
    assert report.certification_effect == "none"
    assert report.is_certification_evidence_candidate is False
    assert "Rscript" in report.command
    assert "external/r/dta_mada_reitsma_fixture.R" in report.command
    if report.executable_found is False:
        assert "Rscript is not available" in report.skip_reason
    assert DTA_R_ADAPTER.is_file()
    assert DTA_PREFLIGHT_SCRIPT.is_file()

    for report in (
        by_adapter["r_metafor_meta_pairwise_preflight"],
        by_adapter["r_netmeta_multiarm_preflight"],
        by_adapter["r_mada_dta_reitsma_preflight"],
        by_adapter["python_cmdstan_nuts_preflight"],
    ):
        assert report.output_artifacts == ()
        assert report.output_sha256 == {}
        assert report.tolerance == ""

    stan_preflight = by_adapter["python_cmdstan_nuts_preflight"]
    assert stan_preflight.target_id == "bayesian_nma_multinma_cmdstan"
    assert stan_preflight.status == "failed"
    assert stan_preflight.certification_effect == "none"
    assert "CmdStan" in stan_preflight.reference_method
    assert "does not execute NUTS" in stan_preflight.skip_reason
    assert STAN_MODEL.is_file()
    assert STAN_PREFLIGHT_SCRIPT.is_file()

    stan_reference = by_adapter["python_cmdstan_nuts_output_validation"]
    assert stan_reference.target_id == "bayesian_nma_multinma_cmdstan"
    assert stan_reference.status == "passed"
    assert stan_reference.certification_effect == "evidence_candidate"
    assert "CmdStan" in stan_reference.reference_method
    assert stan_reference.output_artifacts == (
        "validation/reference_runs/stan_nuts_cmdstan_output.json",
    )
    assert "R-hat <= 1.01" in stan_reference.tolerance
    assert "validation/real_meta/sglt2_hf_primary_events.csv" in stan_reference.input_artifacts
    assert STAN_REFERENCE_SCRIPT.is_file()

    pairwise_reference = by_adapter["r_metafor_meta_pairwise_output_validation"]
    assert pairwise_reference.target_id == "pairwise_metafor_meta"
    assert pairwise_reference.status == "passed"
    assert pairwise_reference.certification_effect == "evidence_candidate"
    assert pairwise_reference.output_artifacts == (
        "validation/reference_runs/pairwise_metafor_meta_output.json",
    )
    assert pairwise_reference.tolerance == "absolute <= 1e-06 for validated components"

    multiarm_reference = by_adapter["r_netmeta_multiarm_output_validation"]
    assert multiarm_reference.target_id == "multiarm_gls_netmeta_portfolio_fixture"
    assert multiarm_reference.status == "passed"
    assert multiarm_reference.certification_effect == "evidence_candidate"
    assert multiarm_reference.output_artifacts == (
        "validation/reference_runs/multiarm_netmeta_output.json",
    )
    assert multiarm_reference.tolerance == "absolute <= 1e-06 for validated components"

    dta_reference = by_adapter["r_mada_dta_reitsma_output_validation"]
    assert dta_reference.target_id == "dta_bivariate_hsroc_reference"
    assert dta_reference.status == "passed"
    assert dta_reference.certification_effect == "evidence_candidate"
    assert dta_reference.output_artifacts == (
        "validation/reference_runs/dta_mada_reitsma_output.json",
    )
    assert "probability <=" in dta_reference.tolerance
    assert "validation/dta/dta_algorithmic_fixture.toml" in dta_reference.input_artifacts

    dta_source_reference = by_adapter["r_mada_dta_midkine_source_output_validation"]
    assert dta_source_reference.target_id == "dta_source_table_mada_reitsma_smoke"
    assert dta_source_reference.status == "passed"
    assert dta_source_reference.certification_effect == "evidence_candidate"
    assert dta_source_reference.reference_method == "mada::reitsma source-backed DTA table"
    assert dta_source_reference.output_artifacts == (
        "validation/reference_runs/dta_mada_reitsma_midkine_source_output.json",
    )
    assert "validation/dta/midkine_elisa_cancer_dta_benchmark.toml" in (
        dta_source_reference.input_artifacts
    )
    assert "AUC exported only" in dta_source_reference.tolerance
    assert DTA_SOURCE_R_ADAPTER.is_file()

    dose_response_reference = by_adapter["r_metafor_dose_response_polynomial_output_validation"]
    assert dose_response_reference.target_id == "dose_response_metafor_polynomial_smoke"
    assert dose_response_reference.status == "passed"
    assert dose_response_reference.certification_effect == "evidence_candidate"
    assert dose_response_reference.reference_method == (
        "metafor fixed-effect polynomial meta-regression"
    )
    assert dose_response_reference.output_artifacts == (
        "validation/reference_runs/dose_response_metafor_polynomial_output.json",
    )
    assert dose_response_reference.tolerance == "absolute <= 1e-06 for validated components"
    assert (
        "validation/dose_response/semaglutide_obesity_dose_response_benchmark.toml"
        in dose_response_reference.input_artifacts
    )
    assert DOSE_RESPONSE_R_ADAPTER.is_file()

    sglt2_survival_reference = by_adapter["r_metafor_sglt2_survival_hr_output_validation"]
    assert sglt2_survival_reference.target_id == "reported_hr_survival_metafor_pairwise"
    assert sglt2_survival_reference.status == "passed"
    assert sglt2_survival_reference.certification_effect == "evidence_candidate"
    assert sglt2_survival_reference.output_artifacts == (
        "validation/reference_runs/sglt2_survival_hr_metafor_output.json",
    )
    assert "validation/survival/sglt2_hf_reported_hr_benchmark.toml" in (
        sglt2_survival_reference.input_artifacts
    )

    pcsk9_survival_reference = by_adapter["r_metafor_pcsk9_survival_hr_output_validation"]
    assert pcsk9_survival_reference.target_id == "reported_hr_survival_metafor_pairwise"
    assert pcsk9_survival_reference.status == "passed"
    assert pcsk9_survival_reference.certification_effect == "evidence_candidate"
    assert pcsk9_survival_reference.output_artifacts == (
        "validation/reference_runs/pcsk9_survival_hr_metafor_output.json",
    )
    assert "validation/survival/pcsk9_mace_reported_hr_benchmark.toml" in (
        pcsk9_survival_reference.input_artifacts
    )
    assert SURVIVAL_HR_R_ADAPTER.is_file()

    ctgov_hr_network_reference = by_adapter["r_netmeta_t2d_ctgov_hr_network_output_validation"]
    assert ctgov_hr_network_reference.target_id == "ctgov_hr_network_netmeta_star"
    assert ctgov_hr_network_reference.status == "passed"
    assert ctgov_hr_network_reference.certification_effect == "evidence_candidate"
    assert ctgov_hr_network_reference.reference_method == (
        "netmeta fixed-effect CT.gov reported-HR star network"
    )
    assert ctgov_hr_network_reference.output_artifacts == (
        "validation/reference_runs/t2d_ctgov_hr_network_netmeta_output.json",
    )
    assert (
        "validation/networks/t2d_mace_ctgov_hr_network_benchmark.toml"
        in ctgov_hr_network_reference.input_artifacts
    )
    assert CTGOV_HR_NETWORK_R_ADAPTER.is_file()

    ctgov_binary_network_reference = by_adapter[
        "r_netmeta_psoriasis_ctgov_binary_network_output_validation"
    ]
    assert ctgov_binary_network_reference.target_id == "ctgov_binary_network_netmeta_closed_loop"
    assert ctgov_binary_network_reference.status == "passed"
    assert ctgov_binary_network_reference.certification_effect == "evidence_candidate"
    assert ctgov_binary_network_reference.reference_method == (
        "netmeta CT.gov arm-count closed-loop binary network"
    )
    assert ctgov_binary_network_reference.output_artifacts == (
        "validation/reference_runs/psoriasis_pasi90_ctgov_binary_network_netmeta_output.json",
    )
    assert (
        "validation/networks/psoriasis_pasi90_ctgov_binary_network_benchmark.toml"
        in ctgov_binary_network_reference.input_artifacts
    )
    assert "absolute <= 1e-06" in ctgov_binary_network_reference.tolerance
    assert MULTIARM_R_ADAPTER.is_file()

    component_reference = by_adapter["r_netmeta_component_cnma_output_validation"]
    assert component_reference.target_id == "component_nma_netmeta_cnma"
    assert component_reference.status == "passed"
    assert component_reference.certification_effect == "evidence_candidate"
    assert component_reference.reference_method == "netmeta::discomb additive CNMA"
    assert component_reference.output_artifacts == (
        "validation/reference_runs/component_netmeta_cnma_output.json",
    )
    assert (
        "validation/component/netmeta_component_fixture_benchmark.toml"
        in component_reference.input_artifacts
    )
    assert "absolute <= 1e-06" in component_reference.tolerance
    assert COMPONENT_CNMA_R_ADAPTER.is_file()

    for report in reports:
        for artifact, expected_sha in report.input_sha256.items():
            assert sha256_file(ROOT / artifact) == expected_sha
        for artifact, expected_sha in report.output_sha256.items():
            assert sha256_file(ROOT / artifact) == expected_sha
    assert set(certification_candidate_artifacts(reports)) == {
        "validation/reference_runs/pairwise_metafor_meta_output.json",
        "validation/reference_runs/multiarm_netmeta_output.json",
        "validation/reference_runs/dta_mada_reitsma_output.json",
        "validation/reference_runs/dta_mada_reitsma_midkine_source_output.json",
        "validation/reference_runs/stan_nuts_cmdstan_output.json",
        "validation/reference_runs/dose_response_metafor_polynomial_output.json",
        "validation/reference_runs/sglt2_survival_hr_metafor_output.json",
        "validation/reference_runs/pcsk9_survival_hr_metafor_output.json",
        "validation/reference_runs/t2d_ctgov_hr_network_netmeta_output.json",
        "validation/reference_runs/psoriasis_pasi90_ctgov_binary_network_netmeta_output.json",
        "validation/reference_runs/component_netmeta_cnma_output.json",
    }


def test_reference_run_rejects_nonpassed_certification_evidence():
    raw = {
        "schema_version": "reference_run/v1",
        "target_id": "pairwise_metafor_meta",
        "adapter_id": "bad_unavailable_evidence",
        "reference_method": "metafor and meta",
        "status": "unavailable",
        "certification_effect": "evidence_candidate",
        "command": ["Rscript", "--vanilla", "external/r/pairwise_metafor_meta.R"],
        "executable": "Rscript",
        "executable_found": False,
        "package_versions": {},
        "input_artifacts": [],
        "input_sha256": {},
        "output_artifacts": [],
        "output_sha256": {},
        "tolerance": "",
        "skip_reason": "Rscript is not available.",
    }

    with pytest.raises(CertificationError, match="non-passed reference runs"):
        ReferenceRunReport.from_mapping(raw)


def test_passed_reference_run_requires_hashes_versions_and_tolerance():
    raw = {
        "schema_version": "reference_run/v1",
        "target_id": "pairwise_metafor_meta",
        "adapter_id": "bad_passed_report",
        "reference_method": "metafor and meta",
        "status": "passed",
        "certification_effect": "evidence_candidate",
        "command": ["Rscript", "--vanilla", "external/r/pairwise_metafor_meta.R"],
        "executable": "Rscript",
        "executable_found": True,
        "package_versions": {"R": "4.6.0"},
        "input_artifacts": ["validation/real_meta/sglt2_hf_primary_events.csv"],
        "input_sha256": {},
        "output_artifacts": ["validation/reference_runs/pairwise_metafor_meta_output.json"],
        "output_sha256": {},
        "tolerance": "absolute <= 1e-6",
        "skip_reason": "",
    }

    with pytest.raises(CertificationError, match="missing input SHA-256"):
        ReferenceRunReport.from_mapping(raw)

    raw["input_sha256"] = {
        "validation/real_meta/sglt2_hf_primary_events.csv": "a" * 64,
    }
    with pytest.raises(CertificationError, match="missing output SHA-256"):
        ReferenceRunReport.from_mapping(raw)

    raw["output_sha256"] = {
        "validation/reference_runs/pairwise_metafor_meta_output.json": "b" * 64,
    }
    report = ReferenceRunReport.from_mapping(raw)
    assert report.is_certification_evidence_candidate is True
    assert certification_candidate_artifacts([report]) == (
        "validation/reference_runs/pairwise_metafor_meta_output.json",
    )
