import math

import pytest

from bias_nma_adv.multiarm import (
    ContrastRow,
    diagnose_multiarm_design,
    fit_multiarm_gls,
    report_multiarm_gls_fit,
)


TOL = 1e-6


FIXTURE_CONSISTENT = [
    ("S1", "A", 10, 100), ("S1", "B", 20, 100),
    ("S2", "A", 12, 120), ("S2", "B", 18, 110),
    ("S3", "A", 8, 90), ("S3", "C", 15, 95),
    ("S4", "A", 11, 100), ("S4", "C", 14, 100),
    ("S5", "B", 9, 80), ("S5", "C", 13, 85),
    ("S6", "A", 15, 150), ("S6", "B", 25, 150), ("S6", "C", 20, 150),
]


FIXTURE_HETEROGENEOUS = [
    ("S1", "A", 10, 100), ("S1", "B", 30, 100),
    ("S2", "A", 25, 120), ("S2", "B", 18, 110),
    ("S3", "A", 8, 90), ("S3", "C", 25, 95),
    ("S4", "A", 22, 100), ("S4", "C", 14, 100),
    ("S5", "B", 9, 80), ("S5", "C", 30, 85),
    ("S6", "A", 15, 150), ("S6", "B", 25, 150), ("S6", "C", 12, 150),
    ("S7", "A", 40, 200), ("S7", "B", 20, 180),
]


def _log_odds(events, n):
    return math.log(events / (n - events))


def _rows_from_arms(arms):
    by_study = {}
    for study, treatment, events, n in arms:
        by_study.setdefault(study, []).append((treatment, events, n))

    rows = []
    for study, study_arms in by_study.items():
        for i in range(len(study_arms)):
            for j in range(i + 1, len(study_arms)):
                t1, e1, n1 = study_arms[i]
                t2, e2, n2 = study_arms[j]
                est = _log_odds(e2, n2) - _log_odds(e1, n1)
                se = math.sqrt(1 / e1 + 1 / (n1 - e1) + 1 / e2 + 1 / (n2 - e2))
                rows.append(ContrastRow(study=study, t1=t1, t2=t2, est=est, se=se))
    return rows


def test_fixed_effect_multiarm_matches_netmeta_portfolio_fixture():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_CONSISTENT), reference_treatment="A")

    assert fit.model == "fixed"
    assert fit.multi_arm_studies == ("S6",)
    assert fit.tau2 == 0.0

    est_b, se_b = fit.effect_vs_reference("B")
    est_c, se_c = fit.effect_vs_reference("C")
    assert est_b == pytest.approx(0.5925559659, abs=TOL)
    assert se_b == pytest.approx(0.2043271463, abs=TOL)
    assert est_c == pytest.approx(0.4841298055, abs=TOL)
    assert se_c == pytest.approx(0.2155057123, abs=TOL)


def test_multiarm_design_diagnostic_reports_rank_connectivity_and_cliques():
    diagnostic = diagnose_multiarm_design(
        _rows_from_arms(FIXTURE_CONSISTENT),
        reference_treatment="A",
    )

    assert diagnostic.reference_treatment == "A"
    assert diagnostic.treatments == ("A", "B", "C")
    assert diagnostic.n_studies == 6
    assert diagnostic.n_contrast_rows == 8
    assert diagnostic.n_parameters == 2
    assert diagnostic.design_rank == 2
    assert diagnostic.connected is True
    assert diagnostic.disconnected_treatments == ()
    assert diagnostic.estimable is True

    by_study = {row.study: row for row in diagnostic.study_diagnostics}
    assert by_study["S6"].multi_arm is True
    assert by_study["S6"].n_contrasts == 3
    assert by_study["S6"].expected_contrasts == 3
    assert by_study["S6"].complete_pairwise_clique is True


def test_multiarm_fit_report_passes_with_design_metadata():
    report = report_multiarm_gls_fit(
        _rows_from_arms(FIXTURE_CONSISTENT),
        reference_treatment="A",
    )

    assert report.status == "passed"
    assert report.stage == "fit"
    assert report.message == "multi-arm GLS fit completed"
    assert report.model == "fixed"
    assert report.reference_treatment == "A"
    assert report.n_studies == 6
    assert report.n_contrast_rows == 8
    assert report.design_rank == 2
    assert report.connected is True
    assert report.estimable is True


def test_fixed_effect_heterogeneous_multiarm_matches_netmeta_portfolio_fixture():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_HETEROGENEOUS), reference_treatment="A")

    est_b, se_b = fit.effect_vs_reference("B")
    est_c, se_c = fit.effect_vs_reference("C")
    assert est_b == pytest.approx(0.0038784945, abs=TOL)
    assert se_b == pytest.approx(0.1604752859, abs=TOL)
    assert est_c == pytest.approx(0.2341504618, abs=TOL)
    assert se_c == pytest.approx(0.2056420348, abs=TOL)


def test_random_effects_multiarm_matches_netmeta_portfolio_fixture():
    fit = fit_multiarm_gls(
        _rows_from_arms(FIXTURE_HETEROGENEOUS),
        reference_treatment="A",
        model="random",
    )

    assert fit.model == "random"
    assert fit.q == pytest.approx(45.1375162628, abs=1e-5)
    assert fit.df == 6
    assert fit.tau2 == pytest.approx(0.8921998165, abs=TOL)

    est_b, se_b = fit.effect_vs_reference("B")
    est_c, se_c = fit.effect_vs_reference("C")
    assert est_b == pytest.approx(0.0805775134, abs=TOL)
    assert se_b == pytest.approx(0.4660587911, abs=TOL)
    assert est_c == pytest.approx(0.3584933242, abs=TOL)
    assert se_c == pytest.approx(0.5262952049, abs=TOL)


def test_fixed_effect_influence_diagnostics_preserve_gls_hat_invariants():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_CONSISTENT), reference_treatment="A")

    diagnostics = fit.influence_diagnostics
    assert len(diagnostics) == 7
    assert sum(row.leverage for row in diagnostics) == pytest.approx(2.0, abs=1e-10)
    assert all(math.isfinite(row.standardized_residual) for row in diagnostics)
    assert all(math.isfinite(row.cook_distance) for row in diagnostics)
    assert all(row.variance > 0.0 for row in diagnostics)

    multiarm_rows = [row for row in diagnostics if row.study == "S6"]
    assert [(row.treatment_from, row.treatment_to) for row in multiarm_rows] == [
        ("A", "B"),
        ("A", "C"),
    ]


def test_fixed_effect_contribution_diagnostics_are_nonnegative_and_normalized():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_CONSISTENT), reference_treatment="A")

    diagnostics = fit.contribution_diagnostics
    assert len(diagnostics) == len(fit.nonreference_treatments) * len(fit.influence_diagnostics)
    assert all(row.absolute_mapping_weight >= 0.0 for row in diagnostics)
    assert all(row.contribution >= 0.0 for row in diagnostics)

    for target in fit.nonreference_treatments:
        target_rows = [row for row in diagnostics if row.target_treatment == target]
        assert target_rows
        assert sum(row.contribution for row in target_rows) == pytest.approx(1.0, abs=1e-12)

    multiarm_rows = [row for row in diagnostics if row.study == "S6"]
    assert any(row.target_treatment == "B" and row.contribution > 0.0 for row in multiarm_rows)
    assert any(row.target_treatment == "C" and row.contribution > 0.0 for row in multiarm_rows)


def test_study_contribution_matrix_aggregates_nonnegative_contributions_by_target():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_CONSISTENT), reference_treatment="A")

    matrix = fit.study_contribution_matrix()

    assert set(matrix) == set(fit.nonreference_treatments)
    for target, study_values in matrix.items():
        assert study_values
        assert all(value >= 0.0 for value in study_values.values())
        assert sum(study_values.values()) == pytest.approx(1.0, abs=1e-12)
        assert "S6" in study_values
        assert study_values["S6"] > 0.0
        assert target in {"B", "C"}


def test_study_contribution_heatmap_is_renderer_ready_and_normalized():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_CONSISTENT), reference_treatment="A")

    heatmap = fit.study_contribution_heatmap()

    assert heatmap.target_treatments == fit.nonreference_treatments
    assert heatmap.studies == tuple(sorted(heatmap.studies))
    assert "S6" in heatmap.studies
    assert len(heatmap.values) == len(fit.nonreference_treatments)
    assert all(len(row) == len(heatmap.studies) for row in heatmap.values)
    assert all(value >= 0.0 for row in heatmap.values for value in row)
    assert all(sum(row) == pytest.approx(1.0, abs=1e-12) for row in heatmap.values)
    assert any("not CINeMA" in warning for warning in heatmap.warnings)


def test_influence_diagnostics_flag_deliberately_discordant_contrast():
    fit = fit_multiarm_gls(_rows_from_arms(FIXTURE_HETEROGENEOUS), reference_treatment="A")

    most_influential = max(fit.influence_diagnostics, key=lambda row: row.cook_distance)

    assert most_influential.study == "S5"
    assert (most_influential.treatment_from, most_influential.treatment_to) == ("B", "C")
    assert most_influential.standardized_residual == pytest.approx(3.4130943718, abs=1e-6)
    assert most_influential.cook_distance == pytest.approx(2.1129830474, abs=1e-6)


def test_incomplete_multiarm_clique_warns_and_drops_study():
    rows = [
        row
        for row in _rows_from_arms(FIXTURE_CONSISTENT)
        if not (row.study == "S6" and {row.t1, row.t2} == {"B", "C"})
    ]

    fit = fit_multiarm_gls(rows, reference_treatment="A")

    assert any("S6" in warning for warning in fit.warnings)
    assert "S6" not in fit.multi_arm_studies


def test_design_diagnostic_flags_incomplete_multiarm_clique_without_fitting():
    rows = [
        row
        for row in _rows_from_arms(FIXTURE_CONSISTENT)
        if not (row.study == "S6" and {row.t1, row.t2} == {"B", "C"})
    ]

    diagnostic = diagnose_multiarm_design(rows, reference_treatment="A")
    by_study = {row.study: row for row in diagnostic.study_diagnostics}

    assert by_study["S6"].multi_arm is True
    assert by_study["S6"].n_contrasts == 2
    assert by_study["S6"].expected_contrasts == 3
    assert by_study["S6"].complete_pairwise_clique is False
    assert any("S6" in warning for warning in diagnostic.warnings)


def test_incompatible_multiarm_covariance_fails_closed():
    rows = [
        ContrastRow("S_bad", "A", "B", 0.1, 0.1),
        ContrastRow("S_bad", "A", "C", 0.1, 0.1),
        ContrastRow("S_bad", "B", "C", 0.1, 10.0),
    ]

    with pytest.raises(ValueError, match="recovered negative arm variance"):
        fit_multiarm_gls(rows, reference_treatment="A")


def test_fit_report_records_incompatible_covariance_without_raising():
    rows = [
        ContrastRow("S_bad", "A", "B", 0.1, 0.1),
        ContrastRow("S_bad", "A", "C", 0.1, 0.1),
        ContrastRow("S_bad", "B", "C", 0.1, 10.0),
    ]

    report = report_multiarm_gls_fit(rows, reference_treatment="A")

    assert report.status == "failed"
    assert report.stage == "design"
    assert "recovered negative arm variance" in report.message
    assert report.estimable is False
    assert report.warnings == (report.message,)


def test_fit_report_records_empty_input_without_raising():
    report = report_multiarm_gls_fit([], reference_treatment="A")

    assert report.status == "failed"
    assert report.stage == "input"
    assert report.message == "no contrasts supplied."
    assert report.reference_treatment == "A"
    assert report.n_contrast_rows == 0


def test_disconnected_network_fails_closed():
    rows = [
        ContrastRow("S1", "A", "B", 0.1, 0.2),
        ContrastRow("S2", "C", "D", 0.2, 0.2),
    ]

    diagnostic = diagnose_multiarm_design(rows, reference_treatment="A")

    assert diagnostic.connected is False
    assert diagnostic.disconnected_treatments == ("C", "D")
    assert diagnostic.estimable is False

    report = report_multiarm_gls_fit(rows, reference_treatment="A")

    assert report.status == "failed"
    assert report.stage == "design"
    assert report.disconnected_treatments == ("C", "D")
    assert "disconnected" in report.message

    with pytest.raises(ValueError, match="disconnected"):
        fit_multiarm_gls(rows, reference_treatment="A")
