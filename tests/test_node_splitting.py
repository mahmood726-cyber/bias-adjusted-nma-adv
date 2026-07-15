import pytest

from bias_nma_adv.multiarm import ContrastRow
from bias_nma_adv.node_splitting import fixed_effect_node_splitting


def test_fixed_effect_node_splitting_detects_closed_loop_discordance():
    rows = [
        ContrastRow("S1", "A", "B", 1.0, 0.1),
        ContrastRow("S2", "A", "C", 0.2, 0.1),
        ContrastRow("S3", "B", "C", -0.2, 0.1),
    ]

    diagnostics = fixed_effect_node_splitting(rows, reference_treatment="A")
    by_pair = {
        (diagnostic.treatment_from, diagnostic.treatment_to): diagnostic
        for diagnostic in diagnostics
    }

    ab = by_pair[("A", "B")]
    assert ab.status == "estimable"
    assert ab.n_direct_contrasts == 1
    assert ab.direct_estimate == pytest.approx(1.0)
    assert ab.direct_se == pytest.approx(0.1)
    assert ab.indirect_estimate == pytest.approx(0.4)
    assert ab.indirect_se == pytest.approx(0.1414213562)
    assert ab.difference == pytest.approx(0.6)
    assert ab.difference_se == pytest.approx(0.1732050808)
    assert ab.z_value == pytest.approx(3.4641016151)
    assert ab.p_value == pytest.approx(0.0005320055)
    assert ab.warning is None


def test_fixed_effect_node_splitting_fails_closed_when_indirect_network_disconnects():
    rows = [
        ContrastRow("S1", "A", "B", 1.0, 0.1),
        ContrastRow("S2", "A", "C", 0.2, 0.1),
    ]

    diagnostics = fixed_effect_node_splitting(rows, reference_treatment="A")

    assert {diagnostic.status for diagnostic in diagnostics} == {"not_estimable"}
    assert all(diagnostic.indirect_estimate is None for diagnostic in diagnostics)
    assert all(diagnostic.p_value is None for diagnostic in diagnostics)
    assert all(diagnostic.warning for diagnostic in diagnostics)
