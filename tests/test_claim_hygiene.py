from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_binary_event_demo_scripts_do_not_label_odds_ratios_as_hazard_ratios():
    paths = [
        ROOT / "run_all_cardio_nmas.py",
        ROOT / "run_hfref_nma.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "Pooled HR" not in text
        assert "HR =" not in text


def test_demo_scripts_are_not_presented_as_validation_evidence():
    for name in ("run_all_cardio_nmas.py", "run_hfref_nma.py", "try_real_cardio_nma.py"):
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        assert "not validation evidence" in text
