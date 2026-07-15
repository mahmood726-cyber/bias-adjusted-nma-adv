from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "validation.yml"


def test_validation_workflow_runs_machine_readable_gates():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m pytest -q" in text
    assert "python scripts\\validate_benchmark_registry.py --json" in text
    assert "python scripts\\write_validation_status.py --output validation_status.json" in text
    assert "actions/upload-artifact@v4" in text
    assert "validation-status" in text


def test_validation_workflow_does_not_claim_certification():
    text = WORKFLOW.read_text(encoding="utf-8").lower()

    forbidden = (
        "production certified",
        "clinical reporting enabled",
        "hta reporting enabled",
        "superiority passed",
    )
    for term in forbidden:
        assert term not in text
