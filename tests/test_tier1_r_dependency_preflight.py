from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "validation" / "dependency_preflights" / "tier1_r_dependency_preflight.toml"


def test_tier1_r_dependency_preflight_records_package_and_runtime_status():
    with PREFLIGHT.open("rb") as handle:
        preflight = tomllib.load(handle)

    assert preflight["schema_version"] == "tier1_r_dependency_preflight/v1"
    assert preflight["status"] == "partial"
    assert preflight["certification_effect"] == "none"
    assert preflight["jags_runtime_status"] == "available_via_JAGS_HOME"

    by_name = {package["name"]: package for package in preflight["packages"]}
    assert by_name["multinma"]["load_status"] == "loaded"
    assert by_name["MBNMAdose"]["load_status"] == "loaded"
    assert by_name["rstan"]["load_status"] == "loaded"
    assert by_name["rjags"]["load_status"] == "loaded_with_JAGS_HOME"
    assert by_name["crossnma"]["load_status"] == "loaded_with_JAGS_HOME"
    assert "compatibility preflight" in by_name["crossnma"]["runtime_blocker"]
    assert "blocks model execution" in by_name["crossnma"]["runtime_blocker"]
    assert "compatible arm-level" in by_name["crossnma"]["runtime_blocker"]
    assert "not model validation" in preflight["claim_limit"]
