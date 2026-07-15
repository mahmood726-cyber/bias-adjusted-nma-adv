from pathlib import Path

from bias_nma_adv.certification import load_reference_targets, summarize_reference_targets


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_reuse_map_names_source_constrained_reuse_lanes():
    text = (ROOT / "docs" / "portfolio_reuse_map.md").read_text(encoding="utf-8")

    for required in (
        "C:\\Projects\\wasserstein",
        "C:\\Projects\\advanced-nma-pooling",
        "C:\\Projects\\complex-evidence-synthesis-map",
        "C:\\Projects\\aact-kit",
        "C:\\Projects\\sheaf-nma",
        "C:\\Projects\\spec-collapse-atlas",
        "C:\\Projects\\topo-transport-ma",
        "C:\\Projects\\allmeta\\shared\\nma-multiarm-v1.js",
        "C:\\Users\\mahmo\\code\\rct-extractor-v2",
        "effect evidence from ClinicalTrials.gov records, PubMed abstracts, and open-access papers",
        "protocol metadata may also come from WHO ICTRP or other registries",
        "Static-Vs-Dynamic Hardcode Disclosure",
    ):
        assert required in text

    assert "must not be hardcoded into shipped code" in text
    assert "non-article fallback PDF" in text
    assert "Portfolio methods are not imported as black boxes" in text


def test_portfolio_reuse_targets_are_registered_as_planned():
    targets = load_reference_targets(ROOT / "validation" / "reference_targets.toml")
    by_id = {target.id: target for target in targets}

    expected = {
        "survival_km_oa_ipd_reconstruction",
        "transportability_support_topology",
        "multiarm_gls_netmeta_portfolio_fixture",
        "source_backed_ingestion_rct_extractor",
        "tier_one_adapter_governance",
        "method_choice_robustness_wrapper",
    }
    assert expected <= set(by_id)
    assert all(by_id[target_id].status == "planned" for target_id in expected)

    summary = summarize_reference_targets(targets)
    assert summary == {"planned": len(targets)}
