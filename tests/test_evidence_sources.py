import pytest

from bias_nma_adv.evidence_sources import (
    ALLOWED_SOURCE_TYPES,
    EFFECT_EVIDENCE_SOURCE_TYPES,
    PROTOCOL_ONLY_SOURCE_TYPES,
    PUBLISHED_EFFECT_SOURCE_TYPES,
    REGISTRY_FIRST_EFFECT_SOURCE_TYPES,
    REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES,
    REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES,
    EvidenceSource,
    EvidenceSourceError,
    validate_sources,
)


def test_allowed_source_types_are_the_project_boundary():
    assert EFFECT_EVIDENCE_SOURCE_TYPES == {
        "aact_clinicaltrials_gov",
        "clinicaltrials_gov",
        "pubmed_abstract",
        "open_access_paper",
        "ema_epar",
        "fda_review",
        "pactr_results",
        "who_ictrp_results",
    }
    assert REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES == {
        "pactr_results",
        "who_ictrp_results",
    }
    assert REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES == {
        "ema_epar",
        "fda_review",
    }
    assert PUBLISHED_EFFECT_SOURCE_TYPES == {
        "open_access_paper",
        "pubmed_abstract",
    }
    assert REGISTRY_FIRST_EFFECT_SOURCE_TYPES == {
        "aact_clinicaltrials_gov",
        "clinicaltrials_gov",
        "ema_epar",
        "fda_review",
        "pactr_results",
        "who_ictrp_results",
    }
    assert PROTOCOL_ONLY_SOURCE_TYPES == {
        "other_trial_registry_protocol",
        "pactr_protocol",
        "who_ictrp_protocol",
    }
    assert ALLOWED_SOURCE_TYPES == EFFECT_EVIDENCE_SOURCE_TYPES | PROTOCOL_ONLY_SOURCE_TYPES


def test_validates_allowed_sources():
    sources = [
        EvidenceSource(
            source_type="aact_clinicaltrials_gov",
            identifier="NCT03036124",
            url="https://aact.ctti-clinicaltrials.org/",
            access_statement="AACT public data mirror derived from ClinicalTrials.gov records.",
        ),
        EvidenceSource(
            source_type="clinicaltrials_gov",
            identifier="NCT03036124",
            url="https://clinicaltrials.gov/study/NCT03036124",
            access_statement="ClinicalTrials.gov public record.",
        ),
        EvidenceSource(
            source_type="pubmed_abstract",
            identifier="31535829",
            url="https://pubmed.ncbi.nlm.nih.gov/31535829/",
            access_statement="PubMed abstract metadata and abstract text.",
        ),
        EvidenceSource(
            source_type="open_access_paper",
            identifier="doi:10.1056/NEJMoa1911303",
            url="https://www.nejm.org/doi/full/10.1056/NEJMoa1911303",
            access_statement="Open access paper or publisher-designated OA full text.",
        ),
        EvidenceSource(
            source_type="who_ictrp_results",
            identifier="PACTR202001234567890",
            url="https://trialsearch.who.int/Trial2.aspx?TrialID=PACTR202001234567890",
            access_statement="WHO ICTRP downloaded public outcome result row with posted results.",
        ),
        EvidenceSource(
            source_type="pactr_results",
            identifier="PACTR202001234567890",
            url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
            access_statement="PACTR public trial record with results available and outcome results.",
        ),
        EvidenceSource(
            source_type="fda_review",
            identifier="NDA 020639",
            url="https://www.accessdata.fda.gov/drugsatfda_docs/nda/2001/020639.cfm",
            access_statement="Public FDA Drugs@FDA statistical review package with per-trial results.",
        ),
        EvidenceSource(
            source_type="ema_epar",
            identifier="EMEA/H/C/000123",
            url="https://www.ema.europa.eu/en/medicines/human/EPAR/example",
            access_statement="Public EMA EPAR assessment report with trial outcome tables.",
        ),
        EvidenceSource(
            source_type="who_ictrp_protocol",
            identifier="DAPA-HF-registry-crosscheck",
            url="https://trialsearch.who.int/Trial2.aspx?TrialID=NCT03036124",
            access_statement="WHO ICTRP registry protocol registration metadata only.",
        ),
        EvidenceSource(
            source_type="pactr_protocol",
            identifier="PACTR202001234567890",
            url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
            access_statement="PACTR registry protocol registration metadata only.",
        ),
        EvidenceSource(
            source_type="other_trial_registry_protocol",
            identifier="regional-registry-123",
            url="https://registry.example.org/trials/123",
            access_statement="Other trial registry protocol record used for registration metadata only.",
        ),
    ]

    validate_sources(sources)


def test_rejects_sources_outside_allowed_boundary():
    source = EvidenceSource(
        source_type="closed_ipd",
        identifier="private-dataset",
        url="https://example.org/private",
        access_statement="Private dataset.",
    )

    with pytest.raises(EvidenceSourceError, match="not allowed"):
        validate_sources([source])


def test_rejects_malformed_identifiers():
    bad_nct = EvidenceSource(
        source_type="clinicaltrials_gov",
        identifier="DAPA-HF",
        url="https://clinicaltrials.gov/study/NCT03036124",
        access_statement="ClinicalTrials.gov public record.",
    )
    with pytest.raises(EvidenceSourceError, match="NCT01234567"):
        validate_sources([bad_nct])

    bad_pmid = EvidenceSource(
        source_type="pubmed_abstract",
        identifier="PMID31535829",
        url="https://pubmed.ncbi.nlm.nih.gov/31535829/",
        access_statement="PubMed abstract metadata and abstract text.",
    )
    with pytest.raises(EvidenceSourceError, match="numeric PMIDs"):
        validate_sources([bad_pmid])


def test_rejects_aact_source_without_aact_ctgov_identity_or_host():
    missing_role = EvidenceSource(
        source_type="aact_clinicaltrials_gov",
        identifier="NCT03036124",
        url="https://aact.ctti-clinicaltrials.org/",
        access_statement="Public registry record.",
    )
    with pytest.raises(EvidenceSourceError, match="ClinicalTrials.gov-derived AACT"):
        validate_sources([missing_role])

    wrong_host = EvidenceSource(
        source_type="aact_clinicaltrials_gov",
        identifier="NCT03036124",
        url="https://example.org/NCT03036124",
        access_statement="AACT public data mirror derived from ClinicalTrials.gov records.",
    )
    with pytest.raises(EvidenceSourceError, match="AACT or ClinicalTrials.gov URL"):
        validate_sources([wrong_host])


def test_rejects_protocol_source_without_protocol_role_or_who_host():
    missing_role = EvidenceSource(
        source_type="other_trial_registry_protocol",
        identifier="regional-registry-123",
        url="https://registry.example.org/trials/123",
        access_statement="Study result page.",
    )
    with pytest.raises(EvidenceSourceError, match="protocol or registration role"):
        validate_sources([missing_role])

    wrong_who_host = EvidenceSource(
        source_type="who_ictrp_protocol",
        identifier="DAPA-HF-registry-crosscheck",
        url="https://example.org/Trial2.aspx?TrialID=NCT03036124",
        access_statement="WHO ICTRP registry protocol registration metadata only.",
    )
    with pytest.raises(EvidenceSourceError, match="who.int"):
        validate_sources([wrong_who_host])

    wrong_pactr_protocol_host = EvidenceSource(
        source_type="pactr_protocol",
        identifier="PACTR202001234567890",
        url="https://example.org/TrialDisplay.aspx?TrialID=PACTR202001234567890",
        access_statement="PACTR registry protocol registration metadata only.",
    )
    with pytest.raises(EvidenceSourceError, match="PACTR registry URL"):
        validate_sources([wrong_pactr_protocol_host])


def test_rejects_registry_result_source_without_result_role_or_registry_host():
    protocol_only_result = EvidenceSource(
        source_type="who_ictrp_results",
        identifier="PACTR202001234567890",
        url="https://trialsearch.who.int/Trial2.aspx?TrialID=PACTR202001234567890",
        access_statement="WHO ICTRP protocol-only registration metadata.",
    )
    with pytest.raises(EvidenceSourceError, match="protocol-only"):
        validate_sources([protocol_only_result])

    missing_result_role = EvidenceSource(
        source_type="pactr_results",
        identifier="PACTR202001234567890",
        url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
        access_statement="PACTR public trial registry record.",
    )
    with pytest.raises(EvidenceSourceError, match="result or outcome role"):
        validate_sources([missing_result_role])

    wrong_pactr_host = EvidenceSource(
        source_type="pactr_results",
        identifier="PACTR202001234567890",
        url="https://example.org/TrialDisplay.aspx?TrialID=PACTR202001234567890",
        access_statement="PACTR public record with results available and outcome results.",
    )
    with pytest.raises(EvidenceSourceError, match="PACTR registry URL"):
        validate_sources([wrong_pactr_host])


def test_rejects_regulatory_review_source_without_review_role_or_host():
    missing_review_role = EvidenceSource(
        source_type="fda_review",
        identifier="NDA 020639",
        url="https://www.accessdata.fda.gov/drugsatfda_docs/nda/2001/020639.cfm",
        access_statement="FDA public page.",
    )
    with pytest.raises(EvidenceSourceError, match="FDA review-package role"):
        validate_sources([missing_review_role])

    wrong_fda_host = EvidenceSource(
        source_type="fda_review",
        identifier="NDA 020639",
        url="https://example.org/020639.cfm",
        access_statement="Public FDA Drugs@FDA statistical review package with per-trial results.",
    )
    with pytest.raises(EvidenceSourceError, match="fda.gov"):
        validate_sources([wrong_fda_host])

    wrong_ema_host = EvidenceSource(
        source_type="ema_epar",
        identifier="EMEA/H/C/000123",
        url="https://example.org/epar/000123",
        access_statement="Public EMA EPAR assessment report with trial outcome tables.",
    )
    with pytest.raises(EvidenceSourceError, match="ema.europa.eu"):
        validate_sources([wrong_ema_host])
