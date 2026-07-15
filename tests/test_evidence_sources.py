import pytest

from bias_nma_adv.evidence_sources import (
    ALLOWED_SOURCE_TYPES,
    EFFECT_EVIDENCE_SOURCE_TYPES,
    PROTOCOL_ONLY_SOURCE_TYPES,
    EvidenceSource,
    EvidenceSourceError,
    validate_sources,
)


def test_allowed_source_types_are_the_project_boundary():
    assert EFFECT_EVIDENCE_SOURCE_TYPES == {
        "clinicaltrials_gov",
        "pubmed_abstract",
        "open_access_paper",
    }
    assert PROTOCOL_ONLY_SOURCE_TYPES == {
        "other_trial_registry_protocol",
        "who_ictrp_protocol",
    }
    assert ALLOWED_SOURCE_TYPES == EFFECT_EVIDENCE_SOURCE_TYPES | PROTOCOL_ONLY_SOURCE_TYPES


def test_validates_allowed_sources():
    sources = [
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
            source_type="who_ictrp_protocol",
            identifier="DAPA-HF-registry-crosscheck",
            url="https://trialsearch.who.int/Trial2.aspx?TrialID=NCT03036124",
            access_statement="WHO ICTRP registry protocol registration metadata only.",
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
