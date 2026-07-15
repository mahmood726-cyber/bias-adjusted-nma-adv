import pytest

from bias_nma_adv.ingestion import (
    EvidenceIngestionRecord,
    IngestionProvenanceError,
    record_from_extractor_row,
    validate_ingestion_records,
)


def test_validates_clinicaltrials_pubmed_and_open_access_sources():
    records = [
        EvidenceIngestionRecord(
            row_id="ctgov_dapa_hf",
            source_type="clinicaltrials_gov",
            nct_id="NCT03036124",
            url="https://clinicaltrials.gov/study/NCT03036124",
            access_statement="ClinicalTrials.gov public registry record.",
        ),
        EvidenceIngestionRecord(
            row_id="pubmed_dapa_hf",
            source_type="pubmed_abstract",
            pmid="31535829",
            url="https://pubmed.ncbi.nlm.nih.gov/31535829/",
            access_statement="PubMed abstract metadata and abstract text.",
        ),
        EvidenceIngestionRecord(
            row_id="pmc_open_access",
            source_type="open_access_paper",
            pmid="22008217",
            pmcid="PMC3196245",
            url="https://europepmc.org/articles/PMC3196245?pdf=render",
            access_statement="Open access paper via Europe PMC.",
        ),
        EvidenceIngestionRecord(
            row_id="doi_open_access",
            source_type="open_access_paper",
            doi="10.1056/NEJMoa1911303",
            url="https://www.nejm.org/doi/full/10.1056/NEJMoa1911303",
            access_statement="Open access paper or publisher-designated OA full text.",
        ),
    ]

    validate_ingestion_records(records)


def test_rejects_open_access_row_when_url_and_source_text_do_not_match_claimed_article():
    record = EvidenceIngestionRecord(
        row_id="bad_fallback",
        source_type="open_access_paper",
        pmid="32865377",
        url="https://hal.univ-lorraine.fr/public/signature_convention_2025.pdf",
        access_statement="Open access candidate from fallback downloader.",
        source_text="",
    )

    with pytest.raises(IngestionProvenanceError, match="verifiable article identity"):
        validate_ingestion_records([record])


def test_extractor_row_does_not_accept_local_filename_as_identity_proof():
    raw = {
        "benchmark_id": "author_meta_trial_00005",
        "study_id": "PMID32865377",
        "pdf_relpath": "rct_trial__32865377__NO_PMCID.pdf",
        "pmcid": None,
        "pmid": "32865377",
        "oa_download_status": "downloaded_fallback_unpaywall_url",
        "oa_download_url": "https://hal.univ-lorraine.fr/public/signature_convention_2025.pdf",
        "model_snapshot_best": {
            "source_text": "",
        },
    }

    record = record_from_extractor_row(raw)

    assert record.pmid == "32865377"
    assert "32865377" not in record.url
    with pytest.raises(IngestionProvenanceError, match="verifiable article identity"):
        record.validate()


def test_extractor_row_accepts_pmc_identity_in_url():
    raw = {
        "benchmark_id": "author_meta_trial_00002",
        "study_id": "PMC3196245",
        "pmcid": "PMC3196245",
        "pmid": "22008217",
        "oa_download_status": "downloaded_fallback_pmcid_direct",
        "oa_download_url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC3196245&blobtype=pdf",
        "model_snapshot_best": {
            "source_text": "odds ratio 1.0, 95% CI 0.8 to 1.1",
        },
    }

    record_from_extractor_row(raw).validate()


def test_rejects_wrong_host_for_registry_and_pubmed_sources():
    registry = EvidenceIngestionRecord(
        row_id="wrong_ctgov_host",
        source_type="clinicaltrials_gov",
        nct_id="NCT03036124",
        url="https://example.org/study/NCT03036124",
        access_statement="ClinicalTrials.gov public registry record.",
    )
    with pytest.raises(IngestionProvenanceError, match="clinicaltrials.gov"):
        registry.validate()

    pubmed = EvidenceIngestionRecord(
        row_id="wrong_pubmed_host",
        source_type="pubmed_abstract",
        pmid="31535829",
        url="https://example.org/31535829",
        access_statement="PubMed abstract metadata and abstract text.",
    )
    with pytest.raises(IngestionProvenanceError, match="pubmed.ncbi.nlm.nih.gov"):
        pubmed.validate()
