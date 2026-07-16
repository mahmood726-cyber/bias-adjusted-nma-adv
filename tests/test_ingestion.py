import pytest

from bias_nma_adv.ingestion import (
    EvidenceIngestionRecord,
    ExtractionProvenance,
    IngestionProvenanceError,
    ProtocolOnlyRegistryRecord,
    ProofCarryingEffectRecord,
    build_protocol_completeness_ledger,
    proof_effect_from_regulatory_review_row,
    proof_effect_from_registry_result_row,
    record_from_protocol_registry_row,
    record_from_regulatory_review_row,
    proof_effect_from_extractor_row,
    record_from_registry_result_row,
    record_from_extractor_row,
    validate_ingestion_records,
    validate_proof_carrying_effects,
)


def test_validates_clinicaltrials_pubmed_and_open_access_sources():
    records = [
        EvidenceIngestionRecord(
            row_id="aact_dapa_hf_results",
            source_type="aact_clinicaltrials_gov",
            nct_id="NCT03036124",
            url="https://aact.ctti-clinicaltrials.org/",
            access_statement="AACT public data mirror derived from ClinicalTrials.gov records.",
            source_text=(
                "AACT row for NCT03036124 from ClinicalTrials.gov results tables "
                "with 386 events among 2373 participants."
            ),
        ),
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
        EvidenceIngestionRecord(
            row_id="fda_review_row",
            source_type="fda_review",
            regulatory_id="NDA 020639",
            url="https://www.accessdata.fda.gov/drugsatfda_docs/nda/2001/020639.cfm",
            access_statement="Public FDA Drugs@FDA statistical review package with per-trial results.",
            source_text=(
                "NDA 020639 Study 301 result: hazard ratio 0.82 with 95% confidence "
                "interval 0.70 to 0.96 among 512 participants."
            ),
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


def test_rejects_aact_source_without_nct_identity_or_aact_role():
    missing_nct_text = EvidenceIngestionRecord(
        row_id="aact_missing_identity",
        source_type="aact_clinicaltrials_gov",
        nct_id="NCT03036124",
        url="https://aact.ctti-clinicaltrials.org/",
        access_statement="AACT public data mirror derived from ClinicalTrials.gov records.",
        source_text="Downloaded result row without the trial identifier.",
    )
    with pytest.raises(IngestionProvenanceError, match="NCT03036124"):
        missing_nct_text.validate()

    missing_aact_role = EvidenceIngestionRecord(
        row_id="aact_missing_role",
        source_type="aact_clinicaltrials_gov",
        nct_id="NCT03036124",
        url="https://aact.ctti-clinicaltrials.org/",
        access_statement="Clinical trial results table.",
        source_text="NCT03036124 result row.",
    )
    with pytest.raises(IngestionProvenanceError, match="ClinicalTrials.gov-derived AACT"):
        missing_aact_role.validate()


def test_protocol_only_registry_source_cannot_supply_model_ready_effect():
    protocol_source = EvidenceIngestionRecord(
        row_id="who_protocol_dapa_hf",
        source_type="who_ictrp_protocol",
        url="https://trialsearch.who.int/Trial2.aspx?TrialID=NCT03036124",
        access_statement="WHO ICTRP registry protocol registration metadata only.",
    )

    with pytest.raises(IngestionProvenanceError, match="protocol-only"):
        protocol_source.validate()


def test_protocol_only_registry_records_feed_metadata_ledger_not_effect_pool():
    records = [
        record_from_protocol_registry_row(
            {
                "source_type": "who_ictrp_protocol",
                "registry_id": "PACTR202001234567890",
                "url": "https://trialsearch.who.int/Trial2.aspx?TrialID=PACTR202001234567890",
                "access_statement": "WHO ICTRP registry protocol registration metadata only.",
                "registered_primary_outcome": "all-cause mortality",
                "reported_primary_outcome": "cardiovascular death",
                "status": "completed",
                "interventions": ["DrugX"],
                "source_text": (
                    "PACTR202001234567890 registered primary outcome: all-cause mortality."
                ),
            }
        ),
        ProtocolOnlyRegistryRecord(
            row_id="pactr_protocol_without_results",
            source_type="pactr_protocol",
            registry_id="PACTR202009876543210",
            url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202009876543210",
            access_statement="PACTR registry protocol registration metadata only.",
            registered_primary_outcome="hospitalization",
            status="completed",
            interventions=("drugx",),
            source_text="PACTR202009876543210 registered primary outcome: hospitalization.",
        ),
    ]

    ledger = build_protocol_completeness_ledger(
        records,
        reported_registry_ids=["PACTR202001234567890"],
    )

    assert all(record.model_ready_effect is False for record in records)
    assert ledger["schema_version"] == "protocol_completeness_ledger/v1"
    assert ledger["model_ready_effect"] is False
    assert ledger["n_denominator_records"] == 2
    assert ledger["n_reported_records"] == 1
    assert ledger["n_unreported_records"] == 1
    assert ledger["unreported_ratio"] == pytest.approx(0.5)
    assert ledger["source_type_counts"] == {
        "pactr_protocol": 1,
        "who_ictrp_protocol": 1,
    }
    assert ledger["registered_primary_outcomes"]["PACTR202001234567890"] == (
        "all-cause mortality"
    )
    assert ledger["outcome_switching_scores"]["PACTR202001234567890"] == 1.0


def test_protocol_only_registry_record_rejects_result_source_type_or_missing_identity():
    with pytest.raises(IngestionProvenanceError, match="protocol-only"):
        record_from_protocol_registry_row(
            {
                "source_type": "pactr_results",
                "registry_id": "PACTR202001234567890",
                "url": "https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
                "access_statement": "PACTR registry protocol registration metadata only.",
                "registered_primary_outcome": "mortality",
                "source_text": "PACTR202001234567890 registered primary outcome: mortality.",
            }
        ).validate()

    missing_identity = ProtocolOnlyRegistryRecord(
        row_id="who_protocol_missing_identity",
        source_type="who_ictrp_protocol",
        registry_id="PACTR202001234567890",
        url="https://trialsearch.who.int/Trial2.aspx?TrialID=OTHER",
        access_statement="WHO ICTRP registry protocol registration metadata only.",
        registered_primary_outcome="mortality",
        source_text="No matching trial identifier here.",
    )
    with pytest.raises(IngestionProvenanceError, match="claimed registry_id"):
        missing_identity.validate()


def test_result_level_registry_sources_can_supply_model_ready_effects_when_numeric():
    records = [
        EvidenceIngestionRecord(
            row_id="who_ictrp_result_row",
            source_type="who_ictrp_results",
            registry_id="PACTR202001234567890",
            url="https://trialsearch.who.int/Trial2.aspx?TrialID=PACTR202001234567890",
            access_statement="WHO ICTRP downloaded public row with results available: yes.",
            source_text=(
                "PACTR202001234567890 outcome result: hazard ratio 0.82 "
                "with 95% confidence interval 0.70 to 0.96."
            ),
        ),
        EvidenceIngestionRecord(
            row_id="pactr_result_row",
            source_type="pactr_results",
            registry_id="PACTR202001234567890",
            url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
            access_statement="PACTR public trial record with results available: yes.",
            source_text=(
                "PACTR202001234567890 numeric outcome results: 14 events among "
                "120 participants and 21 events among 118 participants."
            ),
        ),
    ]

    validate_ingestion_records(records)


def test_registry_result_rows_are_fail_closed_for_protocol_or_non_numeric_rows():
    protocol_only = EvidenceIngestionRecord(
        row_id="who_protocol_result_mislabel",
        source_type="who_ictrp_results",
        registry_id="PACTR202001234567890",
        url="https://trialsearch.who.int/Trial2.aspx?TrialID=PACTR202001234567890",
        access_statement="WHO ICTRP protocol-only registration metadata.",
        source_text="PACTR202001234567890 results available: no.",
    )
    with pytest.raises(IngestionProvenanceError, match="protocol-only|result-level"):
        protocol_only.validate()

    no_registry_id = EvidenceIngestionRecord(
        row_id="pactr_missing_id",
        source_type="pactr_results",
        url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
        access_statement="PACTR public trial record with results available: yes.",
        source_text="Outcome result: hazard ratio 0.90, 95% confidence interval 0.80 to 1.01.",
    )
    with pytest.raises(IngestionProvenanceError, match="registry_id"):
        no_registry_id.validate()

    non_numeric = EvidenceIngestionRecord(
        row_id="pactr_non_numeric_result",
        source_type="pactr_results",
        registry_id="PACTR202001234567890",
        url="https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
        access_statement="PACTR public trial record with results available: yes.",
        source_text="PACTR202001234567890 outcome result was favourable.",
    )
    with pytest.raises(IngestionProvenanceError, match="numeric model-ready"):
        non_numeric.validate()


def test_downloaded_registry_result_row_helpers_create_proof_carrying_effects():
    raw = {
        "source_type": "pactr_results",
        "registry_id": "PACTR202001234567890",
        "url": "https://pactr.samrc.ac.za/TrialDisplay.aspx?TrialID=PACTR202001234567890",
        "access_statement": "PACTR public trial record with results available: yes.",
        "source_text": (
            "PACTR202001234567890 reported result-level outcome: hazard ratio 0.82, "
            "95% confidence interval 0.70 to 0.96."
        ),
        "record_id": "pactr_demo_hr",
        "study_id": "PACTR202001234567890",
        "outcome_name": "time-to-event endpoint",
        "effect_type": "HR",
        "point_estimate": 0.82,
        "ci_lower": 0.70,
        "ci_upper": 0.96,
    }

    ingestion_record = record_from_registry_result_row(raw)
    effect_record = proof_effect_from_registry_result_row(raw)

    assert ingestion_record.registry_id == "PACTR202001234567890"
    assert effect_record.source.source_type == "pactr_results"
    assert effect_record.is_meta_analysis_ready is True
    assert effect_record.to_dict()["source"]["registry_id"] == "PACTR202001234567890"


def test_downloaded_registry_result_helper_rejects_protocol_source_type():
    with pytest.raises(IngestionProvenanceError, match="registry result rows"):
        record_from_registry_result_row(
            {
                "source_type": "who_ictrp_protocol",
                "registry_id": "PACTR202001234567890",
                "url": "https://trialsearch.who.int/Trial2.aspx?TrialID=PACTR202001234567890",
                "source_text": "Protocol row only.",
            }
        )


def test_regulatory_review_rows_can_supply_model_ready_effects_when_numeric():
    raw = {
        "source_type": "fda_review",
        "regulatory_id": "NDA 020639",
        "url": "https://www.accessdata.fda.gov/drugsatfda_docs/nda/2001/020639.cfm",
        "access_statement": "Public FDA Drugs@FDA statistical review package with per-trial results.",
        "source_text": (
            "NDA 020639 Study 301 reported result: hazard ratio 0.82, "
            "95% confidence interval 0.70 to 0.96."
        ),
        "record_id": "fda_demo_hr",
        "study_id": "Study 301",
        "outcome_name": "time-to-event endpoint",
        "effect_type": "HR",
        "point_estimate": 0.82,
        "ci_lower": 0.70,
        "ci_upper": 0.96,
    }

    ingestion_record = record_from_regulatory_review_row(raw)
    effect_record = proof_effect_from_regulatory_review_row(raw)

    assert ingestion_record.regulatory_id == "NDA 020639"
    assert effect_record.source.source_type == "fda_review"
    assert effect_record.is_meta_analysis_ready is True
    assert effect_record.to_dict()["source"]["regulatory_id"] == "NDA 020639"


def test_regulatory_review_rows_are_fail_closed_for_summary_only_or_wrong_host():
    summary_only = EvidenceIngestionRecord(
        row_id="fda_summary_only",
        source_type="fda_review",
        regulatory_id="NDA 020639",
        url="https://www.accessdata.fda.gov/drugsatfda_docs/nda/2001/020639.cfm",
        access_statement="Public FDA Drugs@FDA statistical review package.",
        source_text="NDA 020639 summary only; no per-trial numerical result is reported.",
    )
    with pytest.raises(IngestionProvenanceError, match="per-trial source"):
        summary_only.validate()

    wrong_host = EvidenceIngestionRecord(
        row_id="fda_wrong_host",
        source_type="fda_review",
        regulatory_id="NDA 020639",
        url="https://example.org/nda/020639.cfm",
        access_statement="Public FDA Drugs@FDA statistical review package with per-trial results.",
        source_text="NDA 020639 Study 301 result: hazard ratio 0.82, 95% confidence interval 0.70 to 0.96.",
    )
    with pytest.raises(IngestionProvenanceError, match="fda.gov"):
        wrong_host.validate()


def test_regulatory_review_helper_rejects_non_regulatory_source_type():
    with pytest.raises(IngestionProvenanceError, match="regulatory review rows"):
        record_from_regulatory_review_row(
            {
                "source_type": "pubmed_abstract",
                "regulatory_id": "NDA 020639",
                "url": "https://pubmed.ncbi.nlm.nih.gov/31535829/",
                "source_text": "Not a regulatory review row.",
            }
        )


def _valid_effect_record(**overrides):
    payload = {
        "record_id": "dapa_hf_hr_primary",
        "study_id": "DAPA-HF",
        "outcome_name": "worsening heart failure or cardiovascular death",
        "effect_type": "HR",
        "point_estimate": 0.74,
        "ci_lower": 0.65,
        "ci_upper": 0.85,
        "standard_error": None,
        "p_value": 0.001,
        "timepoint": "median 18.2 months",
        "is_primary": True,
        "is_subgroup": False,
        "computation_origin": "reported",
        "source": EvidenceIngestionRecord(
            row_id="pubmed_dapa_hf",
            source_type="pubmed_abstract",
            pmid="31535829",
            url="https://pubmed.ncbi.nlm.nih.gov/31535829/",
            access_statement="PubMed abstract metadata and abstract text.",
        ),
        "provenance": ExtractionProvenance(
            source_text="Dapagliflozin reduced worsening heart failure or cardiovascular death (hazard ratio, 0.74; 95% CI, 0.65 to 0.85; P<0.001).",
            source_type="text",
            char_start=120,
            char_end=245,
        ),
    }
    payload.update(overrides)
    return ProofCarryingEffectRecord(**payload)


def test_proof_carrying_effect_validates_and_exports_non_certifying_record():
    record = _valid_effect_record()
    exported = record.to_dict()

    assert record.is_meta_analysis_ready is True
    assert record.has_complete_ci is True
    assert exported["schema_version"] == "proof_carrying_effect/v1"
    assert exported["certification_effect"] == "none"
    assert exported["integrity_hash"] == record.integrity_hash
    assert len(record.integrity_hash) == 64


def test_proof_carrying_effect_integrity_hash_includes_extraction_method_context():
    reported = _valid_effect_record(computation_origin="reported")
    computed = _valid_effect_record(computation_origin="computed")

    assert reported.integrity_hash != computed.integrity_hash


def test_proof_effect_from_extractor_row_accepts_source_backed_open_access_record():
    raw = {
        "benchmark_id": "author_meta_trial_00002",
        "study_id": "PMC3196245",
        "pmcid": "PMC3196245",
        "pmid": "22008217",
        "oa_download_status": "downloaded_fallback_pmcid_direct",
        "oa_download_url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC3196245&blobtype=pdf",
        "outcome_name": "all-cause mortality",
        "effect_type": "OR",
        "point_estimate": 1.05,
        "ci_lower": 0.90,
        "ci_upper": 1.22,
        "p_value": 0.51,
        "is_primary": "true",
        "provenance": {
            "source_text": "The odds ratio was 1.05 with 95% CI 0.90 to 1.22.",
            "source_type": "text",
            "page_number": 4,
            "char_start": 10,
            "char_end": 62,
        },
    }

    record = proof_effect_from_extractor_row(raw)
    record.validate()

    assert record.effect_type == "OR"
    assert record.source.pmcid == "PMC3196245"
    assert record.is_primary is True


def test_proof_effect_from_extractor_row_preserves_zero_values():
    raw = {
        "benchmark_id": "author_meta_trial_zero_rd",
        "study_id": "PMC3196245",
        "pmcid": "PMC3196245",
        "pmid": "22008217",
        "oa_download_status": "downloaded_fallback_pmcid_direct",
        "oa_download_url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC3196245&blobtype=pdf",
        "outcome_name": "risk difference outcome",
        "effect_type": "RD",
        "point_estimate": 0.0,
        "standard_error": 0.1,
        "p_value": 0.0,
        "is_subgroup": "false",
        "provenance": {
            "source_text": "The reported risk difference was 0.0 with standard error 0.1 and p=0.0.",
            "source_type": "text",
            "char_start": 0,
            "char_end": 72,
        },
    }

    record = proof_effect_from_extractor_row(raw)
    record.validate()

    assert record.point_estimate == 0.0
    assert record.p_value == 0.0
    assert record.is_subgroup is False


def test_proof_carrying_effect_rejects_missing_uncertainty_and_bad_ci():
    missing_uncertainty = _valid_effect_record(ci_lower=None, ci_upper=None, standard_error=None)
    with pytest.raises(IngestionProvenanceError, match="complete CI or a standard error"):
        missing_uncertainty.validate()

    outside_ci = _valid_effect_record(point_estimate=0.9)
    with pytest.raises(IngestionProvenanceError, match="inside confidence interval"):
        outside_ci.validate()


def test_proof_carrying_effect_rejects_negative_ratio_and_weak_provenance():
    negative_hr = _valid_effect_record(point_estimate=-0.74, ci_lower=-0.85, ci_upper=-0.65)
    with pytest.raises(IngestionProvenanceError, match="ratio-scale point_estimate"):
        negative_hr.validate()

    no_snippet = _valid_effect_record(
        provenance=ExtractionProvenance(source_text="", source_type="text")
    )
    with pytest.raises(IngestionProvenanceError, match="source_text"):
        no_snippet.validate()


def test_proof_carrying_effect_rejects_figure_without_label_and_duplicate_ids():
    figure_without_label = _valid_effect_record(
        provenance=ExtractionProvenance(
            source_text="Kaplan-Meier curve supports the extracted HR.",
            source_type="figure",
            page_number=5,
        )
    )
    with pytest.raises(IngestionProvenanceError, match="figure_label"):
        figure_without_label.validate()

    first = _valid_effect_record()
    duplicate = _valid_effect_record()
    with pytest.raises(IngestionProvenanceError, match="record_id must be unique"):
        validate_proof_carrying_effects([first, duplicate])
