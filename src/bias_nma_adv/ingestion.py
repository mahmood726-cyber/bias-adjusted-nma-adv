"""Source-backed ingestion provenance checks for extracted evidence rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from urllib.parse import unquote, urlparse

from bias_nma_adv.evidence_sources import (
    EFFECT_EVIDENCE_SOURCE_TYPES,
    PROTOCOL_ONLY_SOURCE_TYPES,
    EvidenceSource,
    EvidenceSourceError,
    REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES,
    REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES,
    validate_sources,
)


class IngestionProvenanceError(ValueError):
    """Raised when an extracted row lacks admissible source provenance."""


_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)
_REGISTRY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{2,127}$")
_REGULATORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/ -]{2,127}$")
_NUMERIC_RESULT_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
PROOF_CARRYING_EFFECT_SCHEMA_VERSION = "proof_carrying_effect/v1"

ALLOWED_EFFECT_TYPES = {
    "HR",
    "OR",
    "RR",
    "IRR",
    "GMR",
    "NNT",
    "NNH",
    "MD",
    "SMD",
    "ARD",
    "ARR",
    "RRR",
    "RD",
    "WMD",
}
RATIO_EFFECT_TYPES = {"HR", "OR", "RR", "IRR", "GMR"}
ALLOWED_PROVENANCE_SOURCE_TYPES = {"text", "table", "figure", "ocr", "computed"}
ALLOWED_COMPUTATION_ORIGINS = {"reported", "computed"}


def summarize_proof_carrying_ingestion_contract() -> dict[str, object]:
    """Summarize the static contract for validation-status reports."""

    return {
        "schema_version": PROOF_CARRYING_EFFECT_SCHEMA_VERSION,
        "allowed_effect_types": sorted(ALLOWED_EFFECT_TYPES),
        "allowed_effect_source_types": sorted(EFFECT_EVIDENCE_SOURCE_TYPES),
        "registry_result_source_types": sorted(REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES),
        "regulatory_review_source_types": sorted(REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES),
        "protocol_only_source_types": sorted(PROTOCOL_ONLY_SOURCE_TYPES),
        "protocol_sources_can_supply_model_effects": False,
        "protocol_sources_can_supply_registered_primary_outcomes": True,
        "protocol_sources_can_supply_completeness_denominators": True,
        "registry_result_sources_can_supply_model_effects": True,
        "registry_result_sources_require_numeric_result_text": True,
        "regulatory_review_sources_can_supply_model_effects": True,
        "regulatory_review_sources_require_numeric_result_text": True,
        "ratio_effect_types": sorted(RATIO_EFFECT_TYPES),
        "allowed_provenance_source_types": sorted(ALLOWED_PROVENANCE_SOURCE_TYPES),
        "allowed_computation_origins": sorted(ALLOWED_COMPUTATION_ORIGINS),
        "required_uncertainty": "complete_ci_or_standard_error",
        "requires_source_snippet": True,
        "requires_source_identity": True,
        "certification_effect": "none",
    }


@dataclass(frozen=True)
class ExtractionProvenance:
    """Source location and snippet supporting one extracted number."""

    source_text: str
    source_type: str
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    figure_label: str | None = None
    table_label: str | None = None

    def validate(self, row_id: str) -> None:
        if not self.source_text.strip():
            raise IngestionProvenanceError(f"{row_id}: provenance source_text must not be empty.")
        if self.source_type not in ALLOWED_PROVENANCE_SOURCE_TYPES:
            raise IngestionProvenanceError(
                f"{row_id}: unsupported provenance source_type '{self.source_type}'."
            )
        if self.page_number is not None and self.page_number < 0:
            raise IngestionProvenanceError(f"{row_id}: provenance page_number must be non-negative.")
        if self.char_start is not None and self.char_start < 0:
            raise IngestionProvenanceError(f"{row_id}: provenance char_start must be non-negative.")
        if self.char_end is not None and self.char_end < 0:
            raise IngestionProvenanceError(f"{row_id}: provenance char_end must be non-negative.")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise IngestionProvenanceError(
                f"{row_id}: provenance char_end must be greater than char_start."
            )
        if self.source_type == "figure" and not (self.figure_label or "").strip():
            raise IngestionProvenanceError(f"{row_id}: figure provenance requires figure_label.")
        if self.source_type == "table" and not (self.table_label or "").strip():
            raise IngestionProvenanceError(f"{row_id}: table provenance requires table_label.")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_text": self.source_text,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "figure_label": self.figure_label,
            "table_label": self.table_label,
        }


@dataclass(frozen=True)
class EvidenceIngestionRecord:
    """One extracted row plus the identifiers needed to verify its source.

    The record is deliberately source-focused. Extracted effect sizes can live
    elsewhere; this object answers whether the row is admissible evidence under
    the project boundary before any model consumes it.
    """

    row_id: str
    source_type: str
    url: str
    access_statement: str
    pmid: str | None = None
    nct_id: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    registry_id: str | None = None
    regulatory_id: str | None = None
    source_text: str = ""

    def validate(self) -> None:
        """Validate source type, identifier shape, URL identity, and OA proof."""

        if not self.row_id.strip():
            raise IngestionProvenanceError("row_id must not be empty.")
        _validate_url(self.url)
        self._validate_identifier_shapes()

        if self.source_type == "clinicaltrials_gov":
            self._validate_clinicaltrials_source()
        elif self.source_type == "aact_clinicaltrials_gov":
            self._validate_aact_clinicaltrials_source()
        elif self.source_type == "pubmed_abstract":
            self._validate_pubmed_source()
        elif self.source_type == "open_access_paper":
            self._validate_open_access_source()
        elif self.source_type in REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES:
            self._validate_registry_result_source()
        elif self.source_type in REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES:
            self._validate_regulatory_review_source()
        elif self.source_type in PROTOCOL_ONLY_SOURCE_TYPES:
            raise IngestionProvenanceError(
                f"{self.row_id}: source_type '{self.source_type}' is protocol-only and "
                "cannot supply a model-ready extracted effect."
            )
        else:
            raise IngestionProvenanceError(
                f"source_type '{self.source_type}' is not allowed for ingestion."
            )

    def _validate_identifier_shapes(self) -> None:
        if self.nct_id is not None and not _NCT_RE.match(self.nct_id):
            raise IngestionProvenanceError(
                f"{self.row_id}: malformed ClinicalTrials.gov identifier '{self.nct_id}'."
            )
        if self.pmid is not None and not _PMID_RE.match(self.pmid):
            raise IngestionProvenanceError(f"{self.row_id}: malformed PMID '{self.pmid}'.")
        if self.pmcid is not None and not _PMCID_RE.match(self.pmcid):
            raise IngestionProvenanceError(f"{self.row_id}: malformed PMCID '{self.pmcid}'.")
        if self.doi is not None and not _normalise_doi(self.doi):
            raise IngestionProvenanceError(f"{self.row_id}: malformed DOI '{self.doi}'.")
        if self.registry_id is not None and not _REGISTRY_ID_RE.match(self.registry_id):
            raise IngestionProvenanceError(
                f"{self.row_id}: malformed registry identifier '{self.registry_id}'."
            )
        if self.regulatory_id is not None and not _REGULATORY_ID_RE.match(self.regulatory_id):
            raise IngestionProvenanceError(
                f"{self.row_id}: malformed regulatory identifier '{self.regulatory_id}'."
            )

    def _validate_clinicaltrials_source(self) -> None:
        if self.nct_id is None:
            raise IngestionProvenanceError(f"{self.row_id}: ClinicalTrials.gov row needs nct_id.")
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type="clinicaltrials_gov",
                        identifier=self.nct_id,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc
        host = _host(self.url)
        if host != "clinicaltrials.gov" and not host.endswith(".clinicaltrials.gov"):
            raise IngestionProvenanceError(
                f"{self.row_id}: ClinicalTrials.gov source URL must use clinicaltrials.gov."
            )
        if self.nct_id.lower() not in _normalised_url_text(self.url):
            raise IngestionProvenanceError(
                f"{self.row_id}: ClinicalTrials.gov URL must contain {self.nct_id}."
            )

    def _validate_aact_clinicaltrials_source(self) -> None:
        if self.nct_id is None:
            raise IngestionProvenanceError(
                f"{self.row_id}: AACT ClinicalTrials.gov row needs nct_id."
            )
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type="aact_clinicaltrials_gov",
                        identifier=self.nct_id,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc
        host = _host(self.url)
        allowed_host = (
            host == "aact.ctti-clinicaltrials.org"
            or host.endswith(".aact.ctti-clinicaltrials.org")
            or host == "clinicaltrials.gov"
            or host.endswith(".clinicaltrials.gov")
        )
        if not allowed_host:
            raise IngestionProvenanceError(
                f"{self.row_id}: AACT source URL must use AACT or ClinicalTrials.gov."
            )
        haystacks = (
            _normalised_url_text(self.url),
            _normalise_free_text(self.source_text),
        )
        if self.nct_id.lower() not in "".join(haystacks):
            raise IngestionProvenanceError(
                f"{self.row_id}: AACT source must contain {self.nct_id} in the URL or row text."
            )

    def _validate_pubmed_source(self) -> None:
        if self.pmid is None:
            raise IngestionProvenanceError(f"{self.row_id}: PubMed abstract row needs pmid.")
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type="pubmed_abstract",
                        identifier=self.pmid,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc
        host = _host(self.url)
        if host != "pubmed.ncbi.nlm.nih.gov":
            raise IngestionProvenanceError(
                f"{self.row_id}: PubMed abstract source URL must use pubmed.ncbi.nlm.nih.gov."
            )
        if self.pmid not in _normalised_url_text(self.url):
            raise IngestionProvenanceError(f"{self.row_id}: PubMed URL must contain PMID {self.pmid}.")

    def _validate_open_access_source(self) -> None:
        if not any((self.pmid, self.pmcid, self.doi)):
            raise IngestionProvenanceError(
                f"{self.row_id}: open-access paper row needs PMID, PMCID, or DOI."
            )
        identifier = self.pmcid or self.doi or self.pmid
        assert identifier is not None
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type="open_access_paper",
                        identifier=identifier,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc

        if not self._has_verifiable_article_identity():
            raise IngestionProvenanceError(
                f"{self.row_id}: open-access URL/source text does not contain a "
                "verifiable article identity token matching the claimed PMID, PMCID, or DOI."
            )

    def _validate_registry_result_source(self) -> None:
        if self.registry_id is None:
            raise IngestionProvenanceError(
                f"{self.row_id}: registry result row needs registry_id."
            )
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type=self.source_type,
                        identifier=self.registry_id,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc
        if self.source_type == "who_ictrp_results":
            host = _host(self.url)
            if host != "trialsearch.who.int" and not host.endswith(".who.int"):
                raise IngestionProvenanceError(
                    f"{self.row_id}: WHO ICTRP result URL must use trialsearch.who.int or who.int."
                )
        if self.source_type == "pactr_results":
            host = _host(self.url)
            allowed_host = (
                host == "pactr.samrc.ac.za"
                or host.endswith(".pactr.samrc.ac.za")
                or host == "pactr.org"
                or host.endswith(".pactr.org")
            )
            if not allowed_host:
                raise IngestionProvenanceError(
                    f"{self.row_id}: PACTR result URL must use a PACTR registry host."
                )
        if not self._has_verifiable_registry_identity():
            raise IngestionProvenanceError(
                f"{self.row_id}: registry result URL/source text does not contain "
                f"the claimed registry_id {self.registry_id}."
            )
        self._validate_registry_result_text()

    def _has_verifiable_registry_identity(self) -> bool:
        if self.registry_id is None:
            return False
        token = _normalise_free_text(self.registry_id)
        haystacks = (
            _normalised_url_text(self.url),
            _normalise_free_text(self.source_text),
        )
        return any(token and token in haystack for haystack in haystacks)

    def _validate_registry_result_text(self) -> None:
        text = f"{self.access_statement}\n{self.source_text}".lower()
        if any(
            token in text
            for token in (
                "results available: no",
                "results available no",
                "no results available",
                "results not available",
                "not yet reported",
                "not reported",
                "protocol only",
                "protocol-only",
            )
        ):
            raise IngestionProvenanceError(
                f"{self.row_id}: registry row is not a result-level evidence source."
            )
        if not any(
            token in text
            for token in (
                "results available: yes",
                "results available yes",
                "posted result",
                "reported result",
                "result-level",
                "outcome result",
                "numeric outcome",
                "classification outcome",
                "model-ready",
                "model ready",
            )
        ):
            raise IngestionProvenanceError(
                f"{self.row_id}: registry result source must explicitly state result availability."
            )
        numeric_terms = (
            "hazard ratio",
            "odds ratio",
            "risk ratio",
            "relative risk",
            "confidence interval",
            "standard error",
            "events",
            "event count",
            "participants",
            "mean",
            "sd",
            "tp",
            "fp",
            "fn",
            "tn",
            "sensitivity",
            "specificity",
        )
        if not _NUMERIC_RESULT_RE.search(self.source_text) or not any(
            term in text for term in numeric_terms
        ):
            raise IngestionProvenanceError(
                f"{self.row_id}: registry result source must contain numeric model-ready result text."
            )

    def _validate_regulatory_review_source(self) -> None:
        if self.regulatory_id is None:
            raise IngestionProvenanceError(
                f"{self.row_id}: regulatory review row needs regulatory_id."
            )
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type=self.source_type,
                        identifier=self.regulatory_id,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc
        if not self._has_verifiable_regulatory_identity():
            raise IngestionProvenanceError(
                f"{self.row_id}: regulatory review URL/source text does not contain "
                f"the claimed regulatory_id {self.regulatory_id}."
            )
        self._validate_regulatory_result_text()

    def _has_verifiable_regulatory_identity(self) -> bool:
        if self.regulatory_id is None:
            return False
        token = _normalise_free_text(self.regulatory_id)
        haystacks = (
            _normalised_url_text(self.url),
            _normalise_free_text(self.source_text),
        )
        return any(token and token in haystack for haystack in haystacks)

    def _validate_regulatory_result_text(self) -> None:
        text = f"{self.access_statement}\n{self.source_text}".lower()
        if any(
            token in text
            for token in (
                "summary only",
                "aggregate only",
                "no per-trial",
                "no per trial",
                "results available: no",
                "results not available",
                "not reported",
                "protocol only",
                "protocol-only",
            )
        ):
            raise IngestionProvenanceError(
                f"{self.row_id}: regulatory review row is not a model-ready per-trial source."
            )
        if not any(
            token in text
            for token in (
                "study",
                "trial",
                "per-trial",
                "per trial",
                "result",
                "outcome",
                "table",
            )
        ):
            raise IngestionProvenanceError(
                f"{self.row_id}: regulatory review source must identify trial-level results."
            )
        numeric_terms = (
            "hazard ratio",
            "odds ratio",
            "risk ratio",
            "relative risk",
            "confidence interval",
            "standard error",
            "events",
            "event count",
            "participants",
            "mean",
            "sd",
            "tp",
            "fp",
            "fn",
            "tn",
            "sensitivity",
            "specificity",
            "least squares mean",
            "ls mean",
        )
        if not _NUMERIC_RESULT_RE.search(self.source_text) or not any(
            term in text for term in numeric_terms
        ):
            raise IngestionProvenanceError(
                f"{self.row_id}: regulatory review source must contain numeric model-ready result text."
            )

    def _has_verifiable_article_identity(self) -> bool:
        haystacks = (
            _normalised_url_text(self.url),
            _normalise_free_text(self.source_text),
        )
        tokens = self._identity_tokens()
        return any(token and token in haystack for token in tokens for haystack in haystacks)

    def _identity_tokens(self) -> tuple[str, ...]:
        tokens: list[str] = []
        if self.pmid:
            tokens.append(self.pmid.lower())
            tokens.append(f"pmid{self.pmid.lower()}")
        if self.pmcid:
            tokens.append(self.pmcid.lower())
        if self.doi:
            doi = _normalise_doi(self.doi)
            if doi:
                tokens.append(doi)
                tokens.append(doi.replace("/", ""))
        if self.registry_id:
            tokens.append(_normalise_free_text(self.registry_id))
        if self.regulatory_id:
            tokens.append(_normalise_free_text(self.regulatory_id))
        return tuple(tokens)


@dataclass(frozen=True)
class ProtocolOnlyRegistryRecord:
    """Protocol metadata row that can support bias ledgers but not effect estimates."""

    row_id: str
    source_type: str
    registry_id: str
    url: str
    access_statement: str
    registered_primary_outcome: str
    reported_primary_outcome: str | None = None
    status: str = "unknown"
    interventions: tuple[str, ...] = ()
    source_text: str = ""

    @property
    def model_ready_effect(self) -> bool:
        return False

    def validate(self) -> None:
        if not self.row_id.strip():
            raise IngestionProvenanceError("protocol row_id must not be empty.")
        if self.source_type not in PROTOCOL_ONLY_SOURCE_TYPES:
            raise IngestionProvenanceError(
                f"{self.row_id}: source_type must be protocol-only, got {self.source_type!r}."
            )
        if not _REGISTRY_ID_RE.match(self.registry_id):
            raise IngestionProvenanceError(
                f"{self.row_id}: malformed registry identifier '{self.registry_id}'."
            )
        if not self.registered_primary_outcome.strip():
            raise IngestionProvenanceError(
                f"{self.row_id}: registered_primary_outcome must not be empty."
            )
        try:
            validate_sources(
                [
                    EvidenceSource(
                        source_type=self.source_type,
                        identifier=self.registry_id,
                        url=self.url,
                        access_statement=self.access_statement,
                    )
                ]
            )
        except EvidenceSourceError as exc:
            raise IngestionProvenanceError(f"{self.row_id}: {exc}") from exc
        if not self._has_verifiable_protocol_identity():
            raise IngestionProvenanceError(
                f"{self.row_id}: protocol URL/source text does not contain "
                f"the claimed registry_id {self.registry_id}."
            )

    def outcome_switching_score(self) -> float | None:
        """Return a simple protocol/publication primary-outcome switch score."""

        if self.reported_primary_outcome is None or not self.reported_primary_outcome.strip():
            return None
        registered = self.registered_primary_outcome.strip().lower()
        reported = self.reported_primary_outcome.strip().lower()
        if registered != reported and registered not in reported and reported not in registered:
            return 1.0
        return 0.0

    def _has_verifiable_protocol_identity(self) -> bool:
        token = _normalise_free_text(self.registry_id)
        haystacks = (
            _normalised_url_text(self.url),
            _normalise_free_text(self.source_text),
        )
        return any(token and token in haystack for haystack in haystacks)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "row_id": self.row_id,
            "source_type": self.source_type,
            "registry_id": self.registry_id,
            "url": self.url,
            "access_statement": self.access_statement,
            "registered_primary_outcome": self.registered_primary_outcome,
            "reported_primary_outcome": self.reported_primary_outcome,
            "status": self.status,
            "interventions": list(self.interventions),
            "model_ready_effect": self.model_ready_effect,
            "outcome_switching_score": self.outcome_switching_score(),
        }


@dataclass(frozen=True)
class ProofCarryingEffectRecord:
    """One model-ready extracted effect with source provenance and checks."""

    record_id: str
    study_id: str
    outcome_name: str
    effect_type: str
    point_estimate: float
    source: EvidenceIngestionRecord
    provenance: ExtractionProvenance
    ci_lower: float | None = None
    ci_upper: float | None = None
    standard_error: float | None = None
    p_value: float | None = None
    timepoint: str | None = None
    is_primary: bool | None = None
    is_subgroup: bool = False
    computation_origin: str = "reported"

    def validate(self) -> None:
        if not self.record_id.strip():
            raise IngestionProvenanceError("record_id must not be empty.")
        if not self.study_id.strip():
            raise IngestionProvenanceError(f"{self.record_id}: study_id must not be empty.")
        if not self.outcome_name.strip():
            raise IngestionProvenanceError(f"{self.record_id}: outcome_name must not be empty.")
        if self.effect_type not in ALLOWED_EFFECT_TYPES:
            raise IngestionProvenanceError(
                f"{self.record_id}: unsupported effect_type '{self.effect_type}'."
            )
        if self.computation_origin not in ALLOWED_COMPUTATION_ORIGINS:
            raise IngestionProvenanceError(
                f"{self.record_id}: unsupported computation_origin '{self.computation_origin}'."
            )
        self.source.validate()
        self.provenance.validate(self.record_id)
        self._validate_numeric_fields()

    @property
    def has_complete_ci(self) -> bool:
        return self.ci_lower is not None and self.ci_upper is not None

    @property
    def is_meta_analysis_ready(self) -> bool:
        try:
            self.validate()
        except IngestionProvenanceError:
            return False
        return True

    @property
    def integrity_hash(self) -> str:
        payload = {
            "schema_version": PROOF_CARRYING_EFFECT_SCHEMA_VERSION,
            "record_id": self.record_id,
            "study_id": self.study_id,
            "outcome_name": self.outcome_name,
            "effect_type": self.effect_type,
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "standard_error": self.standard_error,
            "p_value": self.p_value,
            "timepoint": self.timepoint,
            "is_primary": self.is_primary,
            "is_subgroup": self.is_subgroup,
            "computation_origin": self.computation_origin,
            "source": self.source.__dict__,
            "provenance": self.provenance.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": PROOF_CARRYING_EFFECT_SCHEMA_VERSION,
            "record_id": self.record_id,
            "study_id": self.study_id,
            "outcome_name": self.outcome_name,
            "effect_type": self.effect_type,
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "standard_error": self.standard_error,
            "p_value": self.p_value,
            "timepoint": self.timepoint,
            "is_primary": self.is_primary,
            "is_subgroup": self.is_subgroup,
            "computation_origin": self.computation_origin,
            "source": self.source.__dict__,
            "provenance": self.provenance.to_dict(),
            "integrity_hash": self.integrity_hash,
            "certification_effect": "none",
        }

    def _validate_numeric_fields(self) -> None:
        point = _finite_float(self.point_estimate, f"{self.record_id}: point_estimate")
        if self.effect_type in RATIO_EFFECT_TYPES and point <= 0:
            raise IngestionProvenanceError(
                f"{self.record_id}: ratio-scale point_estimate must be positive."
            )

        has_lower = self.ci_lower is not None
        has_upper = self.ci_upper is not None
        if has_lower != has_upper:
            raise IngestionProvenanceError(
                f"{self.record_id}: ci_lower and ci_upper must be supplied together."
            )
        if self.has_complete_ci:
            assert self.ci_lower is not None and self.ci_upper is not None
            lower = _finite_float(self.ci_lower, f"{self.record_id}: ci_lower")
            upper = _finite_float(self.ci_upper, f"{self.record_id}: ci_upper")
            if upper <= lower:
                raise IngestionProvenanceError(f"{self.record_id}: confidence interval is not ordered.")
            if not lower <= point <= upper:
                raise IngestionProvenanceError(
                    f"{self.record_id}: point_estimate must lie inside confidence interval."
                )
            if self.effect_type in RATIO_EFFECT_TYPES and lower <= 0:
                raise IngestionProvenanceError(
                    f"{self.record_id}: ratio-scale confidence interval must be positive."
                )

        if self.standard_error is not None:
            se = _finite_float(self.standard_error, f"{self.record_id}: standard_error")
            if se <= 0:
                raise IngestionProvenanceError(f"{self.record_id}: standard_error must be positive.")
        if not self.has_complete_ci and self.standard_error is None:
            raise IngestionProvenanceError(
                f"{self.record_id}: extracted effect requires either a complete CI or a standard error."
            )
        if self.p_value is not None:
            p_value = _finite_float(self.p_value, f"{self.record_id}: p_value")
            if not 0 <= p_value <= 1:
                raise IngestionProvenanceError(f"{self.record_id}: p_value must be between 0 and 1.")


def record_from_extractor_row(raw: dict[str, object]) -> EvidenceIngestionRecord:
    """Create an ingestion record from an rct-extractor-style benchmark row."""

    snapshot = raw.get("model_snapshot_best") if isinstance(raw.get("model_snapshot_best"), dict) else {}
    source_text = str(snapshot.get("source_text", "")) if isinstance(snapshot, dict) else ""
    status = str(raw.get("oa_download_status", "")).strip()
    access_statement = (
        f"Open access paper candidate from extraction bundle; OA download status: {status}."
    )
    return EvidenceIngestionRecord(
        row_id=str(raw.get("benchmark_id") or raw.get("study_id") or "unknown_row"),
        source_type="open_access_paper",
        url=str(raw.get("oa_download_url", "")),
        access_statement=access_statement,
        pmid=_clean_optional(raw.get("pmid")),
        pmcid=_clean_optional(raw.get("pmcid")),
        doi=_clean_optional(raw.get("doi")),
        source_text=source_text,
    )


def proof_effect_from_extractor_row(raw: dict[str, object]) -> ProofCarryingEffectRecord:
    """Create a proof-carrying effect record from an extractor-style row."""

    provenance_raw = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    snapshot = raw.get("model_snapshot_best") if isinstance(raw.get("model_snapshot_best"), dict) else {}
    source_text = str(
        provenance_raw.get("source_text")
        or (snapshot.get("source_text") if isinstance(snapshot, dict) else "")
        or ""
    )
    provenance = ExtractionProvenance(
        source_text=source_text,
        source_type=str(provenance_raw.get("source_type") or "text"),
        page_number=_optional_int(provenance_raw.get("page_number")),
        char_start=_optional_int(provenance_raw.get("char_start")),
        char_end=_optional_int(provenance_raw.get("char_end")),
        figure_label=_clean_optional(provenance_raw.get("figure_label")),
        table_label=_clean_optional(provenance_raw.get("table_label")),
    )
    return ProofCarryingEffectRecord(
        record_id=str(raw.get("record_id") or raw.get("benchmark_id") or raw.get("study_id") or "unknown_row"),
        study_id=str(raw.get("study_id") or raw.get("benchmark_id") or "unknown_study"),
        outcome_name=str(raw.get("outcome_name") or raw.get("outcome") or "unknown_outcome"),
        effect_type=str(raw.get("effect_type") or raw.get("measure") or ""),
        point_estimate=float(_first_present(raw, "point_estimate", "effect", default=math.nan)),
        ci_lower=_optional_float(raw.get("ci_lower")),
        ci_upper=_optional_float(raw.get("ci_upper")),
        standard_error=_optional_float(_first_present(raw, "standard_error", "se")),
        p_value=_optional_float(raw.get("p_value")),
        timepoint=_clean_optional(raw.get("timepoint")),
        is_primary=_optional_bool(raw.get("is_primary")),
        is_subgroup=_optional_bool(raw.get("is_subgroup")) or False,
        computation_origin=str(raw.get("computation_origin") or "reported"),
        source=record_from_extractor_row(raw),
        provenance=provenance,
    )


def record_from_registry_result_row(raw: dict[str, object]) -> EvidenceIngestionRecord:
    """Create an ingestion record from a downloaded WHO ICTRP or PACTR result row.

    Registry downloads are usually protocol metadata. This helper therefore keeps
    the row fail-closed: callers must label the row as a result source and provide
    a registry identifier, public URL, result-availability statement, and source
    text containing numeric outcome/effect information.
    """

    source_type = str(
        _first_present(raw, "source_type", "registry_source_type", default="")
    ).strip()
    if source_type not in REGISTRY_RESULT_EVIDENCE_SOURCE_TYPES:
        raise IngestionProvenanceError(
            "registry result rows must use source_type 'who_ictrp_results' or 'pactr_results'."
        )
    registry_id = _clean_optional(
        _first_present(raw, "registry_id", "trial_id", "TrialID", "trialid", "id")
    )
    url = str(_first_present(raw, "url", "trial_url", "public_url", "source_url", default=""))
    source_text = str(
        _first_present(
            raw,
            "source_text",
            "result_text",
            "results_text",
            "outcome_results",
            "results_summary",
            default="",
        )
        or ""
    )
    access_statement = str(
        _first_present(
            raw,
            "access_statement",
            "result_access_statement",
            default="Downloaded public registry row with posted result-level outcome data.",
        )
    )
    return EvidenceIngestionRecord(
        row_id=str(_first_present(raw, "row_id", "record_id", "registry_id", default="registry_row")),
        source_type=source_type,
        url=url,
        access_statement=access_statement,
        registry_id=registry_id,
        source_text=source_text,
    )


def proof_effect_from_registry_result_row(raw: dict[str, object]) -> ProofCarryingEffectRecord:
    """Create a proof-carrying effect from a downloaded registry result row."""

    source = record_from_registry_result_row(raw)
    provenance = ExtractionProvenance(
        source_text=source.source_text,
        source_type=str(_first_present(raw, "provenance_source_type", "source_origin", default="text")),
        page_number=_optional_int(raw.get("page_number")),
        char_start=_optional_int(raw.get("char_start")),
        char_end=_optional_int(raw.get("char_end")),
        figure_label=_clean_optional(raw.get("figure_label")),
        table_label=_clean_optional(raw.get("table_label")),
    )
    return ProofCarryingEffectRecord(
        record_id=str(_first_present(raw, "record_id", "row_id", "registry_id", default="registry_effect")),
        study_id=str(_first_present(raw, "study_id", "registry_id", "trial_id", default="registry_study")),
        outcome_name=str(_first_present(raw, "outcome_name", "outcome", default="registry_outcome")),
        effect_type=str(_first_present(raw, "effect_type", "measure", default="")),
        point_estimate=float(_first_present(raw, "point_estimate", "effect", default=math.nan)),
        ci_lower=_optional_float(raw.get("ci_lower")),
        ci_upper=_optional_float(raw.get("ci_upper")),
        standard_error=_optional_float(_first_present(raw, "standard_error", "se")),
        p_value=_optional_float(raw.get("p_value")),
        timepoint=_clean_optional(raw.get("timepoint")),
        is_primary=_optional_bool(raw.get("is_primary")),
        is_subgroup=_optional_bool(raw.get("is_subgroup")) or False,
        computation_origin=str(raw.get("computation_origin") or "reported"),
        source=source,
        provenance=provenance,
    )


def record_from_protocol_registry_row(raw: dict[str, object]) -> ProtocolOnlyRegistryRecord:
    """Create a non-effect protocol metadata record from ICTRP/PACTR/registry rows."""

    source_type = str(_first_present(raw, "source_type", "registry_source_type", default="")).strip()
    registry_id = str(
        _first_present(raw, "registry_id", "trial_id", "TrialID", "trialid", "id", default="")
    ).strip()
    interventions_raw = _first_present(raw, "interventions", "intervention_names", default=())
    if isinstance(interventions_raw, str):
        interventions = tuple(
            item.strip().lower() for item in interventions_raw.split(";") if item.strip()
        )
    else:
        interventions = tuple(
            str(item).strip().lower() for item in interventions_raw or () if str(item).strip()
        )
    return ProtocolOnlyRegistryRecord(
        row_id=str(_first_present(raw, "row_id", "record_id", "registry_id", default="protocol_row")),
        source_type=source_type,
        registry_id=registry_id,
        url=str(_first_present(raw, "url", "trial_url", "public_url", "source_url", default="")),
        access_statement=str(
            _first_present(
                raw,
                "access_statement",
                "protocol_access_statement",
                default="Downloaded public registry protocol row.",
            )
        ),
        registered_primary_outcome=str(
            _first_present(
                raw,
                "registered_primary_outcome",
                "primary_outcome",
                "registered_primary",
                default="",
            )
        ),
        reported_primary_outcome=_clean_optional(
            _first_present(raw, "reported_primary_outcome", "reported_primary")
        ),
        status=str(_first_present(raw, "status", "recruitment_status", default="unknown")).strip().lower(),
        interventions=interventions,
        source_text=str(_first_present(raw, "source_text", "protocol_text", default="") or ""),
    )


def build_protocol_completeness_ledger(
    records: list[ProtocolOnlyRegistryRecord],
    *,
    reported_registry_ids: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a denominator ledger from protocol-only registry metadata.

    The ledger is intentionally not an evidence-effect bundle. It exposes
    registration denominators and primary-outcome anchors for bias review.
    """

    if not records:
        raise IngestionProvenanceError("at least one protocol registry record is required.")
    reported_ids = {_normalise_free_text(item) for item in reported_registry_ids}
    source_type_counts: dict[str, int] = {}
    denominator_records: list[ProtocolOnlyRegistryRecord] = []
    reported_records: list[ProtocolOnlyRegistryRecord] = []
    unreported_records: list[ProtocolOnlyRegistryRecord] = []
    registered_primary_outcomes: dict[str, str] = {}
    outcome_switching_scores: dict[str, float] = {}

    for record in records:
        record.validate()
        source_type_counts[record.source_type] = source_type_counts.get(record.source_type, 0) + 1
        registered_primary_outcomes[record.registry_id] = record.registered_primary_outcome
        switching_score = record.outcome_switching_score()
        if switching_score is not None:
            outcome_switching_scores[record.registry_id] = switching_score
        if record.status not in {"completed", "terminated", "unknown"}:
            continue
        denominator_records.append(record)
        is_reported = (
            _normalise_free_text(record.registry_id) in reported_ids
            or bool((record.reported_primary_outcome or "").strip())
        )
        if is_reported:
            reported_records.append(record)
        else:
            unreported_records.append(record)

    denominator = len(denominator_records)
    unreported_ratio = 0.0 if denominator == 0 else len(unreported_records) / denominator
    return {
        "schema_version": "protocol_completeness_ledger/v1",
        "model_ready_effect": False,
        "protocol_records_can_supply_model_effects": False,
        "protocol_records_can_supply_registered_primary_outcomes": True,
        "protocol_records_can_supply_completeness_denominators": True,
        "n_protocol_records": len(records),
        "n_denominator_records": denominator,
        "n_reported_records": len(reported_records),
        "n_unreported_records": len(unreported_records),
        "unreported_ratio": float(unreported_ratio),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "registered_primary_outcomes": dict(sorted(registered_primary_outcomes.items())),
        "outcome_switching_scores": dict(sorted(outcome_switching_scores.items())),
        "reported_registry_ids": sorted(reported_registry_ids),
        "unreported_registry_ids": sorted(record.registry_id for record in unreported_records),
    }


def record_from_regulatory_review_row(raw: dict[str, object]) -> EvidenceIngestionRecord:
    """Create an ingestion record from a public FDA review or EMA EPAR row.

    Regulatory documents are admissible only when the row is source-bound to a
    public review package and contains numeric per-trial result text. Summary-
    only review statements remain barred from model-ready effects.
    """

    source_type = str(_first_present(raw, "source_type", "regulatory_source_type", default="")).strip()
    if source_type not in REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES:
        raise IngestionProvenanceError(
            "regulatory review rows must use one of "
            f"{sorted(REGULATORY_REVIEW_EVIDENCE_SOURCE_TYPES)}."
        )
    regulatory_id = _clean_optional(
        _first_present(
            raw,
            "regulatory_id",
            "application_number",
            "application_id",
            "epar_id",
            "id",
        )
    )
    url = str(_first_present(raw, "url", "review_url", "public_url", "source_url", default=""))
    source_text = str(
        _first_present(
            raw,
            "source_text",
            "review_text",
            "result_text",
            "results_text",
            "outcome_results",
            "results_summary",
            default="",
        )
        or ""
    )
    access_statement = str(
        _first_present(
            raw,
            "access_statement",
            "review_access_statement",
            default="Downloaded public regulatory review row with per-trial numeric outcome data.",
        )
    )
    return EvidenceIngestionRecord(
        row_id=str(_first_present(raw, "row_id", "record_id", "regulatory_id", default="regulatory_row")),
        source_type=source_type,
        url=url,
        access_statement=access_statement,
        regulatory_id=regulatory_id,
        source_text=source_text,
    )


def proof_effect_from_regulatory_review_row(raw: dict[str, object]) -> ProofCarryingEffectRecord:
    """Create a proof-carrying effect from a public regulatory review row."""

    source = record_from_regulatory_review_row(raw)
    provenance = ExtractionProvenance(
        source_text=source.source_text,
        source_type=str(_first_present(raw, "provenance_source_type", "source_origin", default="text")),
        page_number=_optional_int(raw.get("page_number")),
        char_start=_optional_int(raw.get("char_start")),
        char_end=_optional_int(raw.get("char_end")),
        figure_label=_clean_optional(raw.get("figure_label")),
        table_label=_clean_optional(raw.get("table_label")),
    )
    return ProofCarryingEffectRecord(
        record_id=str(_first_present(raw, "record_id", "row_id", "regulatory_id", default="regulatory_effect")),
        study_id=str(_first_present(raw, "study_id", "trial_id", "regulatory_id", default="regulatory_study")),
        outcome_name=str(_first_present(raw, "outcome_name", "outcome", default="regulatory_outcome")),
        effect_type=str(_first_present(raw, "effect_type", "measure", default="")),
        point_estimate=float(_first_present(raw, "point_estimate", "effect", default=math.nan)),
        ci_lower=_optional_float(raw.get("ci_lower")),
        ci_upper=_optional_float(raw.get("ci_upper")),
        standard_error=_optional_float(_first_present(raw, "standard_error", "se")),
        p_value=_optional_float(raw.get("p_value")),
        timepoint=_clean_optional(raw.get("timepoint")),
        is_primary=_optional_bool(raw.get("is_primary")),
        is_subgroup=_optional_bool(raw.get("is_subgroup")) or False,
        computation_origin=str(raw.get("computation_origin") or "reported"),
        source=source,
        provenance=provenance,
    )


def validate_ingestion_records(records: list[EvidenceIngestionRecord]) -> None:
    """Validate a non-empty batch of ingestion records."""

    if not records:
        raise IngestionProvenanceError("at least one ingestion record is required.")
    for record in records:
        record.validate()


def validate_proof_carrying_effects(records: list[ProofCarryingEffectRecord]) -> None:
    """Validate a non-empty batch of proof-carrying extracted effects."""

    if not records:
        raise IngestionProvenanceError("at least one proof-carrying effect is required.")
    seen: set[str] = set()
    for record in records:
        if record.record_id in seen:
            raise IngestionProvenanceError(
                f"proof-carrying effect record_id must be unique: {record.record_id}"
            )
        seen.add(record.record_id)
        record.validate()


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _first_present(raw: dict[str, object], *keys: str, default: object | None = None) -> object | None:
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise IngestionProvenanceError(f"cannot parse boolean value {value!r}.")


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IngestionProvenanceError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise IngestionProvenanceError(f"{label} must be finite.")
    return number


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IngestionProvenanceError("url must be an absolute HTTP(S) URL.")


def _host(url: str) -> str:
    return urlparse(url).hostname.lower() if urlparse(url).hostname else ""


def _normalised_url_text(url: str) -> str:
    parsed = urlparse(url)
    return _normalise_free_text(" ".join([parsed.netloc, parsed.path, parsed.query]))


def _normalise_free_text(text: str) -> str:
    return re.sub(r"[^a-z0-9./]+", "", unquote(text).lower())


def _normalise_doi(doi: str) -> str:
    text = doi.strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.strip().rstrip(".,;")
    if not text.startswith("10.") or "/" not in text:
        return ""
    return text
