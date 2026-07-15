"""Source-backed ingestion provenance checks for extracted evidence rows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import unquote, urlparse

from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources


class IngestionProvenanceError(ValueError):
    """Raised when an extracted row lacks admissible source provenance."""


_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


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
    source_text: str = ""

    def validate(self) -> None:
        """Validate source type, identifier shape, URL identity, and OA proof."""

        if not self.row_id.strip():
            raise IngestionProvenanceError("row_id must not be empty.")
        _validate_url(self.url)
        self._validate_identifier_shapes()

        if self.source_type == "clinicaltrials_gov":
            self._validate_clinicaltrials_source()
        elif self.source_type == "pubmed_abstract":
            self._validate_pubmed_source()
        elif self.source_type == "open_access_paper":
            self._validate_open_access_source()
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

    def _validate_clinicaltrials_source(self) -> None:
        if self.nct_id is None:
            raise IngestionProvenanceError(f"{self.row_id}: ClinicalTrials.gov row needs nct_id.")
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
        host = _host(self.url)
        if host != "clinicaltrials.gov" and not host.endswith(".clinicaltrials.gov"):
            raise IngestionProvenanceError(
                f"{self.row_id}: ClinicalTrials.gov source URL must use clinicaltrials.gov."
            )
        if self.nct_id.lower() not in _normalised_url_text(self.url):
            raise IngestionProvenanceError(
                f"{self.row_id}: ClinicalTrials.gov URL must contain {self.nct_id}."
            )

    def _validate_pubmed_source(self) -> None:
        if self.pmid is None:
            raise IngestionProvenanceError(f"{self.row_id}: PubMed abstract row needs pmid.")
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

        if not self._has_verifiable_article_identity():
            raise IngestionProvenanceError(
                f"{self.row_id}: open-access URL/source text does not contain a "
                "verifiable article identity token matching the claimed PMID, PMCID, or DOI."
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
        return tuple(tokens)


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


def validate_ingestion_records(records: list[EvidenceIngestionRecord]) -> None:
    """Validate a non-empty batch of ingestion records."""

    if not records:
        raise IngestionProvenanceError("at least one ingestion record is required.")
    for record in records:
        record.validate()


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


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
