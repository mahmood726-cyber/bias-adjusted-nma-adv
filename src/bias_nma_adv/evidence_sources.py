"""Allowed evidence-source policy for real-world validation data."""

from __future__ import annotations

from dataclasses import dataclass
import re


class EvidenceSourceError(ValueError):
    """Raised when a source is outside the allowed evidence boundary."""


ALLOWED_SOURCE_TYPES = {
    "clinicaltrials_gov",
    "pubmed_abstract",
    "open_access_paper",
}

_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")


@dataclass(frozen=True)
class EvidenceSource:
    """Source metadata for one extracted clinical datum."""

    source_type: str
    identifier: str
    url: str
    access_statement: str

    def validate(self) -> None:
        if self.source_type not in ALLOWED_SOURCE_TYPES:
            raise EvidenceSourceError(
                f"source_type '{self.source_type}' is not allowed; use one of "
                f"{sorted(ALLOWED_SOURCE_TYPES)}."
            )
        if not self.identifier.strip():
            raise EvidenceSourceError("identifier must not be empty.")
        if not self.url.startswith(("https://", "http://")):
            raise EvidenceSourceError("url must be an absolute HTTP(S) URL.")
        if not self.access_statement.strip():
            raise EvidenceSourceError("access_statement must describe why the source is admissible.")

        if self.source_type == "clinicaltrials_gov" and not _NCT_RE.match(self.identifier):
            raise EvidenceSourceError("ClinicalTrials.gov identifiers must look like NCT01234567.")
        if self.source_type == "pubmed_abstract" and not _PMID_RE.match(self.identifier):
            raise EvidenceSourceError("PubMed abstract identifiers must be numeric PMIDs.")
        if self.source_type == "open_access_paper":
            statement = self.access_statement.lower()
            if "open access" not in statement and "oa" not in statement:
                raise EvidenceSourceError("open_access_paper sources must state open-access status.")


def validate_sources(sources: list[EvidenceSource]) -> None:
    """Validate a non-empty source list."""

    if not sources:
        raise EvidenceSourceError("at least one evidence source is required.")
    for source in sources:
        source.validate()
