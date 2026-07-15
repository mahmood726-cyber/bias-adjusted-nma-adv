"""Source-backed ingestion provenance checks for extracted evidence rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from urllib.parse import unquote, urlparse

from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources


class IngestionProvenanceError(ValueError):
    """Raised when an extracted row lacks admissible source provenance."""


_NCT_RE = re.compile(r"^NCT\d{8}$")
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)
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
