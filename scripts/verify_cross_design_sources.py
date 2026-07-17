"""Verify cross-design reported-HR tokens against PubMed abstracts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_MANIFEST = Path("validation/cross_design/sglt2_rct_nrs_cross_design.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/sglt2_rct_nrs_cross_design_check.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-cross-design-verifier/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()
    except URLError as exc:
        raise RuntimeError(f"could not fetch {url}: {exc}") from exc


def abstract_from_pubmed_xml(payload: bytes) -> tuple[str, str]:
    root = ET.fromstring(payload)
    pmid = root.findtext(".//MedlineCitation/PMID") or ""
    abstract = " ".join(
        text.strip()
        for element in root.findall(".//AbstractText")
        for text in ["".join(element.itertext())]
        if text.strip()
    )
    return pmid, abstract


def normalise_text(text: str) -> str:
    text = text.lower().replace(",", "")
    text = text.replace("·", ".").replace(" ", "").replace("\u2009", "")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    return re.sub(r"\s+", " ", text)


def tokens_near_anchor(text: str, anchor: str, tokens: list[str], *, window: int = 520) -> tuple[bool, bool]:
    normalised = normalise_text(text)
    normalised_anchor = normalise_text(anchor)
    anchor_indices = [
        match.start()
        for match in re.finditer(re.escape(normalised_anchor), normalised)
    ]
    if not anchor_indices:
        return False, False
    normalised_tokens = [normalise_text(token) for token in tokens]
    for anchor_index in anchor_indices:
        start = max(0, anchor_index - window)
        end = min(len(normalised), anchor_index + len(normalised_anchor) + window)
        nearby = normalised[start:end]
        if all(token in nearby for token in normalised_tokens):
            return True, True
    return True, False


def verify_study(study: dict[str, object], timeout: int) -> dict[str, object]:
    pmid = str(study["pmid"])
    api_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&retmode=xml"
    )
    status, payload = fetch_bytes(api_url, timeout)
    source_pmid, abstract = abstract_from_pubmed_xml(payload) if status == 200 else ("", "")
    normalised = normalise_text(abstract)
    hr = str(study["reported_hr"])
    ci_lower = str(study["ci_lower"])
    ci_upper = str(study["ci_upper"])
    source_terms = [str(term) for term in study["source_terms"]]
    hazard_anchor_found, hazard_effect_tokens_near = tokens_near_anchor(
        abstract,
        "hazard ratio",
        [hr, ci_lower, ci_upper],
    )
    hr_anchor_found, hr_effect_tokens_near = tokens_near_anchor(
        abstract,
        "HR",
        [hr, ci_lower, ci_upper],
    )
    anchor_found = hazard_anchor_found or hr_anchor_found
    effect_tokens_near = hazard_effect_tokens_near or hr_effect_tokens_near
    confidence_interval_found, _ = tokens_near_anchor(
        abstract,
        "confidence interval",
        [ci_lower, ci_upper],
        window=320,
    )
    source_terms_found = all(normalise_text(term) in normalised for term in source_terms)
    verified = all(
        (
            status == 200,
            source_pmid == pmid,
            normalise_text(str(study["nct_id"])) in normalised,
            normalise_text(hr) in normalised,
            normalise_text(ci_lower) in normalised,
            normalise_text(ci_upper) in normalised,
            anchor_found,
            confidence_interval_found,
            source_terms_found,
            effect_tokens_near,
        )
    )
    return {
        "study_id": str(study["study_id"]),
        "pmid": pmid,
        "nct_id": str(study["nct_id"]),
        "design": str(study["design"]),
        "outcome_id": str(study["outcome_id"]),
        "evidence_scope": "pubmed_abstract_cross_design_reported_hr_tokens",
        "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
        "reported_hr": hr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "source_terms": source_terms,
        "pmid_found": source_pmid == pmid,
        "nct_id_found": normalise_text(str(study["nct_id"])) in normalised,
        "hr_token_found": normalise_text(hr) in normalised,
        "ci_lower_token_found": normalise_text(ci_lower) in normalised,
        "ci_upper_token_found": normalise_text(ci_upper) in normalised,
        "hazard_ratio_anchor_found": anchor_found,
        "confidence_interval_anchor_found": confidence_interval_found,
        "source_terms_found": source_terms_found,
        "tokens_near_hazard_ratio_anchor": effect_tokens_near,
        "verified": verified,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(manifest_path: Path, checked_at: str, timeout: int, pause_seconds: float) -> dict[str, object]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for study in manifest["studies"]:
        records.append(verify_study(study, timeout))
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    status = "verified" if records and all(record["verified"] for record in records) else "failed"
    return {
        "schema_version": "cross_design_hr_verification/v1",
        "benchmark_id": str(manifest["benchmark_id"]),
        "checked_at": checked_at,
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "status": status,
        "certification_effect": "none",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checked-at", default=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()

    report = build_report(args.manifest, args.checked_at, args.timeout, args.pause_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
