"""Verify reported hazard-ratio tokens against PubMed abstracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import time
import tomllib
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_MANIFEST = Path("validation/survival/sglt2_hf_reported_hrs.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/sglt2_hf_reported_hr_tokens.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-survival-hr-verifier/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()
    except URLError as exc:
        raise RuntimeError(f"could not fetch {url}: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def abstract_from_pubmed_xml(payload: bytes) -> tuple[str, str]:
    root = ET.fromstring(payload)
    pmid = root.findtext(".//MedlineCitation/PMID") or ""
    abstract = " ".join(
        element_text.strip()
        for element in root.findall(".//AbstractText")
        for element_text in ["".join(element.itertext())]
        if element_text.strip()
    )
    return pmid, abstract


def normalise_text(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    return re.sub(r"\s+", " ", text)


def tokens_near_anchor(text: str, anchor: str, tokens: list[str], *, window: int = 420) -> tuple[bool, bool]:
    normalised = normalise_text(text)
    anchor = normalise_text(anchor)
    anchor_index = normalised.find(anchor)
    if anchor_index < 0:
        return False, False
    start = max(0, anchor_index - window)
    end = min(len(normalised), anchor_index + len(anchor) + window)
    nearby = normalised[start:end]
    return True, all(normalise_text(token) in nearby for token in tokens)


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
    effect_tokens = [hr, ci_lower, ci_upper]
    anchor_found, effect_tokens_near = tokens_near_anchor(abstract, "hazard ratio", effect_tokens)
    _, source_terms_near = tokens_near_anchor(abstract, "hazard ratio", source_terms)
    hr_found = normalise_text(hr) in normalised
    ci_lower_found = normalise_text(ci_lower) in normalised
    ci_upper_found = normalise_text(ci_upper) in normalised
    verified = all(
        (
            status == 200,
            source_pmid == pmid,
            hr_found,
            ci_lower_found,
            ci_upper_found,
            anchor_found,
            effect_tokens_near,
            source_terms_near,
        )
    )
    return {
        "study_id": str(study["study_id"]),
        "pmid": pmid,
        "outcome_id": str(study["outcome_id"]),
        "evidence_scope": "pubmed_abstract_reported_hr_tokens",
        "abstract_sha256": sha256_bytes(abstract.encode("utf-8")),
        "reported_hr": hr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "source_terms": source_terms,
        "hr_token_found": hr_found,
        "ci_lower_token_found": ci_lower_found,
        "ci_upper_token_found": ci_upper_found,
        "hazard_ratio_anchor_found": anchor_found,
        "tokens_near_hazard_ratio_anchor": effect_tokens_near,
        "source_terms_near_hazard_ratio_anchor": source_terms_near,
        "verified": verified,
    }


def build_report(manifest_path: Path, checked_at: str, timeout: int, pause_seconds: float) -> dict[str, object]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for study in manifest["studies"]:
        if study.get("evidence_mode") != "reported_hr_pubmed_abstract":
            raise ValueError(f"{study['study_id']}: evidence_mode must be reported_hr_pubmed_abstract")
        records.append(verify_study(study, timeout))
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    status = "verified" if records and all(record["verified"] for record in records) else "failed"
    return {
        "schema_version": "survival_hr_verification/v1",
        "benchmark_id": str(manifest["benchmark_id"]),
        "checked_at": checked_at,
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "status": status,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
