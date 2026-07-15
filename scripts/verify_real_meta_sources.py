"""Verify real-meta source manifest identities against public CT.gov/PubMed APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import tomllib
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_MANIFEST = Path("validation/real_meta/sglt2_hf_primary_sources.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/sglt2_hf_primary_source_check.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-source-verifier/0.1"})
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


def verify_ctgov(study_id: str, source: dict[str, object], timeout: int) -> dict[str, object]:
    identifier = str(source["identifier"])
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{identifier}"
    status, payload = fetch_bytes(api_url, timeout)
    title = ""
    identity_verified = False
    details: dict[str, object] = {}
    if status == 200:
        parsed = json.loads(payload.decode("utf-8"))
        identification = parsed.get("protocolSection", {}).get("identificationModule", {})
        status_module = parsed.get("protocolSection", {}).get("statusModule", {})
        nct_id = str(identification.get("nctId", ""))
        title = str(identification.get("briefTitle", ""))
        identity_verified = nct_id == identifier
        details = {
            "nct_id": nct_id,
            "brief_title": title,
            "overall_status": str(status_module.get("overallStatus", "")),
        }
    return {
        "study_id": study_id,
        "source_type": "clinicaltrials_gov",
        "identifier": identifier,
        "manifest_url": str(source["url"]),
        "api_url": api_url,
        "http_status": status,
        "identity_verified": identity_verified,
        "response_sha256": sha256_bytes(payload),
        "title": title,
        "evidence_scope": "identity_and_reachability",
        "details": details,
    }


def verify_pubmed(study_id: str, source: dict[str, object], timeout: int) -> dict[str, object]:
    identifier = str(source["identifier"])
    api_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={identifier}&retmode=xml"
    )
    status, payload = fetch_bytes(api_url, timeout)
    title = ""
    identity_verified = False
    details: dict[str, object] = {}
    if status == 200:
        root = ET.fromstring(payload)
        pmid = root.findtext(".//MedlineCitation/PMID") or ""
        title = "".join(root.findtext(".//ArticleTitle") or "").strip()
        journal = root.findtext(".//Journal/ISOAbbreviation") or ""
        abstract_text = " ".join(
            element_text.strip()
            for element in root.findall(".//AbstractText")
            for element_text in ["".join(element.itertext())]
            if element_text.strip()
        )
        identity_verified = pmid == identifier
        details = {
            "pmid": pmid,
            "article_title": title,
            "journal": journal,
            "abstract_present": bool(abstract_text),
            "abstract_sha256": sha256_bytes(abstract_text.encode("utf-8")) if abstract_text else "",
        }
    return {
        "study_id": study_id,
        "source_type": "pubmed_abstract",
        "identifier": identifier,
        "manifest_url": str(source["url"]),
        "api_url": api_url,
        "http_status": status,
        "identity_verified": identity_verified,
        "response_sha256": sha256_bytes(payload),
        "title": title,
        "evidence_scope": "identity_and_reachability",
        "details": details,
    }


def build_report(manifest_path: Path, checked_at: str, timeout: int, pause_seconds: float) -> dict[str, object]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for study in manifest["studies"]:
        study_id = str(study["study_id"])
        for source in study["sources"]:
            source_type = str(source["source_type"])
            if source_type == "clinicaltrials_gov":
                records.append(verify_ctgov(study_id, source, timeout))
            elif source_type == "pubmed_abstract":
                records.append(verify_pubmed(study_id, source, timeout))
            else:
                raise ValueError(f"unsupported source_type for live verification: {source_type}")
            if pause_seconds > 0:
                time.sleep(pause_seconds)

    status = "verified" if records and all(record["identity_verified"] for record in records) else "failed"
    return {
        "schema_version": "source_verification/v1",
        "benchmark_id": str(manifest["benchmark_id"]),
        "checked_at": checked_at,
        "source_manifest": manifest_path.as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
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
