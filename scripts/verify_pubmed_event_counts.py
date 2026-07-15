"""Verify real-meta arm counts against PubMed abstract event-count tokens."""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_DATASET = Path("validation/real_meta/sglt2_hf_primary_events.csv")
DEFAULT_MANIFEST = Path("validation/real_meta/sglt2_hf_primary_sources.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/sglt2_hf_primary_event_counts.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-event-count-verifier/0.1"})
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


def load_rows(dataset: Path) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            study = out.setdefault(str(row["study_id"]), {})
            study[str(row["arm_role"])] = {
                "study_id": row["study_id"],
                "trial": row["trial"],
                "nct_id": row["nct_id"],
                "pmid": row["pmid"],
                "outcome_id": row["outcome_id"],
                "outcome_label": row["outcome_label"],
                "arm_role": row["arm_role"],
                "treatment": row["treatment"],
                "events": int(row["events"]),
                "n": int(row["n"]),
            }
    return out


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


def token_near_any_term(text: str, token: str, terms: list[str], *, window: int = 240) -> tuple[bool, bool]:
    normalised = normalise_text(text)
    token = normalise_text(token)
    token_index = normalised.find(token)
    if token_index < 0:
        return False, False
    start = max(0, token_index - window)
    end = min(len(normalised), token_index + len(token) + window)
    nearby = normalised[start:end]
    term_found = any(normalise_text(term) in nearby for term in terms)
    return True, term_found


def verify_study(
    study: dict[str, object],
    arms: dict[str, dict[str, object]],
    timeout: int,
) -> dict[str, object]:
    active = arms["active"]
    control = arms["control"]
    pmid = str(study["pmid"])
    api_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&retmode=xml"
    )
    status, payload = fetch_bytes(api_url, timeout)
    source_pmid, abstract = abstract_from_pubmed_xml(payload) if status == 200 else ("", "")
    active_token = f"{active['events']} of {active['n']}"
    control_token = f"{control['events']} of {control['n']}"
    active_terms = [str(term) for term in study["active_source_terms"]]
    control_terms = [str(term) for term in study["control_source_terms"]]
    active_token_found, active_term_near = token_near_any_term(abstract, active_token, active_terms)
    control_token_found, control_term_near = token_near_any_term(abstract, control_token, control_terms)
    verified = all(
        (
            status == 200,
            source_pmid == pmid,
            active_token_found,
            control_token_found,
            active_term_near,
            control_term_near,
        )
    )
    return {
        "study_id": str(study["study_id"]),
        "pmid": pmid,
        "outcome_id": str(study["outcome_id"]),
        "outcome_label": str(study["outcome_label"]),
        "evidence_scope": "pubmed_abstract_event_count_tokens",
        "abstract_sha256": sha256_bytes(abstract.encode("utf-8")),
        "active_events": active["events"],
        "active_n": active["n"],
        "control_events": control["events"],
        "control_n": control["n"],
        "active_count_token": active_token,
        "control_count_token": control_token,
        "active_source_terms": active_terms,
        "control_source_terms": control_terms,
        "active_count_token_found": active_token_found,
        "control_count_token_found": control_token_found,
        "active_term_near_count": active_term_near,
        "control_term_near_count": control_term_near,
        "verified": verified,
    }


def build_report(
    dataset: Path,
    manifest_path: Path,
    checked_at: str,
    timeout: int,
    pause_seconds: float,
) -> dict[str, object]:
    rows_by_study = load_rows(dataset)
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for study in manifest["studies"]:
        study_id = str(study["study_id"])
        if study.get("event_count_source_type") != "pubmed_abstract":
            raise ValueError(f"{study_id}: event_count_source_type must be pubmed_abstract")
        records.append(verify_study(study, rows_by_study[study_id], timeout))
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    status = "verified" if records and all(record["verified"] for record in records) else "failed"
    return {
        "schema_version": "event_count_verification/v1",
        "benchmark_id": str(manifest["benchmark_id"]),
        "checked_at": checked_at,
        "dataset": dataset.as_posix(),
        "dataset_sha256": sha256_file(dataset),
        "source_manifest": manifest_path.as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "status": status,
        "certification_effect": "none",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checked-at", default=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()

    report = build_report(args.dataset, args.manifest, args.checked_at, args.timeout, args.pause_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
