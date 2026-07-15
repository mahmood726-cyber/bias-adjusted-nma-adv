"""Verify reported HR network contrasts against ClinicalTrials.gov results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import tomllib
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MANIFEST = Path("validation/networks/t2d_mace_ctgov_hrs.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/t2d_mace_ctgov_hr_network_check.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-ctgov-hr-network-verifier/0.1"})
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


def normalise_text(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"[-\u2010-\u2015]", " ", text)
    return re.sub(r"\s+", " ", text)


def contains_all(text: str, terms: list[str]) -> bool:
    normalised = normalise_text(text)
    return all(normalise_text(term) in normalised for term in terms)


def analysis_matches(analysis: dict[str, object], study: dict[str, object]) -> bool:
    return (
        str(analysis.get("paramType", "")) == "Hazard Ratio (HR)"
        and str(analysis.get("paramValue", "")) == str(study["reported_hr"])
        and str(analysis.get("ciLowerLimit", "")) == str(study["ci_lower"])
        and str(analysis.get("ciUpperLimit", "")) == str(study["ci_upper"])
    )


def find_matching_analysis(parsed: dict[str, object], study: dict[str, object]) -> tuple[bool, bool, str, str]:
    outcome_terms = [str(term) for term in study["outcome_search_terms"]]
    outcome_measures = (
        parsed.get("resultsSection", {})
        .get("outcomeMeasuresModule", {})
        .get("outcomeMeasures", [])
    )
    for outcome in outcome_measures:
        title = str(outcome.get("title", ""))
        outcome_terms_found = contains_all(title, outcome_terms)
        for analysis in outcome.get("analyses", []) or []:
            if analysis_matches(analysis, study):
                return (
                    True,
                    outcome_terms_found,
                    title,
                    str(analysis.get("paramType", "")),
                )
    return False, False, "", ""


def verify_study(study: dict[str, object], timeout: int) -> dict[str, object]:
    nct_id = str(study["nct_id"])
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    status, payload = fetch_bytes(api_url, timeout)
    nct_id_found = False
    status_completed = False
    hazard_ratio_analysis_found = False
    outcome_terms_found = False
    matched_outcome_title = ""
    matched_param_type = ""
    source_terms = [str(term) for term in study["source_terms"]]
    outcome_search_terms = [str(term) for term in study["outcome_search_terms"]]
    source_terms_found = False

    if status == 200:
        parsed = json.loads(payload.decode("utf-8"))
        protocol = parsed.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        nct_id_found = str(identification.get("nctId", "")) == nct_id
        status_completed = str(status_module.get("overallStatus", "")).upper() == "COMPLETED"
        hazard_ratio_analysis_found, outcome_terms_found, matched_outcome_title, matched_param_type = find_matching_analysis(
            parsed, study
        )
        source_terms_found = contains_all(json.dumps(parsed, ensure_ascii=False), source_terms)

    ci_tokens_found = hazard_ratio_analysis_found
    verified = all(
        (
            status == 200,
            nct_id_found,
            status_completed,
            hazard_ratio_analysis_found,
            ci_tokens_found,
            outcome_terms_found,
            source_terms_found,
        )
    )
    return {
        "study_id": str(study["study_id"]),
        "nct_id": nct_id,
        "outcome_id": str(study["outcome_id"]),
        "evidence_scope": "clinicaltrials_gov_reported_hr_analysis",
        "response_sha256": sha256_bytes(payload),
        "reported_hr": str(study["reported_hr"]),
        "ci_lower": str(study["ci_lower"]),
        "ci_upper": str(study["ci_upper"]),
        "source_terms": source_terms,
        "outcome_search_terms": outcome_search_terms,
        "nct_id_found": nct_id_found,
        "status_completed": status_completed,
        "hazard_ratio_analysis_found": hazard_ratio_analysis_found,
        "ci_tokens_found": ci_tokens_found,
        "outcome_terms_found": outcome_terms_found,
        "source_terms_found": source_terms_found,
        "matched_outcome_title": matched_outcome_title,
        "matched_param_type": matched_param_type,
        "verified": verified,
    }


def build_report(manifest_path: Path, checked_at: str, timeout: int, pause_seconds: float) -> dict[str, object]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for study in manifest["studies"]:
        if study.get("evidence_mode") != "reported_hr_clinicaltrials_gov_results":
            raise ValueError(f"{study['study_id']}: evidence_mode must be reported_hr_clinicaltrials_gov_results")
        records.append(verify_study(study, timeout))
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    status = "verified" if records and all(record["verified"] for record in records) else "failed"
    return {
        "schema_version": "ctgov_hr_network_verification/v1",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
