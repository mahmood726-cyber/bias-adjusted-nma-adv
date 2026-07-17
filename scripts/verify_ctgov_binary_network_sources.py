"""Verify a CT.gov arm-count binary network manifest against public sources."""

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


DEFAULT_MANIFEST = Path("validation/networks/psoriasis_pasi90_ctgov_binary_network.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/psoriasis_pasi90_ctgov_binary_network_check.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-binary-network-verifier/0.1"})
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
            digest.update(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return digest.hexdigest()


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[-\u2010-\u2015]", " ", text)
    return re.sub(r"\s+", " ", text)


def contains_all(text: str, terms: list[str]) -> bool:
    clean = normalise(text)
    return all(normalise(term) in clean for term in terms)


def number_equal(observed: object, expected: object) -> bool:
    try:
        return abs(float(str(observed)) - float(str(expected))) <= 1e-12
    except (TypeError, ValueError):
        return False


def verify_ctgov(study: dict[str, object], timeout: int) -> dict[str, object]:
    nct_id = str(study["nct_id"])
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    status, payload = fetch_bytes(api_url, timeout)
    details: dict[str, object] = {
        "nct_id_found": False,
        "status_completed": False,
        "outcome_found": False,
        "arm_counts_found": False,
        "source_terms_found": False,
    }
    if status == 200:
        parsed = json.loads(payload.decode("utf-8"))
        protocol = parsed.get("protocolSection", {})
        details["nct_id_found"] = (
            str(protocol.get("identificationModule", {}).get("nctId", "")) == nct_id
        )
        details["status_completed"] = (
            str(protocol.get("statusModule", {}).get("overallStatus", "")).upper() == "COMPLETED"
        )
        outcome = find_matching_outcome(parsed, study)
        if outcome is not None:
            details["outcome_found"] = True
            details["matched_outcome_title"] = outcome.get("title", "")
            details["matched_param_type"] = outcome.get("paramType", "")
            groups = {str(group["id"]): group for group in outcome.get("groups", [])}
            denoms = {
                str(count["groupId"]): count.get("value")
                for denom in outcome.get("denoms", [])
                for count in denom.get("counts", [])
            }
            measurements = {
                str(measurement["groupId"]): measurement.get("value")
                for klass in outcome.get("classes", [])
                for category in klass.get("categories", [])
                for measurement in category.get("measurements", [])
            }
            arm_checks = []
            for arm in study["arms"]:
                group_id = str(arm["group_id"])
                group_text = json.dumps(groups.get(group_id, {}), ensure_ascii=False)
                arm_checks.append(
                    {
                        "arm_id": arm["arm_id"],
                        "group_id": group_id,
                        "events_found": number_equal(measurements.get(group_id), arm["events"]),
                        "n_found": number_equal(denoms.get(group_id), arm["n"]),
                        "group_title_found": contains_all(group_text, [str(arm["group_title"])]),
                    }
                )
            details["arm_checks"] = arm_checks
            details["arm_counts_found"] = all(
                item["events_found"] and item["n_found"] for item in arm_checks
            )
            details["source_terms_found"] = contains_all(
                json.dumps(outcome, ensure_ascii=False),
                [str(term) for term in study["source_terms"]],
            )
    return {
        "study_id": study["study_id"],
        "source_type": "clinicaltrials_gov",
        "identifier": nct_id,
        "evidence_scope": "clinicaltrials_gov_arm_level_binary_counts",
        "response_sha256": sha256_bytes(payload),
        "verified": all(
            bool(details[key])
            for key in (
                "nct_id_found",
                "status_completed",
                "outcome_found",
                "arm_counts_found",
                "source_terms_found",
            )
        ),
        "details": details,
    }


def find_matching_outcome(
    parsed: dict[str, object],
    study: dict[str, object],
) -> dict[str, object] | None:
    outcomes = (
        parsed.get("resultsSection", {})
        .get("outcomeMeasuresModule", {})
        .get("outcomeMeasures", [])
    )
    terms = [str(term) for term in study["outcome_search_terms"]]
    expected_group_ids = {str(arm["group_id"]) for arm in study["arms"]}
    for outcome in outcomes:
        text = json.dumps(
            {
                "title": outcome.get("title", ""),
                "description": outcome.get("description", ""),
                "timeFrame": outcome.get("timeFrame", ""),
                "unitOfMeasure": outcome.get("unitOfMeasure", ""),
            },
            ensure_ascii=False,
        )
        if not contains_all(text, terms):
            continue
        group_ids = {str(group.get("id", "")) for group in outcome.get("groups", [])}
        if expected_group_ids <= group_ids:
            return outcome
    return None


def verify_pubmed(study: dict[str, object], timeout: int) -> dict[str, object]:
    pmid = str(study["pmid"])
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={pmid}&retmode=json"
    )
    status, payload = fetch_bytes(url, timeout)
    details: dict[str, object] = {
        "pmid_found": False,
        "title_terms_found": False,
    }
    if status == 200:
        parsed = json.loads(payload.decode("utf-8"))
        record = parsed.get("result", {}).get(pmid, {})
        title = str(record.get("title", ""))
        details["title"] = title
        details["pmid_found"] = str(record.get("uid", "")) == pmid
        details["title_terms_found"] = contains_all(
            title,
            [str(term) for term in study["pubmed_title_terms"]],
        )
    return {
        "study_id": study["study_id"],
        "source_type": "pubmed_abstract",
        "identifier": pmid,
        "evidence_scope": "pubmed_abstract_binary_network_identity",
        "response_sha256": sha256_bytes(payload),
        "verified": bool(details["pmid_found"] and details["title_terms_found"]),
        "details": details,
    }


def build_report(manifest_path: Path, checked_at: str, timeout: int, pause_seconds: float) -> dict[str, object]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for study in manifest["studies"]:
        records.append(verify_ctgov(study, timeout))
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        records.append(verify_pubmed(study, timeout))
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return {
        "schema_version": "ctgov_binary_network_verification/v1",
        "benchmark_id": str(manifest["benchmark_id"]),
        "checked_at": checked_at,
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "status": "verified" if all(record["verified"] for record in records) else "failed",
        "certification_effect": "none",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checked-at", default=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pause-seconds", type=float, default=0.4)
    args = parser.parse_args()

    report = build_report(args.manifest, args.checked_at, args.timeout, args.pause_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
