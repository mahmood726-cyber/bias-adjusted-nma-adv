"""Verify a source-backed component-NMA manifest against CT.gov and PubMed."""

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


DEFAULT_MANIFEST = Path("validation/component/sitagliptin_pioglitazone_component.toml")
DEFAULT_OUTPUT = Path("validation/source_checks/sitagliptin_pioglitazone_component_check.json")


def fetch_bytes(url: str, timeout: int) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-component-verifier/0.1"})
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


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[-\u2010-\u2015]", " ", text)
    return re.sub(r"\s+", " ", text)


def contains_all(text: str, terms: list[str]) -> bool:
    clean = normalise(text)
    return all(normalise(term) in clean for term in terms)


def number_equal(observed: object, expected: object, *, atol: float = 1e-12) -> bool:
    try:
        return abs(float(str(observed)) - float(str(expected))) <= atol
    except (TypeError, ValueError):
        return False


def find_outcome(parsed: dict[str, object], manifest: dict[str, object]) -> tuple[dict[str, object] | None, str]:
    outcomes = (
        parsed.get("resultsSection", {})
        .get("outcomeMeasuresModule", {})
        .get("outcomeMeasures", [])
    )
    for outcome in outcomes:
        title = str(outcome.get("title", ""))
        if title == manifest["outcome_title"]:
            return outcome, title
    return None, ""


def verify_ctgov(manifest: dict[str, object], timeout: int) -> dict[str, object]:
    nct_id = str(manifest["nct_id"])
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    status, payload = fetch_bytes(api_url, timeout)
    details: dict[str, object] = {
        "nct_id_found": False,
        "status_completed": False,
        "outcome_found": False,
        "outcome_param_type_found": False,
        "outcome_dispersion_type_found": False,
        "arm_values_found": False,
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
        outcome, title = find_outcome(parsed, manifest)
        details["matched_outcome_title"] = title
        if outcome is not None:
            details["outcome_found"] = True
            details["outcome_param_type_found"] = str(outcome.get("paramType", "")) == str(
                manifest["outcome_param_type"]
            )
            details["outcome_dispersion_type_found"] = str(outcome.get("dispersionType", "")) == str(
                manifest["outcome_dispersion_type"]
            )
            groups = {group["id"]: group for group in outcome.get("groups", [])}
            denoms = {}
            for denom in outcome.get("denoms", []):
                for count in denom.get("counts", []):
                    denoms[str(count["groupId"])] = count["value"]
            measurements = {}
            for klass in outcome.get("classes", []):
                for category in klass.get("categories", []):
                    for measurement in category.get("measurements", []):
                        measurements[str(measurement["groupId"])] = measurement
            arm_checks = []
            for arm in manifest["arms"]:
                group_id = str(arm["group_id"])
                measurement = measurements.get(group_id, {})
                group_text = json.dumps(groups.get(group_id, {}), ensure_ascii=False)
                arm_checks.append(
                    {
                        "arm_id": arm["arm_id"],
                        "group_id": group_id,
                        "n_found": number_equal(denoms.get(group_id, ""), arm["n"]),
                        "value_found": number_equal(measurement.get("value", ""), arm["lsmean"]),
                        "lower_found": number_equal(measurement.get("lowerLimit", ""), arm["lower"]),
                        "upper_found": number_equal(measurement.get("upperLimit", ""), arm["upper"]),
                        "source_terms_found": contains_all(group_text, list(arm["source_terms"])),
                    }
                )
            details["arm_checks"] = arm_checks
            details["arm_values_found"] = all(
                item["n_found"]
                and item["value_found"]
                and item["lower_found"]
                and item["upper_found"]
                for item in arm_checks
            )
            details["source_terms_found"] = all(item["source_terms_found"] for item in arm_checks)
    verified = all(
        bool(details[key])
        for key in (
            "nct_id_found",
            "status_completed",
            "outcome_found",
            "outcome_param_type_found",
            "outcome_dispersion_type_found",
            "arm_values_found",
            "source_terms_found",
        )
    )
    return {
        "source_type": "clinicaltrials_gov",
        "identifier": nct_id,
        "evidence_scope": "clinicaltrials_gov_component_lsmean",
        "response_sha256": sha256_bytes(payload),
        "verified": verified,
        "details": details,
    }


def verify_pubmed(manifest: dict[str, object], timeout: int) -> dict[str, object]:
    pmid = str(manifest["pmid"])
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={pmid}&retmode=json"
    )
    status, payload = fetch_bytes(url, timeout)
    details: dict[str, object] = {
        "pmid_found": False,
        "title_terms_found": False,
        "factorial_terms_found": False,
    }
    if status == 200:
        parsed = json.loads(payload.decode("utf-8"))
        record = parsed.get("result", {}).get(pmid, {})
        title = str(record.get("title", ""))
        details["title"] = title
        details["pmid_found"] = str(record.get("uid", "")) == pmid
        details["title_terms_found"] = contains_all(title, ["sitagliptin", "pioglitazone"])
        details["factorial_terms_found"] = contains_all(title, ["factorial"])
    verified = all(bool(details[key]) for key in ("pmid_found", "title_terms_found", "factorial_terms_found"))
    return {
        "source_type": "pubmed_abstract",
        "identifier": pmid,
        "evidence_scope": "pubmed_abstract_component_identity",
        "response_sha256": sha256_bytes(payload),
        "verified": verified,
        "details": details,
    }


def build_report(manifest_path: Path, checked_at: str, timeout: int, pause_seconds: float) -> dict[str, object]:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    records = [verify_ctgov(manifest, timeout)]
    if pause_seconds > 0:
        time.sleep(pause_seconds)
    records.append(verify_pubmed(manifest, timeout))
    return {
        "schema_version": "component_nma_source_verification/v1",
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
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()

    report = build_report(args.manifest, args.checked_at, args.timeout, args.pause_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
