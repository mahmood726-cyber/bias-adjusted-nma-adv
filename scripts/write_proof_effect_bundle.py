"""Write proof-carrying extracted-effect bundles from source-verified manifests."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.proof_effect_bundle import (  # noqa: E402
    PROOF_EFFECT_BUNDLE_SCHEMA_VERSION,
    ProofEffectBundleError,
    build_reported_hr_proof_effect_bundle,
)
from bias_nma_adv.survival_benchmark import load_survival_hr_manifest  # noqa: E402


DEFAULT_MANIFEST = Path("validation/survival/sglt2_hf_reported_hrs.toml")
DEFAULT_TOKEN_REPORT = Path("validation/source_checks/sglt2_hf_reported_hr_tokens.json")
DEFAULT_IDENTITY_REPORT = Path("validation/source_checks/sglt2_hf_reported_hr_source_check.json")
DEFAULT_OUTPUT = Path("validation/ingestion/sglt2_hf_reported_hr_proof_effects.json")


def fetch_pubmed_abstract(pmid: str, *, timeout: int) -> str:
    """Fetch and concatenate PubMed abstract text for one PMID."""

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={pmid}&retmode=xml"
    )
    request = Request(url, headers={"User-Agent": "bias-adjusted-nma-adv-proof-effect-writer/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"PubMed returned HTTP {exc.code} for PMID {pmid}.") from exc
    except URLError as exc:
        raise RuntimeError(f"could not fetch PubMed PMID {pmid}: {exc}") from exc

    root = ET.fromstring(payload)
    source_pmid = root.findtext(".//MedlineCitation/PMID") or ""
    if source_pmid != pmid:
        raise RuntimeError(f"PubMed identity mismatch: requested {pmid}, received {source_pmid}.")
    abstract = " ".join(
        element_text.strip()
        for element in root.findall(".//AbstractText")
        for element_text in ["".join(element.itertext())]
        if element_text.strip()
    )
    if not abstract:
        raise RuntimeError(f"PubMed PMID {pmid} has no abstract text.")
    return abstract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--token-report", type=Path, default=DEFAULT_TOKEN_REPORT)
    parser.add_argument("--identity-report", type=Path, default=DEFAULT_IDENTITY_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument(
        "--checked-at",
        default=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_survival_hr_manifest(args.manifest)
        abstracts = {}
        for study in manifest.studies:
            abstracts[study.pmid] = fetch_pubmed_abstract(study.pmid, timeout=args.timeout)
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)
        bundle = build_reported_hr_proof_effect_bundle(
            manifest_path=args.manifest,
            token_report_path=args.token_report,
            identity_report_path=args.identity_report,
            abstracts_by_pmid=abstracts,
            checked_at=args.checked_at,
        )
    except (OSError, RuntimeError, ProofEffectBundleError, ET.ParseError) as exc:
        failure = {
            "schema_version": PROOF_EFFECT_BUNDLE_SCHEMA_VERSION,
            "checked_at": args.checked_at,
            "status": "failed",
            "certification_effect": "none",
            "error": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"proof-carrying effect bundle written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
