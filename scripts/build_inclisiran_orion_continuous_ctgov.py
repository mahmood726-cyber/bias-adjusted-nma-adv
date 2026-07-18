"""Build a CT.gov-backed continuous pairwise reference fixture for ORION trials.

This script extracts reported least-squares mean differences for percent LDL-C
change from ORION-9/10/11 inclisiran trials. It writes pinned local artifacts
used by the R/metafor continuous-outcome reference candidate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias_nma_adv.pairwise import PairwiseMetaResult, fit_pairwise_meta  # noqa: E402


Z_975 = 1.959963984540054
SOURCE_POLICY = "clinicaltrials_gov + pubmed_abstract only"
BENCHMARK_ID = "inclisiran_orion_ldlc_pct_ctgov"
OUTCOME_ID = "ldlc_percent_change_day510"
OUTCOME_LABEL = "Percent Change in LDL-C From Baseline to Day 510"
EFFECT_SCALE = "mean_difference_percentage_points"


@dataclass(frozen=True)
class Selection:
    nct_id: str
    pmid: str
    study_label: str
    outcome_title_contains: str
    analysis_index: int
    treatment_group_id: str
    comparator_group_id: str
    estimand: str


SELECTIONS: tuple[Selection, ...] = (
    Selection(
        nct_id="NCT03397121",
        pmid="32197277",
        study_label="ORION-9",
        outcome_title_contains="Percent Change in LDL-C From Baseline To Day 510",
        analysis_index=0,
        treatment_group_id="OG000",
        comparator_group_id="OG001",
        estimand="final_value_day510",
    ),
    Selection(
        nct_id="NCT03399370",
        pmid="32187462",
        study_label="ORION-10",
        outcome_title_contains="Percentage Change in LDL-C From Baseline to Day 510",
        analysis_index=0,
        treatment_group_id="OG000",
        comparator_group_id="OG001",
        estimand="final_value_day510",
    ),
    Selection(
        nct_id="NCT03400800",
        pmid="32187462",
        study_label="ORION-11",
        outcome_title_contains="Percentage Change in LDL-C From Baseline to Day 510",
        analysis_index=0,
        treatment_group_id="OG000",
        comparator_group_id="OG001",
        estimand="final_value_day510",
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--checked-at",
        default=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for selection in SELECTIONS:
        row, source_record = _extract_selection(selection)
        rows.append(row)
        source_records.append(source_record)

    effects_path = root / "validation" / "continuous" / f"{BENCHMARK_ID}_effects.csv"
    source_check_path = root / "validation" / "source_checks" / f"{BENCHMARK_ID}_check.json"
    benchmark_path = root / "validation" / "continuous" / f"{BENCHMARK_ID}_benchmark.toml"

    effects_path.parent.mkdir(parents=True, exist_ok=True)
    source_check_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "study_id",
        "study_label",
        "trial",
        "nct_id",
        "pmid",
        "source_type",
        "source_url",
        "outcome_id",
        "outcome_label",
        "time_frame",
        "estimand",
        "comparison",
        "treatment",
        "comparator",
        "treatment_group_id",
        "comparator_group_id",
        "estimate",
        "se",
        "variance",
        "ci_low",
        "ci_high",
        "ci_level",
        "statistical_method",
        "param_type",
        "se_source",
    ]
    with effects_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    source_check = {
        "schema_version": "ctgov_continuous_source_check/v1",
        "benchmark_id": BENCHMARK_ID,
        "certification_effect": "none",
        "checked_at": args.checked_at,
        "source_policy": SOURCE_POLICY,
        "effect_scale": EFFECT_SCALE,
        "status": "verified",
        "verification_status": "verified",
        "n_records": len(source_records),
        "records": source_records,
    }
    source_check_path.write_text(
        json.dumps(source_check, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    benchmark_path.write_text(
        _benchmark_toml(
            checked_at=args.checked_at,
            effects_path=effects_path.relative_to(root).as_posix(),
            effects_sha256=_sha256(effects_path),
            source_check_path=source_check_path.relative_to(root).as_posix(),
            source_check_sha256=_sha256(source_check_path),
            rows=rows,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return 0


def _extract_selection(selection: Selection) -> tuple[dict[str, Any], dict[str, Any]]:
    study = _fetch_ctgov(selection.nct_id)
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{selection.nct_id}"
    title = str(study["protocolSection"]["identificationModule"]["briefTitle"])
    references = study.get("protocolSection", {}).get("referencesModule", {}).get("references", [])
    matching_references = [ref for ref in references if str(ref.get("pmid", "")) == selection.pmid]
    if not matching_references:
        raise RuntimeError(f"{selection.nct_id}: PMID {selection.pmid} not present in CT.gov references.")

    outcomes = study["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"]
    outcome = _single_outcome(outcomes, selection.outcome_title_contains, selection.nct_id)
    analyses = outcome.get("analyses", [])
    if selection.analysis_index >= len(analyses):
        raise RuntimeError(f"{selection.nct_id}: selected analysis index missing.")
    analysis = analyses[selection.analysis_index]
    group_ids = tuple(str(group_id) for group_id in analysis.get("groupIds", []))
    expected_group_ids = (selection.treatment_group_id, selection.comparator_group_id)
    if group_ids != expected_group_ids:
        raise RuntimeError(
            f"{selection.nct_id}: analysis group IDs {group_ids} do not match {expected_group_ids}."
        )
    if str(analysis.get("paramType", "")) != "Mean Difference (Final Values)":
        raise RuntimeError(f"{selection.nct_id}: selected analysis is not a reported mean difference.")
    ci_level = float(analysis["ciPctValue"])
    if ci_level != 95.0:
        raise RuntimeError(f"{selection.nct_id}: expected a 95 percent confidence interval.")

    estimate = float(analysis["paramValue"])
    ci_low = float(analysis["ciLowerLimit"])
    ci_high = float(analysis["ciUpperLimit"])
    se = (ci_high - ci_low) / (2.0 * Z_975)
    if se <= 0.0:
        raise RuntimeError(f"{selection.nct_id}: derived SE must be positive.")

    groups = {str(group["id"]): group for group in outcome.get("groups", [])}
    treatment = str(groups[selection.treatment_group_id]["title"])
    comparator = str(groups[selection.comparator_group_id]["title"])
    if "inclisiran" not in treatment.lower() or "placebo" not in comparator.lower():
        raise RuntimeError(f"{selection.nct_id}: selected groups are not inclisiran vs placebo.")

    study_id = f"{selection.nct_id}_{selection.study_label.lower().replace('-', '_')}"
    row = {
        "study_id": study_id,
        "study_label": selection.study_label,
        "trial": title,
        "nct_id": selection.nct_id,
        "pmid": selection.pmid,
        "source_type": "clinicaltrials_gov",
        "source_url": f"https://clinicaltrials.gov/study/{selection.nct_id}",
        "outcome_id": OUTCOME_ID,
        "outcome_label": OUTCOME_LABEL,
        "time_frame": str(outcome["timeFrame"]),
        "estimand": selection.estimand,
        "comparison": "inclisiran_vs_placebo",
        "treatment": treatment,
        "comparator": comparator,
        "treatment_group_id": selection.treatment_group_id,
        "comparator_group_id": selection.comparator_group_id,
        "estimate": f"{estimate:.12g}",
        "se": f"{se:.12g}",
        "variance": f"{se * se:.12g}",
        "ci_low": f"{ci_low:.12g}",
        "ci_high": f"{ci_high:.12g}",
        "ci_level": "95",
        "statistical_method": str(analysis.get("statisticalMethod", "")),
        "param_type": str(analysis.get("paramType", "")),
        "se_source": "derived_from_reported_95_ci_using_normal_quantile",
    }
    source_record = {
        "study_id": study_id,
        "nct_id": selection.nct_id,
        "pmid": selection.pmid,
        "api_url": api_url,
        "manifest_url": f"https://clinicaltrials.gov/study/{selection.nct_id}",
        "source_type": "clinicaltrials_gov",
        "evidence_scope": "clinicaltrials_gov_continuous_treatment_difference",
        "result_reference_found": True,
        "reference_citation": str(matching_references[0].get("citation", "")),
        "outcome_title": str(outcome["title"]),
        "time_frame": str(outcome["timeFrame"]),
        "analysis_index": selection.analysis_index,
        "analysis_group_ids": list(group_ids),
        "analysis_group_description": str(analysis.get("groupDescription", "")),
        "analysis_param_type": str(analysis.get("paramType", "")),
        "analysis_param_value": str(analysis.get("paramValue", "")),
        "analysis_ci_level": str(analysis.get("ciPctValue", "")),
        "analysis_ci_lower": str(analysis.get("ciLowerLimit", "")),
        "analysis_ci_upper": str(analysis.get("ciUpperLimit", "")),
        "selected_treatment": treatment,
        "selected_comparator": comparator,
    }
    return row, source_record


def _fetch_ctgov(nct_id: str) -> dict[str, Any]:
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    with urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"{nct_id}: CT.gov API returned HTTP {response.status}.")
        return json.loads(response.read().decode("utf-8"))


def _single_outcome(outcomes: list[dict[str, Any]], title_contains: str, nct_id: str) -> dict[str, Any]:
    matches = [
        outcome
        for outcome in outcomes
        if title_contains.lower() in str(outcome.get("title", "")).lower()
        and str(outcome.get("paramType", "")) == "LEAST_SQUARES_MEAN"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{nct_id}: expected one LS-mean outcome containing {title_contains!r}, found {len(matches)}."
        )
    return matches[0]


def _benchmark_toml(
    *,
    checked_at: str,
    effects_path: str,
    effects_sha256: str,
    source_check_path: str,
    source_check_sha256: str,
    rows: list[dict[str, Any]],
) -> str:
    fixed_effect = _fit_pairwise_summary(rows, method="FE")
    random_effect = _fit_pairwise_summary(rows, method="REML")
    lines = [
        'schema_version = "continuous_pairwise_benchmark/v1"',
        f'benchmark_id = "{BENCHMARK_ID}"',
        'status = "local_pass"',
        'certification_effect = "none"',
        f'checked_at = "{checked_at}"',
        f'source_policy = "{SOURCE_POLICY}"',
        'evidence_mode = "ctgov_reported_adjusted_treatment_difference"',
        f'effect_scale = "{EFFECT_SCALE}"',
        f'effects_csv = "{effects_path}"',
        f'effects_csv_sha256 = "{effects_sha256}"',
        f'source_verification_report = "{source_check_path}"',
        f'source_verification_report_sha256 = "{source_check_sha256}"',
        f"n_studies = {len(rows)}",
        'limitations = ["CT.gov adjusted least-squares mean differences are used as reported", "different ORION trial populations are pooled only for software validation", "not broad continuous-outcome parity", "not clinical guidance", "does not certify model performance"]',
        "",
        "[source_bundle]",
        f'benchmark_id = "{BENCHMARK_ID}"',
        f'manifest_sha256 = "{effects_sha256}"',
        'verification_status = "verified"',
        f"n_records = {len(rows)}",
        f"source_counts = {{clinicaltrials_gov = {len(rows)}}}",
        f"linked_result_pmids = {len({row['pmid'] for row in rows})}",
        "",
        "[candidate]",
        'engine = "bias_nma_adv.pairwise.fit_pairwise_meta"',
        'purpose = "local continuous pairwise software-validation summary only"',
        "",
        "[candidate.fixed_effect]",
        *_pairwise_result_lines(fixed_effect),
        "",
        "[candidate.random_effect]",
        *_pairwise_result_lines(random_effect),
        "",
    ]
    for row in rows:
        lines.extend(
            [
                "[[study_effects]]",
                f'study_id = "{row["study_id"]}"',
                f'study_label = "{row["study_label"]}"',
                f'trial = "{_toml_string(row["trial"])}"',
                f'nct_id = "{row["nct_id"]}"',
                f'pmid = "{row["pmid"]}"',
                f'source_type = "{row["source_type"]}"',
                f'source_url = "{row["source_url"]}"',
                f'outcome_id = "{row["outcome_id"]}"',
                f'outcome_label = "{row["outcome_label"]}"',
                f'time_frame = "{_toml_string(row["time_frame"])}"',
                f'estimand = "{row["estimand"]}"',
                f'comparison = "{row["comparison"]}"',
                f'treatment = "{_toml_string(row["treatment"])}"',
                f'comparator = "{_toml_string(row["comparator"])}"',
                f'treatment_group_id = "{row["treatment_group_id"]}"',
                f'comparator_group_id = "{row["comparator_group_id"]}"',
                f'estimate = {row["estimate"]}',
                f'se = {row["se"]}',
                f'variance = {row["variance"]}',
                f'ci_low = {row["ci_low"]}',
                f'ci_high = {row["ci_high"]}',
                f'ci_level = {row["ci_level"]}',
                f'statistical_method = "{_toml_string(row["statistical_method"])}"',
                f'param_type = "{_toml_string(row["param_type"])}"',
                f'se_source = "{row["se_source"]}"',
                "",
            ]
        )
    return "\n".join(lines)


def _fit_pairwise_summary(rows: list[dict[str, Any]], *, method: str) -> PairwiseMetaResult:
    effects = [float(row["estimate"]) for row in rows]
    variances = [float(row["variance"]) for row in rows]
    return fit_pairwise_meta(effects, variances, method=method, hksj=False)


def _pairwise_result_lines(result: PairwiseMetaResult) -> list[str]:
    return [
        f'method = "{result.method}"',
        f"estimate = {result.estimate:.17g}",
        f"se = {result.se:.17g}",
        f"ci_low = {result.ci_low:.17g}",
        f"ci_high = {result.ci_high:.17g}",
        f"tau2 = {result.tau2:.17g}",
        f"q = {result.q:.17g}",
        f"df = {result.df}",
        f"hksj = {str(result.hksj).lower()}",
        f"hksj_q_factor = {result.hksj_q_factor:.17g}",
        f"warnings = [{', '.join(_quoted_toml_string(item) for item in result.warnings)}]",
    ]


def _toml_string(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _quoted_toml_string(value: Any) -> str:
    return f'"{_toml_string(value)}"'


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
