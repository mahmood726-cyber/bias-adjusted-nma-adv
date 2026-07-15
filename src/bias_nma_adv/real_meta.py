"""Real meta-analysis benchmark helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

from bias_nma_adv.bayesian import BayesianNMAMCMCSampler
from bias_nma_adv.data import EvidenceDataset, ValidationError
from bias_nma_adv.evidence_sources import ALLOWED_SOURCE_TYPES
from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources
from bias_nma_adv.ingestion import EvidenceIngestionRecord, validate_ingestion_records
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler
from bias_nma_adv.pairwise import PairwiseMetaResult, fit_pairwise_meta


TEXT_HASH_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".r",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class ArmEventRow:
    study_id: str
    trial: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    arm_role: str
    treatment: str
    events: int
    n: int


@dataclass(frozen=True)
class EffectEstimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class StudyLogOREffect:
    study_id: str
    trial: str
    nct_id: str
    pmid: str
    outcome_id: str
    outcome_label: str
    active_treatment: str
    control_treatment: str
    active_events: int
    active_n: int
    control_events: int
    control_n: int
    effect_direction: str
    effect_scale: str
    continuity_correction: str
    estimate: float
    variance: float
    se: float


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    payload = source.read_bytes()
    if source.suffix.lower() in TEXT_HASH_EXTENSIONS:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_arm_event_rows(path: str | Path) -> list[ArmEventRow]:
    rows: list[ArmEventRow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = ArmEventRow(
                study_id=raw["study_id"].strip(),
                trial=raw["trial"].strip(),
                nct_id=raw["nct_id"].strip(),
                pmid=raw["pmid"].strip(),
                outcome_id=raw["outcome_id"].strip(),
                outcome_label=raw["outcome_label"].strip(),
                arm_role=raw["arm_role"].strip(),
                treatment=raw["treatment"].strip(),
                events=int(raw["events"]),
                n=int(raw["n"]),
            )
            _validate_arm_event_row(row)
            rows.append(row)

    if not rows:
        raise ValidationError("real-meta arm-event file is empty.")
    _validate_study_pairs(rows)
    return rows


def _validate_arm_event_row(row: ArmEventRow) -> None:
    if row.arm_role not in {"active", "control"}:
        raise ValidationError("arm_role must be 'active' or 'control'.")
    if row.events < 0 or row.n <= 0 or row.events > row.n:
        raise ValidationError("events must satisfy 0 <= events <= n.")

    validate_sources(
        [
            EvidenceSource(
                source_type="clinicaltrials_gov",
                identifier=row.nct_id,
                url=f"https://clinicaltrials.gov/study/{row.nct_id}",
                access_statement="ClinicalTrials.gov public registry record.",
            ),
            EvidenceSource(
                source_type="pubmed_abstract",
                identifier=row.pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{row.pmid}/",
                access_statement="PubMed abstract metadata and abstract text.",
            ),
        ]
    )


def _validate_study_pairs(rows: list[ArmEventRow]) -> None:
    by_study: dict[str, list[ArmEventRow]] = {}
    for row in rows:
        by_study.setdefault(row.study_id, []).append(row)

    for study_id, study_rows in by_study.items():
        roles = [row.arm_role for row in study_rows]
        if roles.count("active") != 1 or roles.count("control") != 1:
            raise ValidationError(f"{study_id}: expected exactly one active and one control arm.")

        _require_single_value(study_rows, "trial", study_id)
        _require_single_value(study_rows, "nct_id", study_id)
        _require_single_value(study_rows, "pmid", study_id)
        _require_single_value(study_rows, "outcome_label", study_id)

        outcomes = {row.outcome_id for row in study_rows}
        if len(outcomes) != 1:
            raise ValidationError(f"{study_id}: mixed outcome IDs are not allowed in one contrast.")
        treatments = {row.treatment for row in study_rows}
        if len(treatments) != len(study_rows):
            raise ValidationError(f"{study_id}: active and control arms must use distinct treatments.")


def _require_single_value(rows: list[ArmEventRow], field_name: str, study_id: str) -> None:
    values = {getattr(row, field_name) for row in rows}
    if len(values) != 1:
        raise ValidationError(f"{study_id}: mixed {field_name} values are not allowed in one contrast.")


def load_real_meta_source_manifest(path: str | Path) -> dict[str, Any]:
    """Load a machine-readable source manifest for real-meta rows."""

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != "real_meta_sources/v1":
        raise ValidationError("source manifest schema_version must be real_meta_sources/v1.")
    studies = payload.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValidationError("source manifest must contain a non-empty studies list.")
    allowed = set(payload.get("allowed_source_types", []))
    if allowed and not allowed <= ALLOWED_SOURCE_TYPES:
        raise ValidationError(f"source manifest contains disallowed source types: {sorted(allowed)}")
    return payload


def validate_real_meta_source_manifest(
    rows: list[ArmEventRow],
    manifest_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate source manifest identity and arm counts against loaded rows."""

    payload = load_real_meta_source_manifest(manifest_path)
    if dataset_path is not None and payload.get("dataset_sha256"):
        actual_sha = sha256_file(dataset_path)
        if actual_sha != payload["dataset_sha256"]:
            raise ValidationError("source manifest dataset_sha256 does not match dataset file.")

    by_study: dict[str, list[ArmEventRow]] = {}
    for row in rows:
        by_study.setdefault(row.study_id, []).append(row)

    manifest_studies = payload["studies"]
    manifest_ids = [str(study.get("study_id", "")) for study in manifest_studies]
    duplicates = sorted({study_id for study_id in manifest_ids if manifest_ids.count(study_id) > 1})
    if duplicates:
        raise ValidationError(f"source manifest contains duplicate study IDs: {duplicates}")
    if set(manifest_ids) != set(by_study):
        missing = sorted(set(by_study) - set(manifest_ids))
        extra = sorted(set(manifest_ids) - set(by_study))
        raise ValidationError(f"source manifest study mismatch; missing={missing}, extra={extra}")

    all_source_types: set[str] = set()
    source_count = 0
    for study in manifest_studies:
        study_id = str(study["study_id"])
        study_rows = by_study[study_id]
        canonical = study_rows[0]
        for field_name in ("trial", "nct_id", "pmid", "outcome_id", "outcome_label"):
            if str(study.get(field_name, "")) != getattr(canonical, field_name):
                raise ValidationError(f"{study_id}: source manifest {field_name} does not match CSV rows.")

        source_entries = study.get("sources")
        if not isinstance(source_entries, list) or not source_entries:
            raise ValidationError(f"{study_id}: source manifest requires at least one source.")
        source_records = [
            _source_record_from_manifest(study_id, canonical, source_entry)
            for source_entry in source_entries
        ]
        validate_ingestion_records(source_records)
        source_types = {record.source_type for record in source_records}
        all_source_types.update(source_types)
        source_count += len(source_records)
        if not {"clinicaltrials_gov", "pubmed_abstract"} <= source_types:
            raise ValidationError(f"{study_id}: source manifest requires CT.gov and PubMed sources.")

        event_count_source_type = str(
            study.get("event_count_source_type")
            or payload.get("event_count_source_type")
            or ""
        )
        if event_count_source_type not in source_types:
            raise ValidationError(
                f"{study_id}: event_count_source_type '{event_count_source_type}' is not one of the declared sources."
            )
        if event_count_source_type == "pubmed_abstract":
            _validate_source_terms(study_id, study.get("active_source_terms"), "active_source_terms")
            _validate_source_terms(study_id, study.get("control_source_terms"), "control_source_terms")
        _validate_manifest_arms(study_id, study_rows, study.get("arms"))

    return {
        "manifest_sha256": sha256_file(manifest_path),
        "n_studies": len(manifest_studies),
        "n_sources": source_count,
        "source_types": sorted(all_source_types),
    }


def _source_record_from_manifest(
    study_id: str,
    row: ArmEventRow,
    source_entry: dict[str, Any],
) -> EvidenceIngestionRecord:
    source_type = str(source_entry.get("source_type", ""))
    identifier = str(source_entry.get("identifier", "")).strip()
    if source_type == "clinicaltrials_gov":
        if not identifier:
            raise ValidationError(f"{study_id}: ClinicalTrials.gov source requires identifier.")
        if identifier != row.nct_id:
            raise ValidationError(f"{study_id}: ClinicalTrials.gov source identifier does not match CSV NCT ID.")
    if source_type == "pubmed_abstract":
        if not identifier:
            raise ValidationError(f"{study_id}: PubMed source requires identifier.")
        if identifier != row.pmid:
            raise ValidationError(f"{study_id}: PubMed source identifier does not match CSV PMID.")
    return EvidenceIngestionRecord(
        row_id=f"{study_id}:{source_type}",
        source_type=source_type,
        url=str(source_entry.get("url", "")),
        access_statement=str(source_entry.get("access_statement", "")),
        pmid=str(source_entry.get("pmid", row.pmid)) if source_type in {"pubmed_abstract", "open_access_paper"} else None,
        nct_id=str(source_entry.get("nct_id", row.nct_id)) if source_type == "clinicaltrials_gov" else None,
        pmcid=str(source_entry["pmcid"]) if "pmcid" in source_entry else None,
        doi=str(source_entry["doi"]) if "doi" in source_entry else None,
        source_text=str(source_entry.get("source_text", "")),
    )


def _validate_manifest_arms(
    study_id: str,
    study_rows: list[ArmEventRow],
    manifest_arms: Any,
) -> None:
    if not isinstance(manifest_arms, list):
        raise ValidationError(f"{study_id}: source manifest arms must be a list.")
    by_role = {str(arm.get("arm_role", "")): arm for arm in manifest_arms if isinstance(arm, dict)}
    if set(by_role) != {row.arm_role for row in study_rows}:
        raise ValidationError(f"{study_id}: source manifest arms do not match CSV arm roles.")
    for row in study_rows:
        arm = by_role[row.arm_role]
        for field_name in ("treatment", "events", "n"):
            if arm.get(field_name) != getattr(row, field_name):
                raise ValidationError(
                    f"{study_id}: source manifest arm {row.arm_role} {field_name} does not match CSV rows."
                )


def _validate_source_terms(study_id: str, terms: Any, field_name: str) -> None:
    if not isinstance(terms, list) or not terms:
        raise ValidationError(f"{study_id}: source manifest requires non-empty {field_name}.")
    if any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValidationError(f"{study_id}: {field_name} entries must be non-empty strings.")


def build_dataset_from_arm_events(rows: list[ArmEventRow]) -> EvidenceDataset:
    dataset = EvidenceDataset()
    for row in rows:
        if row.study_id not in dataset.studies:
            dataset.add_study(row.study_id, "rct")
        arm_id = row.arm_role
        dataset.add_arm(row.study_id, arm_id, row.treatment, row.n)
        dataset.add_outcome_ad(row.study_id, arm_id, row.outcome_id, "binary", row.events)
    return dataset


def study_log_or_effects(rows: list[ArmEventRow]) -> list[StudyLogOREffect]:
    """Compute source-backed study-level log-OR effects from validated two-arm rows."""

    _validate_study_pairs(rows)
    study_effects: list[StudyLogOREffect] = []
    by_study: dict[str, dict[str, ArmEventRow]] = {}
    for row in rows:
        by_study.setdefault(row.study_id, {})[row.arm_role] = row

    for study_id, arms in sorted(by_study.items()):
        active = arms["active"]
        control = arms["control"]
        if min(active.events, control.events, active.n - active.events, control.n - control.events) <= 0:
            raise ValidationError(f"{study_id}: zero-cell studies require an explicit correction policy.")

        log_or = math.log(
            (active.events / (active.n - active.events))
            / (control.events / (control.n - control.events))
        )
        variance = (
            1.0 / active.events
            + 1.0 / (active.n - active.events)
            + 1.0 / control.events
            + 1.0 / (control.n - control.events)
        )
        study_effects.append(
            StudyLogOREffect(
                study_id=study_id,
                trial=active.trial,
                nct_id=active.nct_id,
                pmid=active.pmid,
                outcome_id=active.outcome_id,
                outcome_label=active.outcome_label,
                active_treatment=active.treatment,
                control_treatment=control.treatment,
                active_events=active.events,
                active_n=active.n,
                control_events=control.events,
                control_n=control.n,
                effect_direction="active_vs_control",
                effect_scale="log_or",
                continuity_correction="none_zero_cells_fail_closed",
                estimate=float(log_or),
                variance=float(variance),
                se=math.sqrt(float(variance)),
            )
        )

    if not study_effects:
        raise ValidationError("at least one study-level effect is required.")
    return study_effects


def fixed_effect_log_or_reference(rows: list[ArmEventRow]) -> EffectEstimate:
    study_effects = study_log_or_effects(rows)
    effects = [effect.estimate for effect in study_effects]
    variances = [effect.variance for effect in study_effects]

    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    weights = 1.0 / v
    estimate = float(np.sum(weights * y) / np.sum(weights))
    se = math.sqrt(float(1.0 / np.sum(weights)))
    z = 1.959963984540054
    return EffectEstimate(
        estimate=estimate,
        se=se,
        ci_low=estimate - z * se,
        ci_high=estimate + z * se,
    )


def _pairwise_result_payload(result: PairwiseMetaResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": result.method,
        "estimate": result.estimate,
        "se": result.se,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "tau2": result.tau2,
        "q": result.q,
        "df": result.df,
        "hksj": result.hksj,
        "hksj_q_factor": result.hksj_q_factor,
        "weights": list(result.weights),
        "warnings": list(result.warnings),
    }
    if result.prediction_interval is not None:
        payload["pi_low"] = float(result.prediction_interval[0])
        payload["pi_high"] = float(result.prediction_interval[1])
    return payload


def run_real_meta_benchmark(
    path: str | Path,
    *,
    mcmc_samples: int = 600,
    source_manifest_path: str | Path | None = None,
    reference_treatment: str = "Placebo",
    candidate_treatment: str = "SGLT2i",
) -> dict[str, Any]:
    rows = load_arm_event_rows(path)
    source_manifest = None
    if source_manifest_path is not None:
        source_manifest = validate_real_meta_source_manifest(
            rows,
            source_manifest_path,
            dataset_path=path,
        )
    dataset = build_dataset_from_arm_events(rows)
    outcome_id = rows[0].outcome_id

    pooler = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        down_weight=False,
        random_effects=False,
        exact_binomial=False,
    )
    fit = pooler.fit(dataset, outcome_id, reference_treatment=reference_treatment)
    frequentist = fit.contrast(candidate_treatment, reference_treatment, alpha=0.05)
    reference = fixed_effect_log_or_reference(rows)
    study_effects = study_log_or_effects(rows)
    pairwise_effects = np.asarray([effect.estimate for effect in study_effects], dtype=float)
    pairwise_variances = np.asarray([effect.variance for effect in study_effects], dtype=float)
    pairwise_fixed = fit_pairwise_meta(pairwise_effects, pairwise_variances, method="FE")
    pairwise_reml_hksj = fit_pairwise_meta(
        pairwise_effects,
        pairwise_variances,
        method="REML",
        hksj=True,
        hksj_floor=True,
        prediction_interval=True,
    )

    blocks = pooler._build_study_blocks(dataset, outcome_id, "binary")
    param_names = pooler._build_parameter_names((candidate_treatment,), (), [], False, ())
    y, x, v = pooler._assemble_design(
        blocks,
        param_names,
        reference_treatment=reference_treatment,
        reference_design="rct",
        cov_names=[],
        study_specific_bias=False,
    )
    sampler = BayesianNMAMCMCSampler(
        n_samples=mcmc_samples,
        burn_in=max(100, mcmc_samples // 4),
        thinning=1,
        proposal_sd_beta=0.03,
        proposal_sd_tau=0.01,
    )
    bayes = sampler.fit(
        y=y,
        x=x,
        v=v,
        param_names=param_names,
        blocks=blocks,
        unique_designs=["rct"],
        design_to_idx={"rct": 0},
        bias_prior_sd=1.0,
        treatment_shrinkage_lambda=0.0,
        treatment_centralities={"Placebo": 1.0, "SGLT2i": 1.0},
        seed=20260715,
    )

    return {
        "dataset_sha256": sha256_file(path),
        "n_studies": len({row.study_id for row in rows}),
        "n_arms": len(rows),
        "outcome_id": outcome_id,
        "effect_scale": "log_or",
        "model_config": {
            "reference_treatment": reference_treatment,
            "candidate_treatment": candidate_treatment,
            "pairwise_fixed_effect_method": "FE",
            "pairwise_random_effect_method": "REML",
            "pairwise_hksj": True,
            "pairwise_hksj_floor": True,
            "pairwise_prediction_interval": True,
            "pairwise_prediction_interval_df": "k_minus_1",
            "level": 0.95,
            "bayesian_samples": mcmc_samples,
            "bayesian_seed": 20260715,
        },
        "source_manifest": source_manifest,
        "study_effects": [asdict(effect) for effect in study_effects],
        "reference": asdict(reference),
        "frequentist": {
            "estimate": float(frequentist[0]),
            "se": float(frequentist[1]),
            "ci_low": float(frequentist[2]),
            "ci_high": float(frequentist[3]),
            "warnings": list(fit.warnings),
        },
        "pairwise": {
            "fixed_effect": _pairwise_result_payload(pairwise_fixed),
            "reml_hksj": _pairwise_result_payload(pairwise_reml_hksj),
        },
        "bayesian": {
            "posterior_mean": float(bayes.posterior_means[f"trt_{candidate_treatment}"]),
            "posterior_sd": float(bayes.posterior_sds[f"trt_{candidate_treatment}"]),
            "credible_interval": bayes.credible_intervals[f"trt_{candidate_treatment}"],
            "acceptance_rate": float(bayes.acceptance_rate),
        },
    }
