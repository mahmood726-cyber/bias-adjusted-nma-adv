"""Real meta-analysis benchmark helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from bias_nma_adv.bayesian import BayesianNMAMCMCSampler
from bias_nma_adv.data import EvidenceDataset, ValidationError
from bias_nma_adv.evidence_sources import EvidenceSource, validate_sources
from bias_nma_adv.model import AdvancedBiasAdjustedNMAPooler


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_dataset_from_arm_events(rows: list[ArmEventRow]) -> EvidenceDataset:
    dataset = EvidenceDataset()
    for row in rows:
        if row.study_id not in dataset.studies:
            dataset.add_study(row.study_id, "rct")
        arm_id = row.arm_role
        dataset.add_arm(row.study_id, arm_id, row.treatment, row.n)
        dataset.add_outcome_ad(row.study_id, arm_id, row.outcome_id, "binary", row.events)
    return dataset


def fixed_effect_log_or_reference(rows: list[ArmEventRow]) -> EffectEstimate:
    effects = []
    variances = []
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
        effects.append(log_or)
        variances.append(variance)

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


def run_real_meta_benchmark(path: str | Path, *, mcmc_samples: int = 600) -> dict[str, Any]:
    rows = load_arm_event_rows(path)
    dataset = build_dataset_from_arm_events(rows)
    outcome_id = rows[0].outcome_id

    pooler = AdvancedBiasAdjustedNMAPooler(
        hksj=False,
        down_weight=False,
        random_effects=False,
        exact_binomial=False,
    )
    fit = pooler.fit(dataset, outcome_id, reference_treatment="Placebo")
    frequentist = fit.contrast("SGLT2i", "Placebo", alpha=0.05)
    reference = fixed_effect_log_or_reference(rows)

    blocks = pooler._build_study_blocks(dataset, outcome_id, "binary")
    param_names = pooler._build_parameter_names(("SGLT2i",), (), [], False, ())
    y, x, v = pooler._assemble_design(
        blocks,
        param_names,
        reference_treatment="Placebo",
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
        "reference": reference.__dict__,
        "frequentist": {
            "estimate": float(frequentist[0]),
            "se": float(frequentist[1]),
            "ci_low": float(frequentist[2]),
            "ci_high": float(frequentist[3]),
            "warnings": list(fit.warnings),
        },
        "bayesian": {
            "posterior_mean": float(bayes.posterior_means["trt_SGLT2i"]),
            "posterior_sd": float(bayes.posterior_sds["trt_SGLT2i"]),
            "credible_interval": bayes.credible_intervals["trt_SGLT2i"],
            "acceptance_rate": float(bayes.acceptance_rate),
        },
    }
