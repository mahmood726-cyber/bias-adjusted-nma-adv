"""Experimental diagnostic test accuracy meta-analysis helpers.

This module implements a small bivariate logit-normal DTA prototype for
algorithmic validation. It is not a replacement for mature DTA software such as
``mada::reitsma`` and it does not make source-backed diagnostic claims by
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize


DTA_MODEL_SCHEMA_VERSION = "dta_bivariate_logitnormal_reml/v1"
MIN_DTA_STUDIES = 5


class DTAError(ValueError):
    """Raised when DTA inputs or model fitting are invalid."""


@dataclass(frozen=True)
class DTAStudy:
    """One 2x2 diagnostic accuracy study."""

    study_id: str
    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DTAStudy":
        keys = {"study_id", "tp", "fp", "fn", "tn"}
        if "study" in raw and "study_id" not in raw:
            raw = {**raw, "study_id": raw["study"]}
        missing = sorted(keys - set(raw))
        if missing:
            raise DTAError(f"DTA study is missing required keys: {missing}")
        study = cls(
            study_id=str(raw["study_id"]),
            tp=_as_count(raw["tp"], "tp"),
            fp=_as_count(raw["fp"], "fp"),
            fn=_as_count(raw["fn"], "fn"),
            tn=_as_count(raw["tn"], "tn"),
        )
        study.validate()
        return study

    def validate(self) -> None:
        if not self.study_id.strip():
            raise DTAError("DTA study_id must not be empty.")
        for field in ("tp", "fp", "fn", "tn"):
            if getattr(self, field) < 0:
                raise DTAError(f"DTA {field} count cannot be negative.")
        if self.tp + self.fn <= 0:
            raise DTAError(f"{self.study_id}: diseased denominator must be positive.")
        if self.tn + self.fp <= 0:
            raise DTAError(f"{self.study_id}: non-diseased denominator must be positive.")


@dataclass(frozen=True)
class DTATransformedStudy:
    """Logit-scale sensitivity and false-positive-rate input."""

    study_id: str
    logit_sensitivity: float
    logit_fpr: float
    var_logit_sensitivity: float
    var_logit_fpr: float
    sensitivity: float
    specificity: float
    continuity_correction: float


@dataclass(frozen=True)
class BivariateDTAFit:
    """Bivariate logit-normal REML prototype fit."""

    schema_version: str
    method: str
    n_studies: int
    converged: bool
    objective: float
    pooled_sensitivity: float
    pooled_sensitivity_ci: tuple[float, float]
    pooled_specificity: float
    pooled_specificity_ci: tuple[float, float]
    logit_sensitivity: float
    logit_fpr: float
    se_logit_sensitivity: float
    se_logit_fpr: float
    tau2_sensitivity: float
    tau2_fpr: float
    cov_sensitivity_fpr: float
    rho_sensitivity_fpr: float
    log_diagnostic_odds_ratio: float
    diagnostic_odds_ratio: float
    auc_trapezoid: float
    warnings: tuple[str, ...]
    optimizer_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "n_studies": self.n_studies,
            "converged": self.converged,
            "objective": self.objective,
            "pooled_sensitivity": self.pooled_sensitivity,
            "pooled_sensitivity_ci": list(self.pooled_sensitivity_ci),
            "pooled_specificity": self.pooled_specificity,
            "pooled_specificity_ci": list(self.pooled_specificity_ci),
            "logit_sensitivity": self.logit_sensitivity,
            "logit_fpr": self.logit_fpr,
            "se_logit_sensitivity": self.se_logit_sensitivity,
            "se_logit_fpr": self.se_logit_fpr,
            "tau2_sensitivity": self.tau2_sensitivity,
            "tau2_fpr": self.tau2_fpr,
            "cov_sensitivity_fpr": self.cov_sensitivity_fpr,
            "rho_sensitivity_fpr": self.rho_sensitivity_fpr,
            "log_diagnostic_odds_ratio": self.log_diagnostic_odds_ratio,
            "diagnostic_odds_ratio": self.diagnostic_odds_ratio,
            "auc_trapezoid": self.auc_trapezoid,
            "warnings": list(self.warnings),
            "optimizer_message": self.optimizer_message,
        }


def transform_dta_studies(
    studies: Iterable[DTAStudy | Mapping[str, Any]],
    *,
    continuity_correction: float = 0.5,
) -> tuple[DTATransformedStudy, ...]:
    """Transform 2x2 DTA counts to logit sensitivity and logit FPR."""

    if continuity_correction <= 0 or not math.isfinite(continuity_correction):
        raise DTAError("continuity_correction must be positive and finite.")

    parsed = _parse_studies(studies)
    transformed: list[DTATransformedStudy] = []
    for study in parsed:
        tp, fp, fn, tn = (
            float(study.tp),
            float(study.fp),
            float(study.fn),
            float(study.tn),
        )
        apply_correction = any(value == 0.0 for value in (tp, fp, fn, tn))
        correction = continuity_correction if apply_correction else 0.0
        tp += correction
        fp += correction
        fn += correction
        tn += correction
        sensitivity = tp / (tp + fn)
        fpr = fp / (fp + tn)
        specificity = 1.0 - fpr
        transformed.append(
            DTATransformedStudy(
                study_id=study.study_id,
                logit_sensitivity=math.log(tp / fn),
                logit_fpr=math.log(fp / tn),
                var_logit_sensitivity=(1.0 / tp) + (1.0 / fn),
                var_logit_fpr=(1.0 / fp) + (1.0 / tn),
                sensitivity=sensitivity,
                specificity=specificity,
                continuity_correction=correction,
            )
        )
    return tuple(transformed)


def fit_bivariate_dta_reml(
    studies: Iterable[DTAStudy | Mapping[str, Any]],
    *,
    continuity_correction: float = 0.5,
    min_studies: int = MIN_DTA_STUDIES,
) -> BivariateDTAFit:
    """Fit a bivariate logit-normal random-effects DTA prototype.

    The model is fit with a profiled REML objective over the between-study
    covariance matrix. It is intended for method-contract validation and
    comparison against a reference package on fixtures, not for clinical DTA
    certification.
    """

    transformed = transform_dta_studies(
        studies,
        continuity_correction=continuity_correction,
    )
    if len(transformed) < min_studies:
        raise DTAError(
            f"bivariate DTA fitting requires at least {min_studies} studies; "
            f"received {len(transformed)}."
        )

    y = np.array(
        [[row.logit_sensitivity, row.logit_fpr] for row in transformed],
        dtype=float,
    )
    variances = np.array(
        [[row.var_logit_sensitivity, row.var_logit_fpr] for row in transformed],
        dtype=float,
    )
    starts = _optimizer_starts(y, variances)
    best = None
    for start in starts:
        result = minimize(
            _profile_reml_objective,
            start,
            args=(y, variances),
            method="L-BFGS-B",
            bounds=((-8.0, 3.0), (-8.0, 3.0), (-3.0, 3.0)),
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result

    if best is None:
        raise DTAError("DTA optimizer did not run.")

    psi = _psi_from_params(np.asarray(best.x, dtype=float))
    mu, cov_mu, objective = _profile_mu_and_covariance(psi, y, variances)
    tau_sens = math.sqrt(max(float(psi[0, 0]), 0.0))
    tau_fpr = math.sqrt(max(float(psi[1, 1]), 0.0))
    denom = tau_sens * tau_fpr
    rho = 0.0 if denom == 0.0 else float(np.clip(psi[0, 1] / denom, -0.999, 0.999))
    se_mu = np.sqrt(np.maximum(np.diag(cov_mu), 0.0))
    z = 1.959963984540054

    pooled_sens = _inv_logit(float(mu[0]))
    pooled_fpr = _inv_logit(float(mu[1]))
    pooled_spec = 1.0 - pooled_fpr
    sens_ci = (
        _inv_logit(float(mu[0] - z * se_mu[0])),
        _inv_logit(float(mu[0] + z * se_mu[0])),
    )
    fpr_ci = (
        _inv_logit(float(mu[1] - z * se_mu[1])),
        _inv_logit(float(mu[1] + z * se_mu[1])),
    )
    spec_ci = (1.0 - fpr_ci[1], 1.0 - fpr_ci[0])
    log_dor = float(mu[0] - mu[1])
    warnings = [
        "Experimental bivariate DTA prototype; not source-backed clinical evidence.",
        "Reference matching and source-backed TP/FP/FN/TN manifests are required before DTA claims.",
    ]
    if not bool(best.success):
        warnings.append("Optimizer did not report convergence.")

    return BivariateDTAFit(
        schema_version=DTA_MODEL_SCHEMA_VERSION,
        method="bivariate_logitnormal_reml_prototype",
        n_studies=len(transformed),
        converged=bool(best.success),
        objective=float(objective),
        pooled_sensitivity=pooled_sens,
        pooled_sensitivity_ci=sens_ci,
        pooled_specificity=pooled_spec,
        pooled_specificity_ci=spec_ci,
        logit_sensitivity=float(mu[0]),
        logit_fpr=float(mu[1]),
        se_logit_sensitivity=float(se_mu[0]),
        se_logit_fpr=float(se_mu[1]),
        tau2_sensitivity=float(psi[0, 0]),
        tau2_fpr=float(psi[1, 1]),
        cov_sensitivity_fpr=float(psi[0, 1]),
        rho_sensitivity_fpr=rho,
        log_diagnostic_odds_ratio=log_dor,
        diagnostic_odds_ratio=float(math.exp(np.clip(log_dor, -700.0, 700.0))),
        auc_trapezoid=_conditional_sroc_auc(mu, psi),
        warnings=tuple(warnings),
        optimizer_message=str(best.message),
    )


def _parse_studies(studies: Iterable[DTAStudy | Mapping[str, Any]]) -> tuple[DTAStudy, ...]:
    parsed: list[DTAStudy] = []
    for raw in studies:
        study = raw if isinstance(raw, DTAStudy) else DTAStudy.from_mapping(raw)
        study.validate()
        parsed.append(study)
    if not parsed:
        raise DTAError("at least one DTA study is required.")
    ids = [study.study_id for study in parsed]
    duplicates = sorted({study_id for study_id in ids if ids.count(study_id) > 1})
    if duplicates:
        raise DTAError(f"duplicate DTA study_id values: {duplicates}")
    return tuple(parsed)


def _profile_reml_objective(params: np.ndarray, y: np.ndarray, variances: np.ndarray) -> float:
    try:
        psi = _psi_from_params(params)
        _, _, objective = _profile_mu_and_covariance(psi, y, variances)
    except (DTAError, np.linalg.LinAlgError, FloatingPointError, ValueError):
        return float("inf")
    return objective


def _profile_mu_and_covariance(
    psi: np.ndarray,
    y: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    information = np.zeros((2, 2), dtype=float)
    weighted_y = np.zeros(2, dtype=float)
    inv_covariances: list[np.ndarray] = []
    logdet_sum = 0.0
    for row_variance, row_y in zip(variances, y):
        sigma = psi + np.diag(row_variance)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0 or not math.isfinite(logdet):
            raise DTAError("DTA covariance matrix is not positive definite.")
        inv_sigma = np.linalg.inv(sigma)
        inv_covariances.append(inv_sigma)
        information += inv_sigma
        weighted_y += inv_sigma @ row_y
        logdet_sum += float(logdet)

    info_sign, info_logdet = np.linalg.slogdet(information)
    if info_sign <= 0 or not math.isfinite(info_logdet):
        raise DTAError("DTA REML information matrix is not positive definite.")
    cov_mu = np.linalg.inv(information)
    mu = cov_mu @ weighted_y
    quad = 0.0
    for inv_sigma, row_y in zip(inv_covariances, y):
        resid = row_y - mu
        quad += float(resid.T @ inv_sigma @ resid)
    n_observations = y.shape[0] * 2
    n_fixed = 2
    objective = 0.5 * (
        logdet_sum
        + float(info_logdet)
        + quad
        + (n_observations - n_fixed) * math.log(2.0 * math.pi)
    )
    if not math.isfinite(objective):
        raise DTAError("DTA REML objective is not finite.")
    return mu, cov_mu, objective


def _psi_from_params(params: np.ndarray) -> np.ndarray:
    tau_sens = math.exp(float(params[0]))
    tau_fpr = math.exp(float(params[1]))
    rho = math.tanh(float(params[2]))
    cov = rho * tau_sens * tau_fpr
    return np.array(
        [[tau_sens * tau_sens, cov], [cov, tau_fpr * tau_fpr]],
        dtype=float,
    )


def _optimizer_starts(y: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, ...]:
    tau2_sens = _dersimonian_laird_tau2(y[:, 0], variances[:, 0])
    tau2_fpr = _dersimonian_laird_tau2(y[:, 1], variances[:, 1])
    if y.shape[0] > 2 and np.std(y[:, 0]) > 0 and np.std(y[:, 1]) > 0:
        rho = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    else:
        rho = 0.0
    rho = float(np.clip(rho, -0.8, 0.8))
    starts = [
        (
            math.log(max(math.sqrt(tau2_sens), 1e-3)),
            math.log(max(math.sqrt(tau2_fpr), 1e-3)),
            math.atanh(rho),
        ),
        (math.log(0.02), math.log(0.02), 0.0),
        (math.log(0.10), math.log(0.10), 0.0),
        (math.log(0.30), math.log(0.30), math.atanh(0.3)),
        (math.log(0.30), math.log(0.30), math.atanh(-0.3)),
    ]
    return tuple(np.array(start, dtype=float) for start in starts)


def _dersimonian_laird_tau2(y: np.ndarray, variance: np.ndarray) -> float:
    weights = 1.0 / np.maximum(variance, 1e-12)
    sum_weights = float(np.sum(weights))
    if sum_weights <= 0.0:
        return 0.0
    mu = float(np.sum(weights * y) / sum_weights)
    q_stat = float(np.sum(weights * np.square(y - mu)))
    c_value = sum_weights - float(np.sum(np.square(weights)) / sum_weights)
    if c_value <= 0.0:
        return 0.0
    return max(0.0, (q_stat - (len(y) - 1)) / c_value)


def _conditional_sroc_auc(mu: np.ndarray, psi: np.ndarray, *, n_grid: int = 401) -> float:
    fpr = np.linspace(0.001, 0.999, n_grid)
    logit_fpr = np.log(fpr / (1.0 - fpr))
    if psi[1, 1] > 1e-12:
        slope = float(psi[0, 1] / psi[1, 1])
    else:
        slope = 0.0
    logit_sens = float(mu[0]) + slope * (logit_fpr - float(mu[1]))
    sens = 1.0 / (1.0 + np.exp(-np.clip(logit_sens, -700.0, 700.0)))
    return float(np.trapezoid(sens, fpr))


def _inv_logit(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _as_count(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise DTAError(f"DTA {field} count must be an integer, not bool.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise DTAError(f"DTA {field} count must be an integer.") from exc
    if float(value) != float(numeric):
        raise DTAError(f"DTA {field} count must be an integer.")
    return numeric
