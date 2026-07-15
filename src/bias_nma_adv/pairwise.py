"""Experimental pairwise meta-analysis bridge.

The purpose of this module is to make pairwise conventions explicit before
larger NMA validation work depends on them. It provides deterministic fixed-
and random-effects pooling, heterogeneity estimators, HKSJ scaling, and
prediction intervals. External `metafor`/`meta` parity remains a planned
validation target until independent artifacts are added.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.optimize
import scipy.stats


class PairwiseMetaError(ValueError):
    """Raised when pairwise meta-analysis input or options are invalid."""


@dataclass(frozen=True)
class PairwiseMetaResult:
    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    tau2: float
    q: float
    df: int
    hksj: bool
    hksj_q_factor: float
    prediction_interval: tuple[float, float] | None
    weights: tuple[float, ...]
    warnings: tuple[str, ...]


def fit_pairwise_meta(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    method: str = "REML",
    hksj: bool = False,
    hksj_floor: bool = True,
    prediction_interval: bool = False,
    level: float = 0.95,
) -> PairwiseMetaResult:
    """Fit an intercept-only pairwise meta-analysis model.

    Parameters
    ----------
    effects:
        Study-level effects on a prespecified analysis scale.
    variances:
        Within-study variances for `effects`.
    method:
        One of FE/common, DL, PM/Paule-Mandel, or REML.
    hksj:
        Apply Hartung-Knapp-Sidik-Jonkman scaling to the pooled-effect SE.
    hksj_floor:
        If true, use `max(1, Q / (k - 1))`; if false, use raw `Q / (k - 1)`.
    prediction_interval:
        If true and k >= 2, return a t-based prediction interval using
        `sqrt(se^2 + tau2)`.
    """

    y, v = _validate_effects(effects, variances)
    if not (0.0 < level < 1.0):
        raise PairwiseMetaError("level must be in (0, 1).")

    method_key = _normalise_method(method)
    warnings: list[str] = []
    if len(y) == 1:
        tau2 = 0.0
        if method_key != "FE":
            warnings.append("Only one study supplied; tau2 fixed at 0.0.")
        if hksj:
            warnings.append("HKSJ scaling requires at least two studies; using z interval.")
        hksj = False
    elif method_key == "FE":
        tau2 = 0.0
    elif method_key == "DL":
        tau2 = _tau2_der_simonian_laird(y, v)
    elif method_key == "PM":
        tau2 = _tau2_paule_mandel(y, v)
    elif method_key == "REML":
        tau2 = _tau2_reml(y, v)
    else:  # pragma: no cover - _normalise_method guards this
        raise PairwiseMetaError(f"unsupported method '{method}'.")

    weights = 1.0 / (v + tau2)
    estimate = float(np.sum(weights * y) / np.sum(weights))
    base_se = math.sqrt(float(1.0 / np.sum(weights)))
    q = _q_statistic(y, weights, estimate)
    df = len(y) - 1

    q_factor = 1.0
    se = base_se
    use_t = False
    if hksj and df > 0:
        raw_factor = q / df
        q_factor = max(1.0, raw_factor) if hksj_floor else raw_factor
        se = math.sqrt(max(q_factor, 0.0)) * base_se
        use_t = True

    alpha = 1.0 - level
    if use_t:
        critical = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
    else:
        critical = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = estimate - critical * se
    ci_high = estimate + critical * se

    pred = None
    if prediction_interval:
        if df <= 0:
            warnings.append("Prediction interval requires at least two studies.")
        else:
            pred_critical = scipy.stats.t.ppf(1.0 - alpha / 2.0, df)
            pred_se = math.sqrt(max(se * se + tau2, 0.0))
            pred = (estimate - pred_critical * pred_se, estimate + pred_critical * pred_se)

    return PairwiseMetaResult(
        method=method_key,
        estimate=estimate,
        se=se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        tau2=float(tau2),
        q=float(q),
        df=max(df, 0),
        hksj=bool(hksj),
        hksj_q_factor=float(q_factor),
        prediction_interval=pred,
        weights=tuple(float(w) for w in weights),
        warnings=tuple(warnings),
    )


def _validate_effects(effects: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(effects, dtype=float).reshape(-1)
    v = np.asarray(variances, dtype=float).reshape(-1)
    if y.size == 0:
        raise PairwiseMetaError("at least one study effect is required.")
    if y.shape != v.shape:
        raise PairwiseMetaError("effects and variances must have the same length.")
    if not np.all(np.isfinite(y)):
        raise PairwiseMetaError("all effects must be finite.")
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise PairwiseMetaError("all variances must be finite and positive.")
    return y, v


def _normalise_method(method: str) -> str:
    key = method.strip().upper().replace("-", "_")
    aliases = {
        "COMMON": "FE",
        "COMMON_EFFECT": "FE",
        "FIXED": "FE",
        "FIXED_EFFECT": "FE",
        "FE": "FE",
        "DL": "DL",
        "DERSIMONIAN_LAIRD": "DL",
        "PM": "PM",
        "PAULE_MANDEL": "PM",
        "REML": "REML",
    }
    if key not in aliases:
        raise PairwiseMetaError("method must be FE, DL, PM, or REML.")
    return aliases[key]


def _tau2_der_simonian_laird(y: np.ndarray, v: np.ndarray) -> float:
    fixed_weights = 1.0 / v
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    q = _q_statistic(y, fixed_weights, fixed_mean)
    df = len(y) - 1
    c_value = float(np.sum(fixed_weights) - np.sum(fixed_weights * fixed_weights) / np.sum(fixed_weights))
    return max(0.0, (q - df) / c_value) if c_value > 0.0 else 0.0


def _tau2_paule_mandel(y: np.ndarray, v: np.ndarray) -> float:
    df = len(y) - 1
    if _q_at_tau2(y, v, 0.0) <= df:
        return 0.0
    upper = _upper_tau2_for_root(y, v, df)
    return float(scipy.optimize.brentq(lambda tau: _q_at_tau2(y, v, tau) - df, 0.0, upper))


def _tau2_reml(y: np.ndarray, v: np.ndarray) -> float:
    if len(y) <= 1:
        return 0.0
    upper = _upper_tau2_for_minimization(y, v)
    result = scipy.optimize.minimize_scalar(
        lambda tau: _restricted_log_likelihood(y, v, tau),
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise PairwiseMetaError("REML tau2 optimization failed.")
    value = float(result.x)
    return 0.0 if value < 1e-10 else value


def _q_at_tau2(y: np.ndarray, v: np.ndarray, tau2: float) -> float:
    weights = 1.0 / (v + tau2)
    mean = float(np.sum(weights * y) / np.sum(weights))
    return _q_statistic(y, weights, mean)


def _q_statistic(y: np.ndarray, weights: np.ndarray, mean: float) -> float:
    residual = y - mean
    return float(np.sum(weights * residual * residual))


def _restricted_log_likelihood(y: np.ndarray, v: np.ndarray, tau2: float) -> float:
    total_v = v + tau2
    weights = 1.0 / total_v
    mean = float(np.sum(weights * y) / np.sum(weights))
    q = _q_statistic(y, weights, mean)
    return 0.5 * (float(np.sum(np.log(total_v))) + math.log(float(np.sum(weights))) + q)


def _upper_tau2_for_root(y: np.ndarray, v: np.ndarray, df: int) -> float:
    upper = max(float(np.var(y, ddof=1)), float(np.max(v)), 1.0)
    while _q_at_tau2(y, v, upper) > df:
        upper *= 2.0
        if upper > 1e6:
            raise PairwiseMetaError("could not bracket Paule-Mandel tau2 root.")
    return upper


def _upper_tau2_for_minimization(y: np.ndarray, v: np.ndarray) -> float:
    scale = max(float(np.var(y, ddof=1)), float(np.max(v)), 1.0)
    return scale * 100.0
