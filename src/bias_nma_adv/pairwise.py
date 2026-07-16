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


@dataclass(frozen=True)
class LeaveOneOutDiagnostic:
    """One leave-one-out refit in a pairwise outlier-space diagnostic."""

    omitted_index: int
    omitted_effect: float
    omitted_variance: float
    estimate: float
    se: float
    tau2: float
    q: float
    delta_estimate: float
    absolute_delta_estimate: float
    standardized_delta: float


@dataclass(frozen=True)
class PairwiseOutlierSpaceDiagnostic:
    """Diagnostic-only leave-one-out outlier-space summary."""

    method: str
    full_estimate: float
    full_se: float
    full_tau2: float
    full_q: float
    diagnostics: tuple[LeaveOneOutDiagnostic, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseMethodDiagnostic:
    """Structured result or failure for one pairwise method fit."""

    method: str
    status: str
    estimate: float | None
    se: float | None
    ci_low: float | None
    ci_high: float | None
    tau2: float | None
    q: float | None
    df: int | None
    warnings: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class PairwiseTau2CrossCheckReport:
    """Cross-check pairwise estimates under alternative tau2 estimators."""

    k: int
    primary_method: str
    diagnostics: tuple[PairwiseMethodDiagnostic, ...]
    tau2_min: float | None
    tau2_max: float | None
    max_abs_estimate_delta: float | None
    max_abs_se_delta: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseNumericalStressReport:
    """Deterministic numerical-stability screen for pairwise meta-analysis."""

    k: int
    status: str
    min_variance: float
    max_variance: float
    max_weight_fraction: float
    cross_check: PairwiseTau2CrossCheckReport
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TrimAndFillSensitivity:
    """Bounded trim-and-fill-style sensitivity analysis for funnel asymmetry."""

    k_observed: int
    k_filled: int
    fill_side: str
    center: float
    observed_estimate: float
    adjusted_estimate: float
    observed_ci: tuple[float, float]
    adjusted_ci: tuple[float, float]
    filled_effects: tuple[float, ...]
    filled_variances: tuple[float, ...]
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


def leave_one_out_outlier_diagnostic(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    method: str = "REML",
    hksj: bool = False,
    hksj_floor: bool = True,
) -> PairwiseOutlierSpaceDiagnostic:
    """Run deterministic leave-one-out refits as a GOSH-style smoke diagnostic.

    This is not a full GOSH subset search. It is a bounded outlier-space screen
    that reports how much each single study changes the pooled estimate.
    """

    y, v = _validate_effects(effects, variances)
    if y.size < 3:
        raise PairwiseMetaError("leave-one-out diagnostics require at least three studies.")

    full = fit_pairwise_meta(
        y,
        v,
        method=method,
        hksj=hksj,
        hksj_floor=hksj_floor,
    )
    diagnostics: list[LeaveOneOutDiagnostic] = []
    for omitted_index in range(y.size):
        keep = np.ones(y.size, dtype=bool)
        keep[omitted_index] = False
        refit = fit_pairwise_meta(
            y[keep],
            v[keep],
            method=method,
            hksj=hksj,
            hksj_floor=hksj_floor,
        )
        delta = float(refit.estimate - full.estimate)
        diagnostics.append(
            LeaveOneOutDiagnostic(
                omitted_index=int(omitted_index),
                omitted_effect=float(y[omitted_index]),
                omitted_variance=float(v[omitted_index]),
                estimate=float(refit.estimate),
                se=float(refit.se),
                tau2=float(refit.tau2),
                q=float(refit.q),
                delta_estimate=delta,
                absolute_delta_estimate=abs(delta),
                standardized_delta=float(abs(delta) / max(full.se, 1e-12)),
            )
        )

    warnings = list(full.warnings)
    warnings.append(
        "Leave-one-out outlier-space diagnostics are screening tools and do not replace full GOSH analyses."
    )
    return PairwiseOutlierSpaceDiagnostic(
        method=full.method,
        full_estimate=float(full.estimate),
        full_se=float(full.se),
        full_tau2=float(full.tau2),
        full_q=float(full.q),
        diagnostics=tuple(diagnostics),
        warnings=tuple(warnings),
    )


def tau2_cross_check_report(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    methods: tuple[str, ...] = ("FE", "DL", "PM", "REML"),
    primary_method: str = "REML",
    hksj: bool = False,
    hksj_floor: bool = True,
) -> PairwiseTau2CrossCheckReport:
    """Fit multiple tau2 conventions and report deterministic differences."""

    y, v = _validate_effects(effects, variances)
    primary = _normalise_method(primary_method)
    diagnostics: list[PairwiseMethodDiagnostic] = []
    for method in methods:
        method_key = _normalise_method(method)
        try:
            result = fit_pairwise_meta(
                y,
                v,
                method=method_key,
                hksj=hksj,
                hksj_floor=hksj_floor,
            )
        except PairwiseMetaError as exc:
            diagnostics.append(
                PairwiseMethodDiagnostic(
                    method=method_key,
                    status="failed",
                    estimate=None,
                    se=None,
                    ci_low=None,
                    ci_high=None,
                    tau2=None,
                    q=None,
                    df=None,
                    warnings=(),
                    error=str(exc),
                )
            )
            continue
        diagnostics.append(
            PairwiseMethodDiagnostic(
                method=result.method,
                status="passed",
                estimate=float(result.estimate),
                se=float(result.se),
                ci_low=float(result.ci_low),
                ci_high=float(result.ci_high),
                tau2=float(result.tau2),
                q=float(result.q),
                df=int(result.df),
                warnings=tuple(result.warnings),
                error=None,
            )
        )

    passed = [item for item in diagnostics if item.status == "passed"]
    warnings: list[str] = []
    if not any(item.method == primary and item.status == "passed" for item in diagnostics):
        warnings.append(f"Primary method {primary} did not produce a passed fit.")
    if len(passed) < 2:
        warnings.append("Fewer than two tau2 methods passed; cross-check is weak.")

    tau2_values = [float(item.tau2) for item in passed if item.tau2 is not None]
    estimates = [float(item.estimate) for item in passed if item.estimate is not None]
    ses = [float(item.se) for item in passed if item.se is not None]
    tau2_min = min(tau2_values) if tau2_values else None
    tau2_max = max(tau2_values) if tau2_values else None
    max_abs_estimate_delta = (
        max(estimates) - min(estimates) if len(estimates) >= 2 else None
    )
    max_abs_se_delta = max(ses) - min(ses) if len(ses) >= 2 else None
    if tau2_min is not None and tau2_max is not None and tau2_max > tau2_min:
        warnings.append("Alternative tau2 estimators produce different heterogeneity estimates.")

    return PairwiseTau2CrossCheckReport(
        k=int(y.size),
        primary_method=primary,
        diagnostics=tuple(diagnostics),
        tau2_min=tau2_min,
        tau2_max=tau2_max,
        max_abs_estimate_delta=max_abs_estimate_delta,
        max_abs_se_delta=max_abs_se_delta,
        warnings=tuple(warnings),
    )


def numerical_stress_report(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    methods: tuple[str, ...] = ("FE", "DL", "PM", "REML"),
    dominant_weight_threshold: float = 0.80,
) -> PairwiseNumericalStressReport:
    """Return deterministic sparse/dominant-study and tau2 stress diagnostics."""

    y, v = _validate_effects(effects, variances)
    if not (0.0 < dominant_weight_threshold < 1.0):
        raise PairwiseMetaError("dominant_weight_threshold must be in (0, 1).")

    fixed_weights = 1.0 / v
    max_weight_fraction = float(np.max(fixed_weights / np.sum(fixed_weights)))
    cross_check = tau2_cross_check_report(y, v, methods=methods)
    warnings: list[str] = list(cross_check.warnings)
    if y.size < 3:
        warnings.append("Sparse pairwise evidence: fewer than three studies.")
    if max_weight_fraction >= dominant_weight_threshold:
        warnings.append(
            "Dominant-study warning: one study contributes at least "
            f"{dominant_weight_threshold:.0%} of fixed-effect precision."
        )
    if cross_check.tau2_min == 0.0 and cross_check.tau2_max and cross_check.tau2_max > 0.0:
        warnings.append("Heterogeneity estimate is boundary-sensitive across tau2 methods.")

    return PairwiseNumericalStressReport(
        k=int(y.size),
        status="warning" if warnings else "passed",
        min_variance=float(np.min(v)),
        max_variance=float(np.max(v)),
        max_weight_fraction=max_weight_fraction,
        cross_check=cross_check,
        warnings=tuple(warnings),
    )


def trim_and_fill_sensitivity(
    effects: np.ndarray,
    standard_errors: np.ndarray,
    *,
    method: str = "FE",
    side: str = "auto",
    max_fill: int | None = None,
) -> TrimAndFillSensitivity:
    """Run a bounded mirror-imputation trim-and-fill sensitivity analysis.

    This implements a conservative sensitivity screen, not reference-matched
    Duval-Tweedie trim-and-fill parity. It mirrors the most extreme studies on
    the overrepresented side of the fitted center and refits the pairwise model.
    """

    y = np.asarray(effects, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    if y.shape != se.shape:
        raise PairwiseMetaError("effects and standard_errors must have the same length.")
    if y.size < 3:
        raise PairwiseMetaError("trim-and-fill sensitivity requires at least three studies.")
    if not np.all(np.isfinite(y)):
        raise PairwiseMetaError("all effects must be finite.")
    if not np.all(np.isfinite(se)) or np.any(se <= 0.0):
        raise PairwiseMetaError("all standard errors must be finite and positive.")
    if max_fill is not None and max_fill < 0:
        raise PairwiseMetaError("max_fill must be non-negative.")

    v = se * se
    observed = fit_pairwise_meta(y, v, method=method)
    center = observed.estimate
    side_key = side.strip().lower()
    if side_key not in {"auto", "left", "right", "none"}:
        raise PairwiseMetaError("side must be auto, left, right, or none.")

    deviations = y - center
    left_indices = [idx for idx, value in enumerate(deviations) if value < 0.0]
    right_indices = [idx for idx, value in enumerate(deviations) if value > 0.0]
    if side_key == "auto":
        if len(left_indices) == len(right_indices):
            fill_side = "none"
        elif len(left_indices) < len(right_indices):
            fill_side = "left"
        else:
            fill_side = "right"
    else:
        fill_side = side_key

    if fill_side == "none":
        fill_count = 0
        selected: list[int] = []
    else:
        under_count = len(left_indices) if fill_side == "left" else len(right_indices)
        over_indices = right_indices if fill_side == "left" else left_indices
        fill_count = max(len(over_indices) - under_count, 0)
        if max_fill is not None:
            fill_count = min(fill_count, max_fill)
        selected = sorted(
            over_indices,
            key=lambda idx: abs(float(deviations[idx])),
            reverse=True,
        )[:fill_count]

    filled_effects = tuple(float(2.0 * center - y[idx]) for idx in selected)
    filled_variances = tuple(float(v[idx]) for idx in selected)
    if filled_effects:
        augmented_y = np.concatenate([y, np.asarray(filled_effects)])
        augmented_v = np.concatenate([v, np.asarray(filled_variances)])
        adjusted = fit_pairwise_meta(augmented_y, augmented_v, method=method)
    else:
        adjusted = observed

    warnings = [
        "Trim-and-fill sensitivity is mirror-imputation screening and is not reference-matched Duval-Tweedie parity."
    ]
    if fill_count == 0:
        warnings.append("No studies were imputed under the selected side rule.")

    return TrimAndFillSensitivity(
        k_observed=int(y.size),
        k_filled=int(fill_count),
        fill_side=fill_side,
        center=float(center),
        observed_estimate=float(observed.estimate),
        adjusted_estimate=float(adjusted.estimate),
        observed_ci=(float(observed.ci_low), float(observed.ci_high)),
        adjusted_ci=(float(adjusted.ci_low), float(adjusted.ci_high)),
        filled_effects=filled_effects,
        filled_variances=filled_variances,
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
