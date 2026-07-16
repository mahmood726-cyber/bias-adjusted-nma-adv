"""Experimental pairwise meta-analysis bridge.

The purpose of this module is to make pairwise conventions explicit before
larger NMA validation work depends on them. It provides deterministic fixed-
and random-effects pooling, heterogeneity estimators, HKSJ scaling, and
prediction intervals. External `metafor`/`meta` parity remains a planned
validation target until independent artifacts are added.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
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
class GOSHSubsetDiagnostic:
    """One subset refit in an exhaustive GOSH-style diagnostic."""

    subset_indices: tuple[int, ...]
    k_subset: int
    estimate: float
    se: float
    tau2: float
    q: float
    delta_estimate: float
    absolute_delta_estimate: float


@dataclass(frozen=True)
class PairwiseGOSHDiagnostic:
    """Exhaustive bounded GOSH-style outlier-space diagnostic."""

    method: str
    full_estimate: float
    full_tau2: float
    min_subset_size: int
    n_subsets: int
    diagnostics: tuple[GOSHSubsetDiagnostic, ...]
    max_abs_delta_estimate: float
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
class REMLProfilePoint:
    """One point on the pairwise REML tau2 objective profile."""

    tau2: float
    objective: float


@dataclass(frozen=True)
class REMLLocalMinimumDiagnostic:
    """Grid/profile diagnostic for pairwise REML tau2 optimization."""

    k: int
    optimizer_tau2: float
    optimizer_objective: float
    best_grid_tau2: float
    best_grid_objective: float
    objective_gap: float
    local_minima: tuple[REMLProfilePoint, ...]
    boundary_minimum: bool
    profile: tuple[REMLProfilePoint, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseStressMatrixScenario:
    """One named scenario in a pairwise numerical stress matrix."""

    scenario_id: str
    status: str
    stress_report: PairwiseNumericalStressReport | None
    reml_profile: REMLLocalMinimumDiagnostic | None
    warnings: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class PairwiseNumericalStressMatrix:
    """Deterministic stress matrix over named pairwise datasets."""

    n_scenarios: int
    status_counts: dict[str, int]
    scenarios: tuple[PairwiseStressMatrixScenario, ...]
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


def gosh_outlier_space_diagnostic(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    method: str = "REML",
    min_subset_size: int = 3,
    max_subsets: int = 4096,
) -> PairwiseGOSHDiagnostic:
    """Run exhaustive subset refits for bounded GOSH-style diagnostics.

    This is a deterministic subset refitting diagnostic for small to moderate
    pairwise meta-analyses. It fails closed when the requested subset space is
    larger than ``max_subsets``.
    """

    y, v = _validate_effects(effects, variances)
    if min_subset_size < 2:
        raise PairwiseMetaError("min_subset_size must be at least 2.")
    if min_subset_size > y.size:
        raise PairwiseMetaError("min_subset_size cannot exceed the number of studies.")
    if max_subsets <= 0:
        raise PairwiseMetaError("max_subsets must be positive.")

    subset_count = sum(
        math.comb(int(y.size), size) for size in range(min_subset_size, int(y.size) + 1)
    )
    if subset_count > max_subsets:
        raise PairwiseMetaError(
            f"GOSH subset space has {subset_count} refits, exceeding max_subsets={max_subsets}."
        )

    full = fit_pairwise_meta(y, v, method=method)
    diagnostics: list[GOSHSubsetDiagnostic] = []
    for size in range(min_subset_size, int(y.size) + 1):
        for subset in itertools.combinations(range(int(y.size)), size):
            subset_index = np.asarray(subset, dtype=int)
            refit = fit_pairwise_meta(y[subset_index], v[subset_index], method=method)
            delta = float(refit.estimate - full.estimate)
            diagnostics.append(
                GOSHSubsetDiagnostic(
                    subset_indices=tuple(int(item) for item in subset),
                    k_subset=int(size),
                    estimate=float(refit.estimate),
                    se=float(refit.se),
                    tau2=float(refit.tau2),
                    q=float(refit.q),
                    delta_estimate=delta,
                    absolute_delta_estimate=abs(delta),
                )
            )

    max_abs_delta = (
        max(item.absolute_delta_estimate for item in diagnostics) if diagnostics else 0.0
    )
    warnings = (
        "GOSH-style subset diagnostics are exploratory screens and require reference matching before tier-one parity claims.",
    )
    return PairwiseGOSHDiagnostic(
        method=full.method,
        full_estimate=float(full.estimate),
        full_tau2=float(full.tau2),
        min_subset_size=int(min_subset_size),
        n_subsets=int(subset_count),
        diagnostics=tuple(diagnostics),
        max_abs_delta_estimate=float(max_abs_delta),
        warnings=warnings,
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


def reml_local_minimum_diagnostic(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    n_grid: int = 51,
    tau2_upper: float | None = None,
    objective_tolerance: float = 1e-6,
) -> REMLLocalMinimumDiagnostic:
    """Profile the REML tau2 objective to expose boundary/local-minimum risk."""

    y, v = _validate_effects(effects, variances)
    if y.size < 2:
        raise PairwiseMetaError("REML local-minimum diagnostics require at least two studies.")
    if n_grid < 5:
        raise PairwiseMetaError("n_grid must be at least 5.")
    if tau2_upper is not None and tau2_upper <= 0.0:
        raise PairwiseMetaError("tau2_upper must be positive when supplied.")
    if objective_tolerance < 0.0:
        raise PairwiseMetaError("objective_tolerance must be nonnegative.")

    upper = float(tau2_upper) if tau2_upper is not None else _upper_tau2_for_minimization(y, v)
    grid = np.linspace(0.0, upper, int(n_grid), dtype=float)
    profile = tuple(
        REMLProfilePoint(
            tau2=float(tau2),
            objective=float(_restricted_log_likelihood(y, v, float(tau2))),
        )
        for tau2 in grid
    )
    objectives = np.asarray([point.objective for point in profile], dtype=float)
    best_idx = int(np.argmin(objectives))

    optimizer_tau2 = _tau2_reml(y, v)
    optimizer_objective = float(_restricted_log_likelihood(y, v, optimizer_tau2))
    best_grid = profile[best_idx]
    local_minima = _local_minima_from_profile(profile, tolerance=objective_tolerance)
    boundary_minimum = best_idx in {0, len(profile) - 1}
    objective_gap = float(optimizer_objective - best_grid.objective)

    warnings: list[str] = [
        "REML profile diagnostics are numerical screens and are not reference-package optimizer parity."
    ]
    near_best = tuple(
        point
        for point in local_minima
        if point.objective <= best_grid.objective + objective_tolerance
    )
    if len(near_best) > 1:
        warnings.append("Multiple near-best local minima detected on the REML grid.")
    if boundary_minimum:
        warnings.append("Best REML grid point is on the tau2 boundary.")
    if abs(objective_gap) > max(objective_tolerance, 1e-10):
        warnings.append("Bounded optimizer and grid minimum differ beyond tolerance.")

    return REMLLocalMinimumDiagnostic(
        k=int(y.size),
        optimizer_tau2=float(optimizer_tau2),
        optimizer_objective=optimizer_objective,
        best_grid_tau2=float(best_grid.tau2),
        best_grid_objective=float(best_grid.objective),
        objective_gap=objective_gap,
        local_minima=local_minima,
        boundary_minimum=bool(boundary_minimum),
        profile=profile,
        warnings=tuple(warnings),
    )


def pairwise_numerical_stress_matrix(
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    methods: tuple[str, ...] = ("FE", "DL", "PM", "REML"),
    dominant_weight_threshold: float = 0.80,
    n_grid: int = 51,
) -> PairwiseNumericalStressMatrix:
    """Run pairwise stress and REML-profile diagnostics across scenarios."""

    if not scenarios:
        raise PairwiseMetaError("at least one stress scenario is required.")

    scenario_reports: list[PairwiseStressMatrixScenario] = []
    status_counts: dict[str, int] = {}
    for scenario_id, payload in scenarios.items():
        try:
            effects, variances = payload
            stress = numerical_stress_report(
                effects,
                variances,
                methods=methods,
                dominant_weight_threshold=dominant_weight_threshold,
            )
            profile = reml_local_minimum_diagnostic(
                effects,
                variances,
                n_grid=n_grid,
            )
            warnings = tuple(dict.fromkeys((*stress.warnings, *profile.warnings)))
            status = "warning" if warnings else "passed"
            report = PairwiseStressMatrixScenario(
                scenario_id=str(scenario_id),
                status=status,
                stress_report=stress,
                reml_profile=profile,
                warnings=warnings,
                error=None,
            )
        except (PairwiseMetaError, ValueError) as exc:
            report = PairwiseStressMatrixScenario(
                scenario_id=str(scenario_id),
                status="failed",
                stress_report=None,
                reml_profile=None,
                warnings=(str(exc),),
                error=str(exc),
            )
        scenario_reports.append(report)
        status_counts[report.status] = status_counts.get(report.status, 0) + 1

    warnings = (
        "Pairwise stress matrices are local numerical screens and require reference-package comparisons before robustness claims.",
    )
    return PairwiseNumericalStressMatrix(
        n_scenarios=len(scenario_reports),
        status_counts=dict(sorted(status_counts.items())),
        scenarios=tuple(scenario_reports),
        warnings=warnings,
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


def _local_minima_from_profile(
    profile: tuple[REMLProfilePoint, ...],
    *,
    tolerance: float,
) -> tuple[REMLProfilePoint, ...]:
    local_minima: list[REMLProfilePoint] = []
    for idx, point in enumerate(profile):
        left = profile[idx - 1].objective if idx > 0 else math.inf
        right = profile[idx + 1].objective if idx < len(profile) - 1 else math.inf
        if point.objective <= left + tolerance and point.objective <= right + tolerance:
            local_minima.append(point)
    return tuple(local_minima)


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
