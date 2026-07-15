"""Experimental transportability diagnostics for aggregate meta-analysis.

This module separates two layers:

1. A random-effects meta-regression transport estimate.
2. A support certificate describing whether the target covariate profile is
   interpolation, extrapolation, or a topological gap.

It is not a certification claim. The target in `validation/reference_targets.toml`
remains planned until independent reference artifacts exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

import numpy as np
import scipy.stats
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


COLLAPSIBLE_SCALES = {"RD", "MD", "SMD"}
RATIO_SCALES = {"RR", "logRR", "HR", "logHR"}
NON_COLLAPSIBLE_SCALES = {"OR", "logOR"}


class TransportabilityError(ValueError):
    """Raised when a transportability analysis is malformed or unsupported."""


@dataclass(frozen=True)
class MetaRegressionFit:
    beta: np.ndarray
    cov_beta: np.ndarray
    tau2: float
    design: np.ndarray
    modifiers: np.ndarray
    y: np.ndarray
    variances: np.ndarray
    effect_scale: str
    converged: bool
    n_iter: int

    def conditional_effect(self, x: np.ndarray) -> float:
        """Predicted conditional effect at an effect-modifier profile."""

        xf = _target_design_row(x, self.modifiers.shape[1])
        return float(xf @ self.beta)


@dataclass(frozen=True)
class TransportEstimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    target_x: np.ndarray
    effect_scale: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Hole:
    birth: float
    death: float
    persistence: float
    significant: bool


@dataclass(frozen=True)
class SupportCertificate:
    grade: str
    d_nn: float
    spacing: float
    connect_scale: float
    in_hull: bool
    topological_gap: bool
    n_subpopulations: int
    holes: tuple[Hole, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.grade in {"GOLD", "SILVER"}


@dataclass(frozen=True)
class PersistenceDiagram:
    h0: np.ndarray
    h1: np.ndarray

    def finite(self, dim: int) -> np.ndarray:
        diagram = self.h0 if dim == 0 else self.h1
        if diagram.size == 0:
            return diagram.reshape(0, 2)
        return diagram[np.isfinite(diagram[:, 1])]

    def max_persistence(self, dim: int) -> float:
        finite = self.finite(dim)
        if finite.size == 0:
            return 0.0
        return float(np.max(finite[:, 1] - finite[:, 0]))

    def n_components_at(self, scale: float) -> int:
        born = self.h0[:, 0] <= scale
        alive = ~(self.h0[:, 1] <= scale)
        return int(np.sum(born & alive))


def fit_meta_regression(
    y: np.ndarray,
    variances: np.ndarray,
    modifiers: np.ndarray | None = None,
    *,
    effect_scale: str = "RD",
    max_iter: int = 100,
    tol: float = 1e-7,
) -> MetaRegressionFit:
    """Fit REML random-effects meta-regression on study-level modifiers."""

    y_arr, v_arr, mod_arr = _validate_meta_regression_inputs(y, variances, modifiers)
    design = np.column_stack([np.ones(len(y_arr)), mod_arr]) if mod_arr.size else np.ones((len(y_arr), 1))
    tau2, converged, n_iter = _reml_tau2(y_arr, design, v_arr, max_iter=max_iter, tol=tol)
    weights = np.diag(1.0 / (v_arr + tau2))
    precision = design.T @ weights @ design
    cov_beta = np.linalg.pinv(precision)
    beta = cov_beta @ design.T @ weights @ y_arr
    return MetaRegressionFit(
        beta=beta,
        cov_beta=cov_beta,
        tau2=float(tau2),
        design=design,
        modifiers=mod_arr,
        y=y_arr,
        variances=v_arr,
        effect_scale=effect_scale,
        converged=converged,
        n_iter=n_iter,
    )


def transport_effect(
    fit: MetaRegressionFit,
    target_x: np.ndarray,
    *,
    target_x_cov: np.ndarray | None = None,
    level: float = 0.95,
    baseline_risk_idx: int | None = None,
) -> TransportEstimate:
    """Transport a fitted meta-regression to a target modifier profile."""

    if not (0.0 < level < 1.0):
        raise TransportabilityError("level must be in (0, 1).")
    warnings: list[str] = []
    target = _validate_target_x(target_x, fit.modifiers.shape[1])
    _check_collapsibility(fit.effect_scale, fit.modifiers, baseline_risk_idx, warnings)

    xf = _target_design_row(target, fit.modifiers.shape[1])
    estimate = float(xf @ fit.beta)
    variance = float(xf @ fit.cov_beta @ xf)
    if target_x_cov is not None:
        target_cov = np.asarray(target_x_cov, dtype=float)
        if target_cov.shape != (fit.modifiers.shape[1], fit.modifiers.shape[1]):
            raise TransportabilityError("target_x_cov shape does not match modifiers.")
        beta_mods = fit.beta[1:]
        variance += float(beta_mods @ target_cov @ beta_mods)
    se = math.sqrt(max(variance, 0.0))
    z = scipy.stats.norm.ppf(0.5 + level / 2.0)
    return TransportEstimate(
        estimate=estimate,
        se=se,
        ci_low=estimate - z * se,
        ci_high=estimate + z * se,
        target_x=target,
        effect_scale=fit.effect_scale,
        warnings=tuple(warnings),
    )


def certify_support(
    modifiers: np.ndarray,
    target_x: np.ndarray,
    *,
    n_boot: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
) -> SupportCertificate:
    """Certify target support in effect-modifier space."""

    cloud = _validate_modifier_cloud(modifiers)
    target = _validate_target_x(target_x, cloud.shape[1])
    spacing = _nearest_neighbour_spacing(cloud)
    connect_scale = _connect_scale(cloud)
    d_nn = float(cdist(cloud, target.reshape(1, -1)).min())
    in_hull = is_in_convex_hull(cloud, target)
    holes, _ = significant_holes(cloud, n_boot=n_boot, alpha=alpha, seed=seed)
    confirmed = [hole for hole in holes if hole.significant]
    n_subpopulations = count_subpopulations(cloud)

    eps = max(spacing, 1e-9)
    topological_gap = bool(in_hull and d_nn > connect_scale and confirmed)
    if d_nn <= eps:
        grade = "GOLD"
        reason = "target lies among studies"
    elif topological_gap:
        grade = "GAP"
        reason = "target is inside the convex hull but lies in a confirmed topological hole"
    elif in_hull and d_nn <= connect_scale:
        grade = "SILVER"
        reason = "target is inside the support body"
    elif (not in_hull) and d_nn <= connect_scale:
        grade = "BRONZE"
        reason = "target is just outside the evidence periphery"
    else:
        grade = "NONE"
        reason = "target is beyond the evidence connection scale"

    return SupportCertificate(
        grade=grade,
        d_nn=d_nn,
        spacing=spacing,
        connect_scale=connect_scale,
        in_hull=in_hull,
        topological_gap=topological_gap,
        n_subpopulations=n_subpopulations,
        holes=tuple(holes),
        reason=reason,
    )


def supported_transport(
    y: np.ndarray,
    variances: np.ndarray,
    modifiers: np.ndarray,
    target_x: np.ndarray,
    *,
    effect_scale: str = "RD",
    target_x_cov: np.ndarray | None = None,
    baseline_risk_idx: int | None = None,
    n_boot: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[TransportEstimate, SupportCertificate, MetaRegressionFit]:
    """Fit, transport, and certify support in one explicit pipeline."""

    fit = fit_meta_regression(y, variances, modifiers, effect_scale=effect_scale)
    estimate = transport_effect(
        fit,
        target_x,
        target_x_cov=target_x_cov,
        baseline_risk_idx=baseline_risk_idx,
    )
    certificate = certify_support(modifiers, target_x, n_boot=n_boot, alpha=alpha, seed=seed)
    return estimate, certificate, fit


def compute_persistence(points: np.ndarray, *, max_dim: int = 1, max_edge: float | None = None) -> PersistenceDiagram:
    """Compute small-cloud Vietoris-Rips H0/H1 persistence over GF(2)."""

    cloud = _validate_modifier_cloud(points)
    n = len(cloud)
    distances = _pairwise_distances(cloud)
    if max_edge is None:
        max_edge = float(distances.max()) if n > 1 else 0.0

    simplices: list[tuple[float, int, tuple[int, ...]]] = []
    for idx in range(n):
        simplices.append((0.0, 0, (idx,)))
    for i, j in itertools.combinations(range(n), 2):
        weight = float(distances[i, j])
        if weight <= max_edge:
            simplices.append((weight, 1, (i, j)))
    if max_dim >= 1:
        for i, j, k in itertools.combinations(range(n), 3):
            weight = max(float(distances[i, j]), float(distances[i, k]), float(distances[j, k]))
            if weight <= max_edge:
                simplices.append((weight, 2, (i, j, k)))

    simplices.sort(key=lambda item: (item[0], item[1], item[2]))
    simplex_index = {(dim, vertices): idx for idx, (_, dim, vertices) in enumerate(simplices)}
    filtrations = np.asarray([item[0] for item in simplices], dtype=float)
    dimensions = np.asarray([item[1] for item in simplices], dtype=int)

    columns: list[list[int]] = []
    for _, dim, vertices in simplices:
        if dim == 0:
            columns.append([])
        elif dim == 1:
            a, b = vertices
            columns.append(sorted((simplex_index[(0, (a,))], simplex_index[(0, (b,))])))
        else:
            a, b, c = vertices
            columns.append(
                sorted(
                    (
                        simplex_index[(1, (a, b))],
                        simplex_index[(1, (a, c))],
                        simplex_index[(1, (b, c))],
                    )
                )
            )

    pivot: dict[int, int] = {}
    low_of: list[int | None] = [None] * len(simplices)
    for col_idx in range(len(simplices)):
        column = columns[col_idx]
        low = _low(column)
        while low is not None and low in pivot:
            column = _symmetric_difference(column, columns[pivot[low]])
            low = _low(column)
        columns[col_idx] = column
        if low is not None:
            pivot[low] = col_idx
            low_of[col_idx] = low

    h0: list[tuple[float, float]] = []
    h1: list[tuple[float, float]] = []
    paired_births = set(pivot.keys())
    for col_idx, low in enumerate(low_of):
        if low is None:
            continue
        birth = filtrations[low]
        death = filtrations[col_idx]
        if death <= birth:
            continue
        if dimensions[low] == 0:
            h0.append((float(birth), float(death)))
        elif dimensions[low] == 1:
            h1.append((float(birth), float(death)))

    for idx, column in enumerate(columns):
        if column or idx in paired_births:
            continue
        if dimensions[idx] == 0:
            h0.append((float(filtrations[idx]), math.inf))
        elif dimensions[idx] == 1:
            h1.append((float(filtrations[idx]), math.inf))

    h0_array = np.asarray(sorted(h0), dtype=float).reshape(-1, 2) if h0 else np.zeros((0, 2))
    h1_array = np.asarray(sorted(h1), dtype=float).reshape(-1, 2) if h1 else np.zeros((0, 2))
    return PersistenceDiagram(h0=h0_array, h1=h1_array)


def bottleneck_distance(diagram_a: np.ndarray, diagram_b: np.ndarray) -> float:
    """Bottleneck distance between finite persistence diagrams."""

    a = np.asarray(diagram_a, dtype=float).reshape(-1, 2)
    b = np.asarray(diagram_b, dtype=float).reshape(-1, 2)
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    if len(a) == 0 and len(b) == 0:
        return 0.0

    def linf(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return float(max(abs(point_a[0] - point_b[0]), abs(point_a[1] - point_b[1])))

    def diag(point: np.ndarray) -> float:
        return float(abs(point[1] - point[0]) / 2.0)

    candidates = {0.0}
    for point in a:
        candidates.add(diag(point))
    for point in b:
        candidates.add(diag(point))
    for point_a in a:
        for point_b in b:
            candidates.add(linf(point_a, point_b))
    candidate_list = sorted(candidates)

    n_a = len(a)
    n_b = len(b)
    size = n_a + n_b

    def feasible(threshold: float) -> bool:
        if size == 0:
            return True
        rows: list[int] = []
        cols: list[int] = []
        for i in range(n_a):
            for j in range(n_b):
                if linf(a[i], b[j]) <= threshold:
                    rows.append(i)
                    cols.append(j)
            if diag(a[i]) <= threshold:
                rows.append(i)
                cols.append(n_b + i)
        for j in range(n_b):
            if diag(b[j]) <= threshold:
                rows.append(n_a + j)
                cols.append(j)
        for j in range(n_b):
            for i in range(n_a):
                rows.append(n_a + j)
                cols.append(n_b + i)
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
        match = maximum_bipartite_matching(graph, perm_type="column")
        return bool(np.all(match != -1))

    lo = 0
    hi = len(candidate_list) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(candidate_list[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(candidate_list[lo])


def significant_holes(
    modifiers: np.ndarray,
    *,
    n_boot: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[tuple[Hole, ...], float]:
    """Return H1 holes significant under a bootstrap bottleneck band."""

    if n_boot < 0:
        raise TransportabilityError("n_boot must be non-negative.")
    cloud = _validate_modifier_cloud(modifiers)
    observed = compute_persistence(cloud, max_dim=1).finite(1)
    if len(cloud) < 4 or observed.size == 0:
        return (), 0.0
    rng = np.random.default_rng(seed)
    distances: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cloud), size=len(cloud))
        boot = compute_persistence(cloud[idx], max_dim=1).finite(1)
        distances.append(bottleneck_distance(observed, boot))
    c_alpha = float(np.quantile(distances, 1.0 - alpha)) if distances else 0.0
    holes = []
    for birth, death in observed:
        persistence = float(death - birth)
        holes.append(Hole(float(birth), float(death), persistence, bool(persistence > 2.0 * c_alpha)))
    holes.sort(key=lambda hole: -hole.persistence)
    return tuple(holes), c_alpha


def count_subpopulations(modifiers: np.ndarray, *, gap_factor: float = 4.0) -> int:
    """Count latent subpopulations using long MST edges as H0 gaps."""

    cloud = _validate_modifier_cloud(modifiers)
    if len(cloud) < 2:
        return 1
    distances = cdist(cloud, cloud)
    edges = minimum_spanning_tree(distances).toarray()
    edge_lengths = edges[edges > 0]
    if edge_lengths.size < 2:
        return 1
    median_edge = float(np.median(edge_lengths))
    if median_edge <= 0.0:
        return 1
    return int(1 + np.sum(edge_lengths > gap_factor * median_edge))


def is_in_convex_hull(modifiers: np.ndarray, target_x: np.ndarray) -> bool:
    """Return whether target_x is inside the convex hull of the modifier cloud."""

    cloud = _validate_modifier_cloud(modifiers)
    target = _validate_target_x(target_x, cloud.shape[1])
    if cloud.shape[1] == 1:
        return bool(float(cloud.min()) <= target[0] <= float(cloud.max()))
    try:
        return bool(Delaunay(cloud).find_simplex(target) >= 0)
    except Exception:
        n_rows = len(cloud)
        a_eq = np.vstack([cloud.T, np.ones(n_rows)])
        b_eq = np.concatenate([target, [1.0]])
        res = linprog(np.zeros(n_rows), A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * n_rows)
        return bool(res.success)


def _validate_meta_regression_inputs(
    y: np.ndarray,
    variances: np.ndarray,
    modifiers: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    v_arr = np.asarray(variances, dtype=float).reshape(-1)
    if y_arr.size < 2:
        raise TransportabilityError("at least two studies are required.")
    if y_arr.shape != v_arr.shape:
        raise TransportabilityError("y and variances must have the same length.")
    if not np.all(np.isfinite(y_arr)):
        raise TransportabilityError("all effects must be finite.")
    if not np.all(np.isfinite(v_arr)) or np.any(v_arr <= 0.0):
        raise TransportabilityError("all variances must be finite and positive.")
    if modifiers is None:
        mod_arr = np.zeros((len(y_arr), 0), dtype=float)
    else:
        mod_arr = np.asarray(modifiers, dtype=float)
        if mod_arr.ndim == 1:
            mod_arr = mod_arr.reshape(-1, 1)
        if mod_arr.shape[0] != len(y_arr):
            raise TransportabilityError("modifiers must have one row per study.")
        if not np.all(np.isfinite(mod_arr)):
            raise TransportabilityError("all modifiers must be finite.")
    if len(y_arr) <= mod_arr.shape[1] + 1:
        raise TransportabilityError("not enough studies for the requested meta-regression.")
    return y_arr, v_arr, mod_arr


def _validate_modifier_cloud(modifiers: np.ndarray) -> np.ndarray:
    cloud = np.asarray(modifiers, dtype=float)
    if cloud.ndim == 1:
        cloud = cloud.reshape(-1, 1)
    if cloud.ndim != 2 or cloud.shape[0] < 1 or cloud.shape[1] < 1:
        raise TransportabilityError("modifier cloud must be a non-empty 2D array.")
    if not np.all(np.isfinite(cloud)):
        raise TransportabilityError("modifier cloud contains non-finite values.")
    return cloud


def _validate_target_x(target_x: np.ndarray, n_modifiers: int) -> np.ndarray:
    target = np.asarray(target_x, dtype=float).reshape(-1)
    if target.shape != (n_modifiers,):
        raise TransportabilityError(
            f"target_x must have length {n_modifiers}; got {target.shape[0]}."
        )
    if not np.all(np.isfinite(target)):
        raise TransportabilityError("target_x contains non-finite values.")
    return target


def _reml_tau2(
    y: np.ndarray,
    design: np.ndarray,
    variances: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> tuple[float, bool, int]:
    tau2 = max(float(np.var(y, ddof=1) - np.mean(variances)), 0.0)
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        weights = np.diag(1.0 / (variances + tau2))
        xt_w = design.T @ weights
        xt_w_x_inv = np.linalg.pinv(xt_w @ design)
        projection = weights - weights @ design @ xt_w_x_inv @ xt_w
        py = projection @ y
        score = -0.5 * np.trace(projection) + 0.5 * float(py @ py)
        information = 0.5 * float(np.trace(projection @ projection))
        if information <= 0.0:
            break
        candidate = max(tau2 + score / information, 0.0)
        if abs(candidate - tau2) < tol:
            tau2 = candidate
            converged = True
            break
        tau2 = candidate
    return float(tau2), converged, iteration


def _check_collapsibility(
    effect_scale: str,
    modifiers: np.ndarray,
    baseline_risk_idx: int | None,
    warnings: list[str],
) -> None:
    if effect_scale in NON_COLLAPSIBLE_SCALES:
        raise TransportabilityError(
            f"Refusing to transport on non-collapsible scale '{effect_scale}'. "
            "Use a collapsible estimand such as RD, MD, SMD, or a prespecified transformation."
        )
    if effect_scale in RATIO_SCALES and baseline_risk_idx is not None and modifiers.size:
        if not (0 <= baseline_risk_idx < modifiers.shape[1]):
            raise TransportabilityError("baseline_risk_idx is out of range.")
        baseline = modifiers[:, baseline_risk_idx]
        baseline_sd = float(np.std(baseline))
        if baseline_sd > 0.1:
            warnings.append(
                f"Baseline-risk SD across studies is {baseline_sd:.3f} on ratio scale "
                f"{effect_scale}; rerun on RD scale as a sensitivity analysis."
            )


def _target_design_row(target_x: np.ndarray, n_modifiers: int) -> np.ndarray:
    target = _validate_target_x(target_x, n_modifiers)
    return np.concatenate([[1.0], target])


def _nearest_neighbour_spacing(cloud: np.ndarray) -> float:
    if len(cloud) < 2:
        return 0.0
    distances = cdist(cloud, cloud)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    positive = nearest[nearest > 0.0]
    return float(np.median(positive)) if positive.size else 0.0


def _connect_scale(cloud: np.ndarray) -> float:
    if len(cloud) < 2:
        return 0.0
    return float(minimum_spanning_tree(cdist(cloud, cloud)).toarray().max())


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))


def _low(column: list[int]) -> int | None:
    return column[-1] if column else None


def _symmetric_difference(left: list[int], right: list[int]) -> list[int]:
    out: list[int] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        elif left[i] < right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
