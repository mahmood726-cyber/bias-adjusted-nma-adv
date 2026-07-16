"""Experimental contrast-level NMA with multi-arm covariance preservation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContrastRow:
    """One contrast-level study estimate.

    The estimate is oriented as treatment `t2` minus treatment `t1` on the
    analysis scale, matching netmeta-style contrast input after orientation.
    """

    study: str
    t1: str
    t2: str
    est: float
    se: float

    def __post_init__(self) -> None:
        if not self.study:
            raise ValueError("study must not be empty.")
        if not self.t1 or not self.t2:
            raise ValueError("treatments must not be empty.")
        if self.t1 == self.t2:
            raise ValueError("contrast treatments must differ.")
        if not math.isfinite(self.est):
            raise ValueError("est must be finite.")
        if not math.isfinite(self.se) or self.se <= 0.0:
            raise ValueError("se must be finite and positive.")


@dataclass(frozen=True)
class _StudyContrast:
    to: str
    from_: str
    est: float
    se: float | None = None


@dataclass(frozen=True)
class _Study:
    study_id: str
    arms: tuple[str, ...]
    contrasts: tuple[_StudyContrast, ...]
    arm_variances: dict[str, float] | None
    base: str | None
    multi_arm: bool


@dataclass(frozen=True)
class _AssembledNetwork:
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray
    nonreference_treatments: tuple[str, ...]
    row_studies: tuple[str, ...]
    row_contrasts: tuple[tuple[str, str], ...]
    blocks: tuple[tuple[int, int, bool], ...]


@dataclass(frozen=True)
class StudyDesignDiagnostic:
    """Pre-fit design check for one study's contrast rows."""

    study: str
    n_arms: int
    n_contrasts: int
    expected_contrasts: int
    multi_arm: bool
    complete_pairwise_clique: bool


@dataclass(frozen=True)
class MultiArmDesignDiagnostic:
    """Pre-fit network design check for contrast-level multi-arm NMA."""

    reference_treatment: str
    treatments: tuple[str, ...]
    n_studies: int
    n_contrast_rows: int
    n_parameters: int
    design_rank: int
    connected: bool
    disconnected_treatments: tuple[str, ...]
    estimable: bool
    study_diagnostics: tuple[StudyDesignDiagnostic, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MultiArmFitReport:
    """Deterministic status report for a multi-arm GLS fit attempt."""

    status: str
    stage: str
    message: str
    model: str
    reference_treatment: str
    n_studies: int
    n_contrast_rows: int
    n_parameters: int
    design_rank: int
    connected: bool
    estimable: bool
    disconnected_treatments: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ContrastInfluenceDiagnostic:
    """Diagnostic for one contrast row in a fitted GLS network model."""

    row_index: int
    study: str
    treatment_from: str
    treatment_to: str
    observed: float
    fitted: float
    residual: float
    variance: float
    leverage: float
    standardized_residual: float
    cook_distance: float


@dataclass(frozen=True)
class ContrastContributionDiagnostic:
    """Absolute GLS mapping contribution for one row to one fitted parameter.

    The signed mapping coefficient is retained for auditability, but the
    contribution proportion is normalized from absolute coefficients so callers
    do not interpret signed GLS algebra as a percentage evidence flow.
    """

    target_treatment: str
    row_index: int
    study: str
    treatment_from: str
    treatment_to: str
    mapping_weight: float
    absolute_mapping_weight: float
    contribution: float


@dataclass(frozen=True)
class MultiArmNMAFit:
    """Result from experimental multi-arm contrast-level GLS NMA."""

    reference_treatment: str
    treatments: tuple[str, ...]
    nonreference_treatments: tuple[str, ...]
    model: str
    estimates: np.ndarray
    covariance: np.ndarray
    tau2: float
    q: float | None
    df: int | None
    multi_arm_studies: tuple[str, ...]
    warnings: tuple[str, ...]
    influence_diagnostics: tuple[ContrastInfluenceDiagnostic, ...]
    contribution_diagnostics: tuple[ContrastContributionDiagnostic, ...]

    def effect_vs_reference(self, treatment: str) -> tuple[float, float]:
        """Return estimate and SE for treatment versus the reference."""

        if treatment == self.reference_treatment:
            return 0.0, 0.0
        try:
            idx = self.nonreference_treatments.index(treatment)
        except ValueError as exc:
            raise ValueError(f"Unknown treatment '{treatment}'.") from exc
        est = float(self.estimates[idx])
        se = float(math.sqrt(max(self.covariance[idx, idx], 0.0)))
        return est, se

    def contrast(self, treatment_a: str, treatment_b: str) -> tuple[float, float]:
        """Return estimate and SE for treatment_a minus treatment_b."""

        beta_a, var_a, cov_ab = self._beta_var_cov(treatment_a, treatment_b)
        beta_b, var_b, _ = self._beta_var_cov(treatment_b, treatment_a)
        est = beta_a - beta_b
        var = max(var_a + var_b - 2.0 * cov_ab, 0.0)
        return float(est), float(math.sqrt(var))

    def _beta_var_cov(self, treatment: str, other: str) -> tuple[float, float, float]:
        if treatment == self.reference_treatment:
            beta = 0.0
            var = 0.0
            row = None
        else:
            try:
                row = self.nonreference_treatments.index(treatment)
            except ValueError as exc:
                raise ValueError(f"Unknown treatment '{treatment}'.") from exc
            beta = float(self.estimates[row])
            var = float(self.covariance[row, row])

        if treatment == self.reference_treatment or other == self.reference_treatment:
            cov = 0.0
        else:
            try:
                col = self.nonreference_treatments.index(other)
            except ValueError as exc:
                raise ValueError(f"Unknown treatment '{other}'.") from exc
            cov = float(self.covariance[row, col]) if row is not None else 0.0
        return beta, var, cov


def fit_multiarm_gls(
    rows: Iterable[ContrastRow | dict[str, object]],
    *,
    reference_treatment: str | None = None,
    model: str = "fixed",
) -> MultiArmNMAFit:
    """Fit fixed- or random-effects contrast-level NMA preserving multi-arm covariance.

    This is an experimental reference-matching module. It requires full pairwise
    contrast cliques for multi-arm studies so arm-level covariance can be
    recovered; incomplete multi-arm studies are dropped with a warning.
    """

    parsed_rows = tuple(_coerce_row(row) for row in rows)
    if not parsed_rows:
        raise ValueError("no contrasts supplied.")

    studies, warnings = _build_studies(parsed_rows)
    if not studies:
        raise ValueError("; ".join(warnings) if warnings else "no valid studies supplied.")

    treatments = tuple(sorted({row.t1 for row in parsed_rows} | {row.t2 for row in parsed_rows}))
    reference = reference_treatment or treatments[0]
    if reference not in treatments:
        raise ValueError(f"reference_treatment '{reference}' not present in network.")
    _validate_connected(parsed_rows, reference)

    assembled = _assemble(studies, treatments, reference)
    if assembled.x.shape[0] < assembled.x.shape[1]:
        raise ValueError(
            f"insufficient contrasts (N={assembled.x.shape[0]} < K={assembled.x.shape[1]})."
        )

    fixed = _gls(assembled.x, assembled.y, assembled.v)
    if fixed is None:
        raise ValueError("singular design or disconnected treatment network.")

    model_key = model.lower()
    diagnostic_v = assembled.v
    if model_key in {"fe", "fixed", "common"}:
        fit = fixed
        tau2 = 0.0
        q_info = None
        result_model = "fixed"
    elif model_key in {"re", "random"}:
        q_info = _generalized_dl(assembled.x, assembled.y, assembled.v, fixed, assembled.blocks)
        tau2 = q_info["tau2"]
        fit = fixed
        if tau2 > 0.0:
            random_v = _random_effects_covariance(assembled.v, assembled.blocks, tau2)
            random_fit = _gls(assembled.x, assembled.y, random_v)
            if random_fit is not None:
                fit = random_fit
                diagnostic_v = random_v
        result_model = "random"
    else:
        raise ValueError("model must be fixed/common or random/re.")

    multi_arm = tuple(study.study_id for study in studies if study.multi_arm)
    influence_diagnostics = _influence_diagnostics(assembled, fit, diagnostic_v)
    contribution_diagnostics = _contribution_diagnostics(assembled, fit)
    return MultiArmNMAFit(
        reference_treatment=reference,
        treatments=treatments,
        nonreference_treatments=assembled.nonreference_treatments,
        model=result_model,
        estimates=fit["d"],
        covariance=fit["cov"],
        tau2=float(tau2),
        q=float(q_info["q"]) if q_info else None,
        df=int(q_info["df"]) if q_info else None,
        multi_arm_studies=multi_arm,
        warnings=tuple(warnings),
        influence_diagnostics=influence_diagnostics,
        contribution_diagnostics=contribution_diagnostics,
    )


def diagnose_multiarm_design(
    rows: Iterable[ContrastRow | dict[str, object]],
    *,
    reference_treatment: str | None = None,
) -> MultiArmDesignDiagnostic:
    """Run deterministic pre-fit checks for a contrast-level NMA design."""

    parsed_rows = tuple(_coerce_row(row) for row in rows)
    if not parsed_rows:
        raise ValueError("no contrasts supplied.")

    treatments = tuple(sorted({row.t1 for row in parsed_rows} | {row.t2 for row in parsed_rows}))
    reference = reference_treatment or treatments[0]
    if reference not in treatments:
        raise ValueError(f"reference_treatment '{reference}' not present in network.")

    study_diagnostics = _study_design_diagnostics(parsed_rows)
    disconnected_treatments = _disconnected_treatments(parsed_rows, reference)
    connected = not disconnected_treatments
    warnings: list[str] = []
    studies: tuple[_Study, ...] = ()
    if connected:
        try:
            studies, warnings = _build_studies(parsed_rows)
        except ValueError as exc:
            warnings = [str(exc)]

    n_parameters = max(len(treatments) - 1, 0)
    design_rank = 0
    if connected and studies:
        assembled = _assemble(studies, treatments, reference)
        design_rank = int(np.linalg.matrix_rank(assembled.x))
    estimable = bool(connected and design_rank == n_parameters and len(parsed_rows) >= n_parameters)

    return MultiArmDesignDiagnostic(
        reference_treatment=reference,
        treatments=treatments,
        n_studies=len({row.study for row in parsed_rows}),
        n_contrast_rows=len(parsed_rows),
        n_parameters=n_parameters,
        design_rank=design_rank,
        connected=connected,
        disconnected_treatments=disconnected_treatments,
        estimable=estimable,
        study_diagnostics=study_diagnostics,
        warnings=tuple(warnings),
    )


def report_multiarm_gls_fit(
    rows: Iterable[ContrastRow | dict[str, object]],
    *,
    reference_treatment: str | None = None,
    model: str = "fixed",
) -> MultiArmFitReport:
    """Return a structured pass/fail report for a multi-arm GLS fit attempt."""

    parsed_rows = tuple(_coerce_row(row) for row in rows)
    try:
        design = diagnose_multiarm_design(parsed_rows, reference_treatment=reference_treatment)
    except ValueError as exc:
        return _empty_fit_report(
            status="failed",
            stage="input",
            message=str(exc),
            model=model,
            reference_treatment=reference_treatment or "",
        )

    if not design.estimable:
        message = _design_failure_message(design)
        return _fit_report_from_design(
            design,
            status="failed",
            stage="design",
            message=message,
            model=model,
            warnings=design.warnings,
        )

    try:
        fit = fit_multiarm_gls(parsed_rows, reference_treatment=design.reference_treatment, model=model)
    except ValueError as exc:
        return _fit_report_from_design(
            design,
            status="failed",
            stage="fit",
            message=str(exc),
            model=model,
            warnings=design.warnings,
        )

    return _fit_report_from_design(
        design,
        status="passed",
        stage="fit",
        message="multi-arm GLS fit completed",
        model=fit.model,
        warnings=tuple(design.warnings) + tuple(fit.warnings),
    )


def _coerce_row(row: ContrastRow | dict[str, object]) -> ContrastRow:
    if isinstance(row, ContrastRow):
        return row
    return ContrastRow(
        study=str(row["study"]),
        t1=str(row["t1"]),
        t2=str(row["t2"]),
        est=float(row["est"]),
        se=float(row["se"]),
    )


def _empty_fit_report(
    *,
    status: str,
    stage: str,
    message: str,
    model: str,
    reference_treatment: str,
) -> MultiArmFitReport:
    return MultiArmFitReport(
        status=status,
        stage=stage,
        message=message,
        model=model,
        reference_treatment=reference_treatment,
        n_studies=0,
        n_contrast_rows=0,
        n_parameters=0,
        design_rank=0,
        connected=False,
        estimable=False,
        disconnected_treatments=(),
        warnings=(),
    )


def _fit_report_from_design(
    design: MultiArmDesignDiagnostic,
    *,
    status: str,
    stage: str,
    message: str,
    model: str,
    warnings: tuple[str, ...],
) -> MultiArmFitReport:
    return MultiArmFitReport(
        status=status,
        stage=stage,
        message=message,
        model=model,
        reference_treatment=design.reference_treatment,
        n_studies=design.n_studies,
        n_contrast_rows=design.n_contrast_rows,
        n_parameters=design.n_parameters,
        design_rank=design.design_rank,
        connected=design.connected,
        estimable=design.estimable,
        disconnected_treatments=design.disconnected_treatments,
        warnings=warnings,
    )


def _design_failure_message(design: MultiArmDesignDiagnostic) -> str:
    if design.disconnected_treatments:
        return (
            "design is disconnected from the reference treatment: "
            f"{list(design.disconnected_treatments)}"
        )
    if design.warnings:
        return "; ".join(design.warnings)
    return (
        "design is not estimable: "
        f"rank {design.design_rank} for {design.n_parameters} parameters"
    )


def _study_design_diagnostics(rows: tuple[ContrastRow, ...]) -> tuple[StudyDesignDiagnostic, ...]:
    grouped: dict[str, list[ContrastRow]] = {}
    order: list[str] = []
    for row in rows:
        if row.study not in grouped:
            grouped[row.study] = []
            order.append(row.study)
        grouped[row.study].append(row)

    diagnostics: list[StudyDesignDiagnostic] = []
    for study_id in order:
        study_rows = grouped[study_id]
        arms = {row.t1 for row in study_rows} | {row.t2 for row in study_rows}
        expected = len(arms) * (len(arms) - 1) // 2
        unique_pairs = {tuple(sorted((row.t1, row.t2))) for row in study_rows}
        diagnostics.append(
            StudyDesignDiagnostic(
                study=study_id,
                n_arms=len(arms),
                n_contrasts=len(unique_pairs),
                expected_contrasts=expected,
                multi_arm=len(arms) > 2,
                complete_pairwise_clique=len(unique_pairs) == expected,
            )
        )
    return tuple(diagnostics)


def _build_studies(rows: tuple[ContrastRow, ...]) -> tuple[tuple[_Study, ...], list[str]]:
    grouped: dict[str, list[ContrastRow]] = {}
    order: list[str] = []
    for row in rows:
        if row.study not in grouped:
            grouped[row.study] = []
            order.append(row.study)
        grouped[row.study].append(row)

    studies: list[_Study] = []
    warnings: list[str] = []
    for study_id in order:
        study_rows = grouped[study_id]
        arms = tuple(sorted({row.t1 for row in study_rows} | {row.t2 for row in study_rows}))
        if len(arms) < 2:
            warnings.append(f"study {study_id}: fewer than two arms")
            continue

        if len(arms) == 2:
            row = study_rows[0]
            base = arms[0]
            to = row.t2
            from_ = row.t1
            est = row.est
            if from_ != base:
                to, from_ = from_, to
                est = -est
            studies.append(
                _Study(
                    study_id=study_id,
                    arms=arms,
                    contrasts=(_StudyContrast(to=to, from_=from_, est=est, se=row.se),),
                    arm_variances=None,
                    base=None,
                    multi_arm=False,
                )
            )
            continue

        needed_pairs = len(arms) * (len(arms) - 1) // 2
        pair_rows: dict[tuple[str, str], ContrastRow] = {}
        for row in study_rows:
            key = tuple(sorted((row.t1, row.t2)))
            pair_rows[key] = row
        if len(pair_rows) < needed_pairs:
            warnings.append(
                f"study {study_id}: multi-arm ({len(arms)} arms) but only "
                f"{len(pair_rows)}/{needed_pairs} pairwise contrasts present; "
                "cannot recover arm covariance"
            )
            continue

        se2 = np.zeros((len(arms), len(arms)), dtype=float)
        total = 0.0
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                pair = pair_rows[(arms[i], arms[j])]
                value = pair.se * pair.se
                se2[i, j] = value
                se2[j, i] = value
                total += value

        total_arm_variance = total / (len(arms) - 1)
        arm_variances: dict[str, float] = {}
        for idx, arm in enumerate(arms):
            row_sum = float(np.sum(se2[idx, :]))
            variance = (row_sum - total_arm_variance) / (len(arms) - 2)
            if variance < -1e-10:
                raise ValueError(
                    f"study {study_id}: recovered negative arm variance for {arm}; "
                    "multi-arm covariance is incompatible with a positive-definite GLS fit"
                )
            arm_variances[arm] = max(float(variance), 0.0)

        base = arms[0]
        contrasts: list[_StudyContrast] = []
        for arm in arms[1:]:
            pair = pair_rows[tuple(sorted((base, arm)))]
            est = pair.est
            if pair.t1 == arm and pair.t2 == base:
                est = -est
            elif pair.t1 == base and pair.t2 == arm:
                est = pair.est
            elif pair.t2 != arm:
                est = -est
            contrasts.append(_StudyContrast(to=arm, from_=base, est=est, se=None))

        studies.append(
            _Study(
                study_id=study_id,
                arms=arms,
                contrasts=tuple(contrasts),
                arm_variances=arm_variances,
                base=base,
                multi_arm=True,
            )
        )

    return tuple(studies), warnings


def _validate_connected(rows: tuple[ContrastRow, ...], reference: str) -> None:
    missing = _disconnected_treatments(rows, reference)
    if missing:
        raise ValueError(
            "disconnected treatment network; reference component does not include "
            f"{list(missing)}"
        )


def _disconnected_treatments(rows: tuple[ContrastRow, ...], reference: str) -> tuple[str, ...]:
    adjacency: dict[str, set[str]] = {}
    for row in rows:
        adjacency.setdefault(row.t1, set()).add(row.t2)
        adjacency.setdefault(row.t2, set()).add(row.t1)

    seen = {reference}
    stack = [reference]
    while stack:
        treatment = stack.pop()
        for neighbour in adjacency.get(treatment, set()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)

    missing = sorted(set(adjacency) - seen)
    return tuple(missing)


def _assemble(
    studies: tuple[_Study, ...],
    treatments: tuple[str, ...],
    reference: str,
) -> _AssembledNetwork:
    nonreference = tuple(treatment for treatment in treatments if treatment != reference)
    treatment_index = {treatment: idx for idx, treatment in enumerate(nonreference)}
    n_rows = sum(len(study.contrasts) for study in studies)
    n_cols = len(nonreference)
    x = np.zeros((n_rows, n_cols), dtype=float)
    y = np.zeros(n_rows, dtype=float)
    v = np.zeros((n_rows, n_rows), dtype=float)
    row_studies: list[str] = []
    row_contrasts: list[tuple[str, str]] = []
    blocks: list[tuple[int, int, bool]] = []

    cursor = 0
    for study in studies:
        start = cursor
        for contrast in study.contrasts:
            y[cursor] = contrast.est
            if contrast.to != reference:
                x[cursor, treatment_index[contrast.to]] += 1.0
            if contrast.from_ != reference:
                x[cursor, treatment_index[contrast.from_]] -= 1.0
            row_studies.append(study.study_id)
            row_contrasts.append((contrast.from_, contrast.to))
            cursor += 1

        width = len(study.contrasts)
        if study.multi_arm:
            assert study.arm_variances is not None
            assert study.base is not None
            base_variance = study.arm_variances[study.base]
            for i, contrast_i in enumerate(study.contrasts):
                v[start + i, start + i] = study.arm_variances[contrast_i.to] + base_variance
                for j in range(i + 1, width):
                    v[start + i, start + j] = base_variance
                    v[start + j, start + i] = base_variance
        else:
            se = study.contrasts[0].se
            assert se is not None
            v[start, start] = se * se
        blocks.append((start, width, study.multi_arm))

    return _AssembledNetwork(
        x=x,
        y=y,
        v=v,
        nonreference_treatments=nonreference,
        row_studies=tuple(row_studies),
        row_contrasts=tuple(row_contrasts),
        blocks=tuple(blocks),
    )


def _gls(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray] | None:
    try:
        _validate_positive_definite_matrix(v, label="within-study covariance")
        v_inv = np.linalg.inv(v)
        xt_v_inv = x.T @ v_inv
        xt_v_inv_x = xt_v_inv @ x
        cov = np.linalg.inv(xt_v_inv_x)
    except np.linalg.LinAlgError:
        return None
    _validate_positive_definite_matrix(cov, label="fitted-parameter covariance")
    d = cov @ xt_v_inv @ y
    return {"d": d, "cov": cov, "v_inv": v_inv}


def _validate_positive_definite_matrix(matrix: np.ndarray, *, label: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} matrix must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} matrix must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} matrix must be symmetric.")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError(f"{label} matrix must have a positive diagonal.")
    symmetrized = (matrix + matrix.T) / 2.0
    try:
        np.linalg.cholesky(symmetrized)
    except np.linalg.LinAlgError as exc:
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetrized)))
        raise ValueError(
            f"{label} matrix must be positive definite; "
            f"minimum eigenvalue is {min_eigenvalue:.6g}."
        ) from exc


def _generalized_dl(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    fixed_fit: dict[str, np.ndarray],
    blocks: tuple[tuple[int, int, bool], ...],
) -> dict[str, float]:
    w = fixed_fit["v_inv"]
    residual = y - x @ fixed_fit["d"]
    q = float(residual.T @ w @ residual)
    df = int(x.shape[0] - x.shape[1])

    xt_w_x_inv = fixed_fit["cov"]
    p_matrix = w - w @ x @ xt_w_x_inv @ x.T @ w
    r_matrix = _heterogeneity_correlation(x.shape[0], blocks)
    c_value = float(np.sum(p_matrix * r_matrix))
    tau2 = max(0.0, (q - df) / c_value) if c_value > 0.0 else 0.0
    return {"tau2": float(tau2), "q": q, "df": float(df), "c": c_value}


def _heterogeneity_correlation(
    n_rows: int,
    blocks: tuple[tuple[int, int, bool], ...],
) -> np.ndarray:
    r_matrix = np.eye(n_rows, dtype=float)
    for start, width, multi_arm in blocks:
        if not multi_arm:
            continue
        for i in range(width):
            for j in range(i + 1, width):
                r_matrix[start + i, start + j] = 0.5
                r_matrix[start + j, start + i] = 0.5
    return r_matrix


def _random_effects_covariance(
    v: np.ndarray,
    blocks: tuple[tuple[int, int, bool], ...],
    tau2: float,
) -> np.ndarray:
    random_v = np.array(v, copy=True)
    for start, width, multi_arm in blocks:
        for i in range(width):
            random_v[start + i, start + i] += tau2
            if multi_arm:
                for j in range(i + 1, width):
                    random_v[start + i, start + j] += tau2 / 2.0
                    random_v[start + j, start + i] += tau2 / 2.0
    return random_v


def _influence_diagnostics(
    assembled: _AssembledNetwork,
    fit: dict[str, np.ndarray],
    covariance: np.ndarray,
) -> tuple[ContrastInfluenceDiagnostic, ...]:
    fitted = assembled.x @ fit["d"]
    residual = assembled.y - fitted
    hat = assembled.x @ fit["cov"] @ assembled.x.T @ fit["v_inv"]
    diagonal_v = np.diag(covariance)
    n_parameters = max(int(assembled.x.shape[1]), 1)

    diagnostics: list[ContrastInfluenceDiagnostic] = []
    for idx, study in enumerate(assembled.row_studies):
        treatment_from, treatment_to = assembled.row_contrasts[idx]
        leverage = float(hat[idx, idx])
        variance = float(diagonal_v[idx])
        one_minus_leverage = max(1.0 - leverage, 1e-12)
        residual_scale = math.sqrt(max(variance * one_minus_leverage, 1e-12))
        standardized = float(residual[idx] / residual_scale)
        cook_denominator = n_parameters * one_minus_leverage
        cook = float((standardized * standardized * max(leverage, 0.0)) / cook_denominator)
        diagnostics.append(
            ContrastInfluenceDiagnostic(
                row_index=idx,
                study=study,
                treatment_from=treatment_from,
                treatment_to=treatment_to,
                observed=float(assembled.y[idx]),
                fitted=float(fitted[idx]),
                residual=float(residual[idx]),
                variance=variance,
                leverage=leverage,
                standardized_residual=standardized,
                cook_distance=cook,
            )
        )
    return tuple(diagnostics)


def _contribution_diagnostics(
    assembled: _AssembledNetwork,
    fit: dict[str, np.ndarray],
) -> tuple[ContrastContributionDiagnostic, ...]:
    mapping = fit["cov"] @ assembled.x.T @ fit["v_inv"]
    diagnostics: list[ContrastContributionDiagnostic] = []
    for parameter_idx, target_treatment in enumerate(assembled.nonreference_treatments):
        signed_weights = mapping[parameter_idx, :]
        absolute_weights = np.abs(signed_weights)
        denominator = float(np.sum(absolute_weights))
        if denominator <= 0.0:
            continue
        for row_idx, study in enumerate(assembled.row_studies):
            treatment_from, treatment_to = assembled.row_contrasts[row_idx]
            absolute_weight = float(absolute_weights[row_idx])
            diagnostics.append(
                ContrastContributionDiagnostic(
                    target_treatment=target_treatment,
                    row_index=row_idx,
                    study=study,
                    treatment_from=treatment_from,
                    treatment_to=treatment_to,
                    mapping_weight=float(signed_weights[row_idx]),
                    absolute_mapping_weight=absolute_weight,
                    contribution=float(absolute_weight / denominator),
                )
            )
    return tuple(diagnostics)
