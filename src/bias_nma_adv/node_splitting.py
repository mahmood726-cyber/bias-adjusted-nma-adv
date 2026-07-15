"""Experimental fixed-effect node-splitting diagnostics for contrast NMA."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from bias_nma_adv.multiarm import ContrastRow, fit_multiarm_gls
from bias_nma_adv.pairwise import fit_pairwise_meta


@dataclass(frozen=True)
class NodeSplitDiagnostic:
    """Direct-versus-indirect diagnostic for one treatment comparison."""

    treatment_from: str
    treatment_to: str
    status: str
    n_direct_contrasts: int
    direct_estimate: float | None
    direct_se: float | None
    indirect_estimate: float | None
    indirect_se: float | None
    difference: float | None
    difference_se: float | None
    z_value: float | None
    p_value: float | None
    warning: str | None


def fixed_effect_node_splitting(
    rows: Iterable[ContrastRow | dict[str, object]],
    *,
    reference_treatment: str | None = None,
) -> tuple[NodeSplitDiagnostic, ...]:
    """Compare direct and indirect fixed-effect evidence for observed contrasts.

    Each diagnostic is oriented as `treatment_to - treatment_from` using
    lexicographic treatment ordering for reproducibility. If removing direct
    evidence disconnects the remaining network, the comparison is returned as
    `not_estimable` rather than filled with a synthetic indirect estimate.
    """

    parsed_rows = tuple(_coerce_row(row) for row in rows)
    if not parsed_rows:
        raise ValueError("no contrasts supplied.")

    pairs = tuple(sorted({tuple(sorted((row.t1, row.t2))) for row in parsed_rows}))
    diagnostics: list[NodeSplitDiagnostic] = []
    for treatment_from, treatment_to in pairs:
        direct_rows = [
            row for row in parsed_rows if {row.t1, row.t2} == {treatment_from, treatment_to}
        ]
        effects = np.asarray(
            [_oriented_effect(row, treatment_from, treatment_to) for row in direct_rows],
            dtype=float,
        )
        variances = np.asarray([row.se * row.se for row in direct_rows], dtype=float)
        direct = fit_pairwise_meta(effects, variances, method="FE")

        indirect_rows = tuple(
            row for row in parsed_rows if {row.t1, row.t2} != {treatment_from, treatment_to}
        )
        if not indirect_rows:
            diagnostics.append(
                _not_estimable(
                    treatment_from,
                    treatment_to,
                    len(direct_rows),
                    direct.estimate,
                    direct.se,
                    "no indirect evidence remains after direct contrasts are removed",
                )
            )
            continue

        try:
            indirect_fit = fit_multiarm_gls(
                indirect_rows,
                reference_treatment=reference_treatment,
                model="fixed",
            )
            indirect_estimate, indirect_se = indirect_fit.contrast(treatment_to, treatment_from)
        except ValueError as exc:
            diagnostics.append(
                _not_estimable(
                    treatment_from,
                    treatment_to,
                    len(direct_rows),
                    direct.estimate,
                    direct.se,
                    str(exc),
                )
            )
            continue

        difference = float(direct.estimate - indirect_estimate)
        difference_se = math.sqrt(direct.se * direct.se + indirect_se * indirect_se)
        z_value = difference / difference_se if difference_se > 0.0 else math.nan
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0)) if math.isfinite(z_value) else math.nan
        diagnostics.append(
            NodeSplitDiagnostic(
                treatment_from=treatment_from,
                treatment_to=treatment_to,
                status="estimable",
                n_direct_contrasts=len(direct_rows),
                direct_estimate=float(direct.estimate),
                direct_se=float(direct.se),
                indirect_estimate=float(indirect_estimate),
                indirect_se=float(indirect_se),
                difference=difference,
                difference_se=float(difference_se),
                z_value=float(z_value),
                p_value=float(p_value),
                warning=None,
            )
        )

    return tuple(diagnostics)


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


def _oriented_effect(row: ContrastRow, treatment_from: str, treatment_to: str) -> float:
    if row.t1 == treatment_from and row.t2 == treatment_to:
        return row.est
    if row.t1 == treatment_to and row.t2 == treatment_from:
        return -row.est
    raise ValueError(
        f"row {row.study} is not a {treatment_from} to {treatment_to} contrast."
    )


def _not_estimable(
    treatment_from: str,
    treatment_to: str,
    n_direct_contrasts: int,
    direct_estimate: float,
    direct_se: float,
    warning: str,
) -> NodeSplitDiagnostic:
    return NodeSplitDiagnostic(
        treatment_from=treatment_from,
        treatment_to=treatment_to,
        status="not_estimable",
        n_direct_contrasts=n_direct_contrasts,
        direct_estimate=float(direct_estimate),
        direct_se=float(direct_se),
        indirect_estimate=None,
        indirect_se=None,
        difference=None,
        difference_se=None,
        z_value=None,
        p_value=None,
        warning=warning,
    )
