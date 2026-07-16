"""Fail-closed CmdStan/NUTS integration preflight contracts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any


@dataclass(frozen=True)
class CmdStanPreflightReport:
    """Machine-readable availability report for CmdStan-backed sampling."""

    status: str
    cmdstanpy_available: bool
    cmdstan_available: bool
    cmdstan_path: str | None
    cmdstanpy_version: str | None
    required_diagnostics: tuple[str, ...]
    missing_components: tuple[str, ...]
    certification_effect: str
    message: str


REQUIRED_STAN_DIAGNOSTICS = (
    "r_hat",
    "ess_bulk",
    "ess_tail",
    "divergent_transitions",
    "treedepth_saturation",
    "mcse",
    "prior_predictive_checks",
    "posterior_predictive_checks",
)


def cmdstan_preflight() -> CmdStanPreflightReport:
    """Check whether CmdStanPy and CmdStan are available without running a model."""

    missing: list[str] = []
    if importlib.util.find_spec("cmdstanpy") is None:
        return CmdStanPreflightReport(
            status="unavailable",
            cmdstanpy_available=False,
            cmdstan_available=False,
            cmdstan_path=None,
            cmdstanpy_version=None,
            required_diagnostics=REQUIRED_STAN_DIAGNOSTICS,
            missing_components=("cmdstanpy", "cmdstan"),
            certification_effect="none",
            message="CmdStanPy is not installed; Stan/NUTS backend remains unavailable.",
        )

    try:
        import cmdstanpy  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - defensive environment guard
        return CmdStanPreflightReport(
            status="failed",
            cmdstanpy_available=True,
            cmdstan_available=False,
            cmdstan_path=None,
            cmdstanpy_version=None,
            required_diagnostics=REQUIRED_STAN_DIAGNOSTICS,
            missing_components=("cmdstan",),
            certification_effect="none",
            message=f"CmdStanPy import failed: {exc}",
        )

    version = str(getattr(cmdstanpy, "__version__", "unknown"))
    cmdstan_path: str | None = None
    cmdstan_available = False
    try:
        cmdstan_path = str(cmdstanpy.cmdstan_path())
        cmdstan_available = bool(cmdstan_path)
    except Exception:
        missing.append("cmdstan")

    if not cmdstan_available:
        return CmdStanPreflightReport(
            status="unavailable",
            cmdstanpy_available=True,
            cmdstan_available=False,
            cmdstan_path=cmdstan_path,
            cmdstanpy_version=version,
            required_diagnostics=REQUIRED_STAN_DIAGNOSTICS,
            missing_components=tuple(missing or ["cmdstan"]),
            certification_effect="none",
            message="CmdStanPy is installed, but CmdStan is not available.",
        )

    return CmdStanPreflightReport(
        status="available",
        cmdstanpy_available=True,
        cmdstan_available=True,
        cmdstan_path=cmdstan_path,
        cmdstanpy_version=version,
        required_diagnostics=REQUIRED_STAN_DIAGNOSTICS,
        missing_components=(),
        certification_effect="none",
        message="CmdStanPy and CmdStan are available; model-specific reference runs are still required.",
    )


def summarize_cmdstan_preflight(report: CmdStanPreflightReport) -> dict[str, Any]:
    """Return JSON/TOML-friendly preflight fields."""

    return {
        "status": report.status,
        "cmdstanpy_available": report.cmdstanpy_available,
        "cmdstan_available": report.cmdstan_available,
        "cmdstan_path": report.cmdstan_path,
        "cmdstanpy_version": report.cmdstanpy_version,
        "required_diagnostics": list(report.required_diagnostics),
        "missing_components": list(report.missing_components),
        "certification_effect": report.certification_effect,
        "message": report.message,
    }
