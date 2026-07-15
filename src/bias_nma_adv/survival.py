"""Survival reconstruction and Time-to-Event analysis using the Guyot algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    from scipy.interpolate import interp1d
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

@dataclass
class IPDRecord:
    """Individual patient survival data record."""
    time: float
    event: int  # 1 = event (death/progression), 0 = censored
    arm: int = 0  # 0 = baseline/control, 1 = active/comparative


def _interpolate_nar(
    nar_times: np.ndarray,
    nar_values: np.ndarray,
    curve_times: np.ndarray,
    total_n: int,
) -> np.ndarray:
    """Interpolate numbers-at-risk onto the curve timepoints (monotone)."""
    nar_times = np.array(nar_times, dtype=float)
    nar_values = np.array(nar_values, dtype=float)

    for i in range(1, len(nar_values)):
        if nar_values[i] > nar_values[i - 1]:
            nar_values[i] = nar_values[i - 1]

    if _HAVE_SCIPY and len(nar_times) >= 2:
        try:
            f = interp1d(
                nar_times,
                nar_values,
                kind="linear",
                bounds_error=False,
                fill_value=(nar_values[0], nar_values[-1]),
            )
            interp = f(curve_times)
        except Exception:
            interp = np.interp(curve_times, nar_times, nar_values)
    else:
        interp = np.interp(curve_times, nar_times, nar_values)

    interp = np.clip(interp, 1, total_n)
    for i in range(1, len(interp)):
        if interp[i] > interp[i - 1]:
            interp[i] = interp[i - 1]
    return interp.astype(int)


def reconstruct_ipd_guyot(
    times: np.ndarray,
    survivals: np.ndarray,
    n_risk_times: Optional[np.ndarray] = None,
    n_risk_values: Optional[np.ndarray] = None,
    total_n: Optional[int] = None,
    arm: int = 0,
) -> List[IPDRecord]:
    """Reconstruct IPD from a KM curve via the Guyot algorithm.

    Parameters
    ----------
    times, survivals : digitised curve points (survival in [0, 1]).
    n_risk_times, n_risk_values : the at-risk table (times and counts).
    total_n : total patients; defaults to the first at-risk count.
    arm : arm label written onto each record (0/1).
    """
    times = np.asarray(times, dtype=float)
    survivals = np.asarray(survivals, dtype=float)

    if times.size == 0:
        return []

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    survivals = survivals[sort_idx]

    # survival must be monotone non-increasing
    for i in range(1, len(survivals)):
        if survivals[i] > survivals[i - 1]:
            survivals[i] = survivals[i - 1]

    explicit_n = total_n is not None

    if total_n is None:
        if n_risk_values is not None and len(n_risk_values) > 0:
            total_n = int(n_risk_values[0])
        else:
            total_n = 100

    has_nar = (
        n_risk_times is not None
        and n_risk_values is not None
        and len(n_risk_times) >= 2
        and len(n_risk_values) >= 2
    )
    if has_nar:
        n_at_risk = _interpolate_nar(n_risk_times, n_risk_values, times, total_n)
    else:
        n_at_risk = np.maximum((survivals * total_n).astype(int), 1)

    ipd: List[IPDRecord] = []
    event_carry = 0.0  # fractional-event accumulator (error diffusion)
    km_recon = 1.0  # reconstructed survival so far (Guyot recursion anchor)
    for i in range(1, len(times)):
        s_curr = survivals[i]
        n_prev = n_at_risk[i - 1]
        n_curr = n_at_risk[i] if i < len(n_at_risk) else 1
        t_prev, t_curr = times[i - 1], times[i]

        if km_recon <= 0 or s_curr <= 0:
            continue

        cond_prob = 1.0 - (s_curr / km_recon)
        cond_prob = min(max(cond_prob, 0.0), 1.0)
        event_carry += n_prev * cond_prob
        n_events = int(np.floor(event_carry))
        event_carry -= n_events
        n_events = max(0, min(n_events, n_prev - 1))
        
        if n_prev > 0:
            km_recon *= 1.0 - n_events / n_prev

        n_censored = max(0, n_prev - n_curr - n_events)

        if n_events > 0:
            for et in np.linspace(t_prev + 1e-3, t_curr - 2e-3, n_events):
                ipd.append(IPDRecord(time=float(et), event=1, arm=arm))
        if n_censored > 0:
            for ct in np.linspace(t_curr - 1e-3, t_curr - 0.5e-3, n_censored):
                ipd.append(IPDRecord(time=float(ct), event=0, arm=arm))

    n_remaining = total_n - len(ipd)
    if n_remaining > 0:
        final_censored = n_remaining if (has_nar or explicit_n) else min(n_remaining, 10)
        for _ in range(final_censored):
            ipd.append(IPDRecord(time=float(times[-1]), event=0, arm=arm))

    return ipd


class SurvivalIPDReconstructor:
    """Helper wrapper to reconstruct and format multi-arm time-to-event datasets."""

    def __init__(self):
        self.arms_data: dict[int, list[IPDRecord]] = {}

    def add_arm_curve(
        self,
        arm_id: int,
        times: List[float] | np.ndarray,
        survivals: List[float] | np.ndarray,
        n_risk_times: Optional[List[float] | np.ndarray] = None,
        n_risk_values: Optional[List[int] | np.ndarray] = None,
        total_n: Optional[int] = None
    ) -> None:
        """Reconstruct IPD records for a single study arm."""
        t_arr = np.asarray(times, dtype=float)
        s_arr = np.asarray(survivals, dtype=float)
        
        nr_times = np.asarray(n_risk_times, dtype=float) if n_risk_times is not None else None
        nr_vals = np.asarray(n_risk_values, dtype=int) if n_risk_values is not None else None
        
        ipd = reconstruct_ipd_guyot(
            t_arr, s_arr, nr_times, nr_vals, total_n, arm=arm_id
        )
        self.arms_data[arm_id] = ipd

    def get_combined_ipd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine all arms into flat arrays suitable for modeling.

        Returns (times, events, arm_labels).
        """
        all_times = []
        all_events = []
        all_arms = []
        
        for arm_id, records in self.arms_data.items():
            for rec in records:
                all_times.append(rec.time)
                all_events.append(rec.event)
                all_arms.append(rec.arm)
                
        return (
            np.array(all_times, dtype=float),
            np.array(all_events, dtype=int),
            np.array(all_arms, dtype=int)
        )

    def audit_reconstruction(
        self,
        arm_id: int,
        digitized_times: List[float] | np.ndarray,
        digitized_survivals: List[float] | np.ndarray
    ) -> Tuple[float, bool, str]:
        """Audit the reconstruction quality of an arm using Integrated Absolute Error (IAE).

        Returns (integrated_absolute_error, is_valid, warning_message).
        """
        records = self.arms_data.get(arm_id, [])
        if not records:
            return 0.0, False, "No reconstructed IPD available for this arm."

        rec_times = np.array([r.time for r in records], dtype=float)
        rec_events = np.array([r.event for r in records], dtype=int)

        # Compute KM curve from reconstructed IPD
        km_times, km_survivals = calculate_km_curve(rec_times, rec_events)

        # Calculate Wasserstein distance
        dist = calculate_wasserstein_distance(
            np.asarray(digitized_times, dtype=float),
            np.asarray(digitized_survivals, dtype=float),
            km_times,
            km_survivals
        )

        # Normalize by follow-up span to get scale-free IAE
        span = float(km_times[-1] - km_times[0])
        iae = dist / max(span, 1e-6)

        is_valid = bool(iae <= 0.02)
        warning = "" if is_valid else f"Warning: High reconstruction error (Integrated Absolute Error = {iae:.4f} > 0.02)."

        return iae, is_valid, warning


def calculate_km_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Kaplan-Meier survival curve from patient-level IPD."""
    if len(times) == 0:
        return np.array([0.0]), np.array([1.0])

    idx = np.argsort(times)
    sorted_times = times[idx]
    sorted_events = events[idx]

    unique_times = np.unique(sorted_times)
    survivals = []
    current_s = 1.0
    n = len(sorted_times)

    for ut in unique_times:
        d = np.sum((sorted_times == ut) & (sorted_events == 1))
        c = np.sum((sorted_times == ut) & (sorted_events == 0))
        if n > 0:
            current_s *= (1.0 - d / n)
            n -= (d + c)
        survivals.append(current_s)

    return np.concatenate([[0.0], unique_times]), np.concatenate([[1.0], survivals])


def calculate_wasserstein_distance(
    times1: np.ndarray,
    survivals1: np.ndarray,
    times2: np.ndarray,
    survivals2: np.ndarray,
    max_time: Optional[float] = None
) -> float:
    """Calculate the L1 Wasserstein distance between two survival curves."""
    if len(times1) == 0 or len(times2) == 0:
        return 0.0

    # Ensure monotonic non-increasing curves
    surv1 = np.asarray(survivals1, dtype=float).copy()
    surv2 = np.asarray(survivals2, dtype=float).copy()
    for i in range(1, len(surv1)):
        surv1[i] = min(surv1[i], surv1[i-1])
    for i in range(1, len(surv2)):
        surv2[i] = min(surv2[i], surv2[i-1])

    # Construct common grid
    grid = sorted(list(set(times1) | set(times2)))
    if max_time is not None:
        grid = [t for t in grid if t <= max_time]

    # Step function interpolation on common grid
    def step_interpolate(t_arr, s_arr, g):
        interpolated = []
        for t in g:
            idx = np.where(t_arr <= t)[0]
            if len(idx) > 0:
                interpolated.append(s_arr[idx[-1]])
            else:
                interpolated.append(1.0)
        return np.array(interpolated)

    s1_interp = step_interpolate(times1, surv1, grid)
    s2_interp = step_interpolate(times2, surv2, grid)

    # Riemann sum integral of absolute differences
    total_dist = 0.0
    for i in range(1, len(grid)):
        width = grid[i] - grid[i-1]
        diff = abs(s1_interp[i-1] - s2_interp[i-1])
        total_dist += diff * width

    return total_dist
