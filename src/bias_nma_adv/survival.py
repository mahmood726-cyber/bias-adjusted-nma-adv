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


def _pava_decreasing(values: List[float]) -> np.ndarray:
    """Pool-adjacent violators projection for a non-increasing sequence."""

    sequence = list(map(float, values))
    if not sequence:
        return np.asarray([], dtype=float)
    # Negate and use the standard increasing-PAVA merge rule.
    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    for index, value in enumerate((-item for item in sequence)):
        current_value = float(value)
        current_weight = 1.0
        current_start = index
        while block_values and block_values[-1] >= current_value:
            previous_value = block_values.pop()
            previous_weight = block_weights.pop()
            current_start = block_starts.pop()
            current_value = (
                previous_value * previous_weight + current_value * current_weight
            ) / (previous_weight + current_weight)
            current_weight += previous_weight
        block_values.append(current_value)
        block_weights.append(current_weight)
        block_starts.append(current_start)

    output = [0.0] * len(sequence)
    end = len(sequence)
    for block_index in range(len(block_values) - 1, -1, -1):
        start = block_starts[block_index]
        for index in range(start, end):
            output[index] = -block_values[block_index]
        end = start
    return np.asarray(output, dtype=float)


def _build_risk_indices(curve_times: List[float], risk_times: List[float]) -> tuple[list[int], list[int]]:
    """Map number-at-risk anchors to digitized curve intervals."""

    n_times = len(curve_times)
    lower = [0] * len(risk_times)
    upper = [0] * len(risk_times)
    for i, risk_time in enumerate(risk_times):
        k = 0
        while k < n_times and curve_times[k] < risk_time - 1e-9:
            k += 1
        lower[i] = min(k, n_times - 1)
    for i in range(len(risk_times)):
        upper[i] = (
            max(lower[i + 1] - 1, lower[i])
            if i < len(risk_times) - 1
            else n_times - 1
        )
    return lower, upper


def _guyot_core(
    curve_times: List[float],
    survivals: np.ndarray,
    risk_times: List[float],
    risk_values: List[int],
    total_events: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """Iteratively estimate events and censoring while matching risk anchors."""

    lower, upper = _build_risk_indices(curve_times, risk_times)
    n_intervals = len(risk_values)
    n_times = len(curve_times)
    n_censor = [0] * n_intervals
    n_hat = [risk_values[0] + 1] * (n_times + 1)
    censored = [0] * n_times
    events = [0] * n_times
    km_hat = [1.0] * n_times
    last_event_index = [0] * n_intervals

    def distribute_censoring(interval_index: int, count: int) -> None:
        for k in range(lower[interval_index], upper[interval_index] + 1):
            censored[k] = 0
        if count <= 0:
            return
        start = curve_times[lower[interval_index]]
        end = curve_times[min(lower[interval_index + 1], n_times - 1)]
        span = (end - start) or 1.0
        for censor_index in range(count):
            censor_time = start + span * (censor_index + 0.5) / count
            k = lower[interval_index]
            while k < upper[interval_index] and curve_times[k + 1] <= censor_time + 1e-12:
                k += 1
            censored[k] += 1

    for interval_index in range(n_intervals - 1):
        s_lower = float(survivals[lower[interval_index]]) or 1e-12
        n_censor[interval_index] = int(
            round(
                risk_values[interval_index]
                * (survivals[lower[interval_index + 1]] / s_lower)
                - risk_values[interval_index + 1]
            )
        )
        guard = 0
        while (
            n_hat[lower[interval_index + 1]] > risk_values[interval_index + 1]
            or (
                n_hat[lower[interval_index + 1]] < risk_values[interval_index + 1]
                and n_censor[interval_index] > 0
            )
        ):
            guard += 1
            if guard > 5000:
                break
            if n_censor[interval_index] <= 0:
                for k in range(lower[interval_index], upper[interval_index] + 1):
                    censored[k] = 0
                n_censor[interval_index] = 0
            else:
                distribute_censoring(interval_index, n_censor[interval_index])

            n_hat[lower[interval_index]] = risk_values[interval_index]
            last = last_event_index[interval_index]
            for k in range(lower[interval_index], upper[interval_index] + 1):
                if interval_index == 0 and k == lower[interval_index]:
                    events[k] = 0
                    km_hat[k] = 1.0
                else:
                    reference_survival = km_hat[last] or 1e-12
                    events[k] = int(round(n_hat[k] * (1.0 - survivals[k] / reference_survival)))
                    events[k] = max(0, min(events[k], n_hat[k]))
                km_hat[k] = (km_hat[last] or 1.0) * (
                    1.0 - events[k] / (n_hat[k] or 1)
                )
                n_hat[k + 1] = max(0, n_hat[k] - events[k] - censored[k])
                if events[k] != 0:
                    last = k
            n_censor[interval_index] += (
                n_hat[lower[interval_index + 1]] - risk_values[interval_index + 1]
            )
            last_event_index[interval_index + 1] = last

    interval_index = n_intervals - 1
    if n_times - 1 >= lower[interval_index]:
        n_hat[lower[interval_index]] = risk_values[interval_index]
        last = last_event_index[interval_index]
        for k in range(lower[interval_index], n_times):
            if interval_index == 0 and k == lower[interval_index]:
                events[k] = 0
                km_hat[k] = 1.0
                n_hat[k + 1] = n_hat[k] - censored[k]
                continue
            reference_survival = km_hat[last] or 1e-12
            events[k] = int(round(n_hat[k] * (1.0 - survivals[k] / reference_survival)))
            events[k] = max(0, min(events[k], n_hat[k]))
            km_hat[k] = reference_survival * (1.0 - events[k] / (n_hat[k] or 1))
            censored[k] = max(0, n_hat[k] - events[k]) if k == n_times - 1 else 0
            n_hat[k + 1] = max(0, n_hat[k] - events[k] - censored[k])
            if events[k] != 0:
                last = k

    if total_events is not None and total_events < 0:
        raise ValueError("total_events must be non-negative when supplied.")
    return events, censored


def _normalize_and_expand(
    curve_times: List[float],
    events: List[int],
    censored: List[int],
    total_n: int,
    *,
    total_events: Optional[int] = None,
    follow_up: Optional[float] = None,
    arm: int = 0,
) -> list[IPDRecord]:
    """Expand event/censor counts into exactly ``total_n`` IPD records."""

    n_times = len(curve_times)
    event_counts = list(events)
    censor_counts = list(censored)
    remaining = int(total_n)
    for k in range(n_times):
        event_counts[k] = max(0, min(int(event_counts[k]), remaining))
        censor_counts[k] = max(0, min(int(censor_counts[k]), remaining - event_counts[k]))
        remaining -= event_counts[k] + censor_counts[k]
    tail_censored = max(0, remaining)
    tail_time = float(follow_up) if follow_up is not None else float(curve_times[-1])

    if total_events is not None:
        delta = int(total_events) - sum(event_counts)
        if delta > 0:
            guard = 0
            while delta > 0 and guard < 1000:
                guard += 1
                weighted_events = sum(
                    event_counts[k] for k in range(1, n_times) if censor_counts[k] > 0
                )
                if weighted_events <= 0:
                    break
                moved = 0
                for k in range(1, n_times):
                    if delta <= 0:
                        break
                    if censor_counts[k] <= 0:
                        continue
                    desired = max(0, int(round(delta * event_counts[k] / weighted_events)))
                    take = min(desired, censor_counts[k], delta)
                    censor_counts[k] -= take
                    event_counts[k] += take
                    delta -= take
                    moved += take
                if moved == 0:
                    break
            for k in range(n_times - 1, 0, -1):
                if delta <= 0:
                    break
                take = min(censor_counts[k], delta)
                censor_counts[k] -= take
                event_counts[k] += take
                delta -= take
            while delta > 0 and tail_censored > 0:
                tail_censored -= 1
                event_counts[n_times - 1] += 1
                delta -= 1
        elif delta < 0:
            needed = -delta
            total_observed_events = sum(event_counts[k] for k in range(1, n_times))
            for k in range(n_times - 1, 0, -1):
                if needed <= 0:
                    break
                desired = (
                    min(event_counts[k], int(round((-delta) * event_counts[k] / total_observed_events)))
                    if total_observed_events > 0
                    else event_counts[k]
                )
                take = min(desired, event_counts[k], needed)
                event_counts[k] -= take
                censor_counts[k] += take
                needed -= take
            for k in range(n_times - 1, 0, -1):
                if needed <= 0:
                    break
                take = min(event_counts[k], needed)
                event_counts[k] -= take
                censor_counts[k] += take
                needed -= take

    records: list[IPDRecord] = []
    for k, time in enumerate(curve_times):
        records.extend(IPDRecord(time=float(time), event=1, arm=arm) for _ in range(event_counts[k]))
        records.extend(IPDRecord(time=float(time), event=0, arm=arm) for _ in range(censor_counts[k]))
    records.extend(IPDRecord(time=tail_time, event=0, arm=arm) for _ in range(tail_censored))
    return records


def reconstruct_ipd_guyot(
    times: np.ndarray,
    survivals: np.ndarray,
    n_risk_times: Optional[np.ndarray] = None,
    n_risk_values: Optional[np.ndarray] = None,
    total_n: Optional[int] = None,
    arm: int = 0,
    total_events: Optional[int] = None,
    follow_up: Optional[float] = None,
) -> List[IPDRecord]:
    """Reconstruct IPD from a KM curve via a faithful Guyot-style loop.

    Parameters
    ----------
    times, survivals : digitised curve points (survival in [0, 1]).
    n_risk_times, n_risk_values : the at-risk table (times and counts).
    total_n : total patients; defaults to the first at-risk count.
    arm : arm label written onto each record (0/1).
    total_events : optional source-reported event total to reconcile by swapping
        censor/event indicators without changing the population size.
    follow_up : optional administrative censoring time for tail records.
    """
    times = np.asarray(times, dtype=float)
    survivals = np.asarray(survivals, dtype=float)

    if times.shape != survivals.shape:
        raise ValueError("times and survivals must have the same shape.")
    if times.size == 0:
        return []

    points = [
        (float(time), float(survival))
        for time, survival in zip(times, survivals)
        if np.isfinite(time) and np.isfinite(survival)
    ]
    points.sort(key=lambda item: item[0])
    if not points:
        return []
    if points[0][0] > 1e-9 or points[0][1] < 1.0 - 1e-9:
        points.insert(0, (0.0, 1.0))
    curve_times = [item[0] for item in points]
    survival_curve = _pava_decreasing(
        [min(1.0, max(0.0, item[1])) for item in points]
    )

    if n_risk_times is not None and n_risk_values is not None and len(n_risk_times):
        risk_order = np.argsort(np.asarray(n_risk_times, dtype=float))
        risk_times = [float(np.asarray(n_risk_times, dtype=float)[i]) for i in risk_order]
        risk_values = [int(round(float(np.asarray(n_risk_values, dtype=float)[i]))) for i in risk_order]
    else:
        risk_times = []
        risk_values = []

    if total_n is None:
        if risk_values:
            total_n = int(risk_values[0])
        else:
            total_n = 100
    if int(total_n) <= 0:
        raise ValueError("total_n must be positive.")
    if not risk_times or risk_times[0] > curve_times[0] + 1e-9:
        risk_times = [curve_times[0]] + risk_times
        risk_values = [int(total_n)] + risk_values

    events, censored = _guyot_core(
        curve_times,
        survival_curve,
        risk_times,
        risk_values,
        total_events=total_events,
    )
    return _normalize_and_expand(
        curve_times,
        events,
        censored,
        int(total_n),
        total_events=total_events,
        follow_up=follow_up,
        arm=arm,
    )


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
