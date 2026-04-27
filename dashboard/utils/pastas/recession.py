"""Recession analysis and Master Recession Curve extraction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _exp_decay_with_baseline(t, amplitude, T, baseline):
    return baseline + amplitude * np.exp(-t / T)


def compute_recession_analysis(
    model, tmin: str, tmax: str,
    min_duration_days: int = 30,
    min_drop_m: float = 0.05,
    max_interruption_days: int = 3,
) -> dict:
    """Extract recession segments, fit exponential decay with baseline, compute MRC."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    if obs is None or len(obs) < 60:
        return {"segments": [], "mrc": None, "T_median_days": None,
                "T_mean_days": None, "T_std_days": None, "n_segments": 0}

    obs_clean = obs.dropna()
    values = obs_clean.values.flatten()
    dates = obs_clean.index
    diff = np.diff(values)

    segments = []
    in_recession = False
    start_idx = 0
    interruption_count = 0

    for i in range(len(diff)):
        if diff[i] < 0:
            if not in_recession:
                in_recession = True
                start_idx = i
                interruption_count = 0
            else:
                interruption_count = 0
        else:
            if in_recession:
                interruption_count += 1
                if interruption_count > max_interruption_days:
                    end_idx = i - interruption_count
                    duration = end_idx - start_idx
                    drop = values[start_idx] - values[end_idx]
                    if duration >= min_duration_days and drop >= min_drop_m:
                        segments.append((start_idx, end_idx))
                    in_recession = False
                    interruption_count = 0

    if in_recession:
        end_idx = len(diff) - interruption_count
        duration = end_idx - start_idx
        if end_idx > start_idx:
            drop = values[start_idx] - values[end_idx]
            if duration >= min_duration_days and drop >= min_drop_m:
                segments.append((start_idx, end_idx))

    fitted_segments = []
    all_T = []
    normalized_curves = []

    for start_idx, end_idx in segments:
        seg_values = values[start_idx:end_idx + 1]
        seg_dates = dates[start_idx:end_idx + 1]
        t = np.arange(len(seg_values), dtype=float)

        amplitude_init = seg_values[0] - seg_values[-1]
        baseline_init = seg_values[-1]
        if amplitude_init <= 0:
            continue

        try:
            popt, _ = curve_fit(
                _exp_decay_with_baseline, t, seg_values,
                p0=[amplitude_init, 50.0, baseline_init],
                bounds=([0, 1, -np.inf], [np.inf, 10000, np.inf]),
                maxfev=5000,
            )
            amplitude_fit, T_fit, baseline_fit = popt
            predicted = _exp_decay_with_baseline(t, amplitude_fit, T_fit, baseline_fit)
            ss_res = np.sum((seg_values - predicted) ** 2)
            ss_tot = np.sum((seg_values - np.mean(seg_values)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if T_fit > 0 and r_squared > 0.3:
                fitted_segments.append({
                    "start": str(seg_dates[0].date()) if hasattr(seg_dates[0], 'date') else str(seg_dates[0])[:10],
                    "end": str(seg_dates[-1].date()) if hasattr(seg_dates[-1], 'date') else str(seg_dates[-1])[:10],
                    "h0": round(float(seg_values[0]), 3),
                    "h_end": round(float(seg_values[-1]), 3),
                    "T_days": round(float(T_fit), 1),
                    "r_squared": round(float(r_squared), 3),
                    "duration_days": len(seg_values),
                })
                all_T.append(T_fit)

                norm = (seg_values - baseline_fit) / amplitude_fit if amplitude_fit > 0 else seg_values / seg_values[0]
                normalized_curves.append((t, norm))
        except (RuntimeError, ValueError):
            continue

    mrc = None
    if normalized_curves:
        max_len = max(len(t) for t, _ in normalized_curves)
        t_mrc = np.arange(max_len, dtype=float)
        all_norm = []
        for t, norm in normalized_curves:
            padded = np.full(max_len, np.nan)
            padded[:len(norm)] = norm
            all_norm.append(padded)
        mean_curve = np.nanmean(all_norm, axis=0)
        valid = ~np.isnan(mean_curve)
        mrc = {
            "normalized_time": t_mrc[valid].tolist(),
            "normalized_level": mean_curve[valid].tolist(),
        }

    return {
        "segments": fitted_segments,
        "mrc": mrc,
        "T_median_days": round(float(np.median(all_T)), 1) if all_T else None,
        "T_mean_days": round(float(np.mean(all_T)), 1) if all_T else None,
        "T_std_days": round(float(np.std(all_T)), 1) if all_T else None,
        "n_segments": len(fitted_segments),
    }
