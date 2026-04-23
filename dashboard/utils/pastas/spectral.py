"""Power spectral density comparison between observed and simulated series."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import welch


def compute_spectral_analysis(model, tmin: str, tmax: str) -> dict:
    """Compute PSD for observed and simulated series using Welch's method."""
    obs = model.observations(tmin=tmin, tmax=tmax)
    sim = model.simulate(tmin=tmin, tmax=tmax)

    if obs is None or sim is None or len(obs) < 365:
        return {"frequencies": [], "periods_days": [], "psd_observed": [],
                "psd_simulated": [], "coherence": []}

    obs_clean = obs.dropna()
    sim_aligned = sim.reindex(obs_clean.index).dropna()

    if len(sim_aligned) < 365:
        return {"frequencies": [], "periods_days": [], "psd_observed": [],
                "psd_simulated": [], "coherence": []}

    obs_vals = obs_clean.loc[sim_aligned.index].values.flatten()
    sim_vals = sim_aligned.values.flatten()

    # Remove mean (detrend)
    obs_vals = obs_vals - np.mean(obs_vals)
    sim_vals = sim_vals - np.mean(sim_vals)

    # Welch PSD (fs=1 sample/day, nperseg=min(365, n//2))
    nperseg = min(365, len(obs_vals) // 2)
    if nperseg < 64:
        return {"frequencies": [], "periods_days": [], "psd_observed": [],
                "psd_simulated": [], "coherence": []}

    f_obs, psd_obs = welch(obs_vals, fs=1.0, nperseg=nperseg)
    f_sim, psd_sim = welch(sim_vals, fs=1.0, nperseg=nperseg)

    # Skip DC component (f=0)
    mask = f_obs > 0
    freqs = f_obs[mask]
    periods = 1.0 / freqs

    # Coherence: ratio of PSDs (clipped for display)
    psd_o = psd_obs[mask]
    psd_s = psd_sim[mask]
    coherence = np.where(psd_o > 0, psd_s / psd_o, 1.0)
    coherence = np.clip(coherence, 0.01, 100.0)

    return {
        "frequencies": freqs.tolist(),
        "periods_days": periods.tolist(),
        "psd_observed": psd_o.tolist(),
        "psd_simulated": psd_s.tolist(),
        "coherence": coherence.tolist(),
    }
