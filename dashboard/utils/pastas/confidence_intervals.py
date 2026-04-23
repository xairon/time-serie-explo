"""Bootstrap confidence bands for Pastas model simulations."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_confidence_bands(model, tmin: str, tmax: str, n_bootstrap: int = 200) -> dict:
    """Bootstrap residuals to compute prediction confidence bands."""
    residuals = model.residuals(tmin=tmin, tmax=tmax)
    simulation = model.simulate(tmin=tmin, tmax=tmax)

    clean_residuals = residuals.dropna().values
    sim_values = simulation.values.flatten()

    if len(clean_residuals) < 20:
        return {
            "index": [d.isoformat() for d in simulation.index],
            "p5": sim_values.tolist(), "p25": sim_values.tolist(),
            "p75": sim_values.tolist(), "p95": sim_values.tolist(),
            "simulation": sim_values.tolist(),
        }

    rng = np.random.default_rng(42)
    n_steps = len(sim_values)
    bootstrap_matrix = np.zeros((n_bootstrap, n_steps))

    for i in range(n_bootstrap):
        resampled = rng.choice(clean_residuals, size=n_steps, replace=True)
        bootstrap_matrix[i] = sim_values + resampled

    return {
        "index": [d.isoformat() for d in simulation.index],
        "p5": np.percentile(bootstrap_matrix, 5, axis=0).tolist(),
        "p25": np.percentile(bootstrap_matrix, 25, axis=0).tolist(),
        "p75": np.percentile(bootstrap_matrix, 75, axis=0).tolist(),
        "p95": np.percentile(bootstrap_matrix, 95, axis=0).tolist(),
        "simulation": sim_values.tolist(),
    }
