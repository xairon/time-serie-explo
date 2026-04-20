"""Compare multiple fitted Pastas models."""
from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd

from dashboard.utils.pastas.io import load_model


def compare_models(run_ids: list[str]) -> list[dict[str, Any]]:
    """Load N models and return side-by-side metrics + aligned series."""
    client = mlflow.tracking.MlflowClient()
    results = []

    for run_id in run_ids:
        run = client.get_run(run_id)
        model = load_model(run_id)

        tmin = run.data.params.get("tmin")
        tmax = run.data.params.get("tmax")
        if tmin in (None, "", "auto", "None"):
            tmin = None
        if tmax in (None, "", "auto", "None"):
            tmax = None

        sim = model.simulate(tmin=tmin, tmax=tmax)
        obs = model.observations(tmin=tmin, tmax=tmax)

        results.append({
            "run_id": run_id,
            "name": run.info.run_name or run_id[:8],
            "code_bss": run.data.tags.get("station_id", ""),
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "observed": obs,
            "simulated": sim,
        })

    return results
