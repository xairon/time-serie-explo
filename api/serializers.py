from __future__ import annotations

import math
from typing import Any

import numpy as np


def serialize_tensor(tensor) -> list:
    """Convert a torch.Tensor or numpy ndarray to a Python list."""
    if hasattr(tensor, "detach"):
        # torch.Tensor
        return tensor.detach().cpu().numpy().tolist()
    if isinstance(tensor, np.ndarray):
        return tensor.tolist()
    return list(tensor)


def serialize_timeseries(ts) -> list[dict]:
    """Convert a Darts TimeSeries to a list of {time, ...values} dicts."""
    df = ts.to_dataframe()
    df.index.name = "time"
    records = df.reset_index().to_dict(orient="records")
    return clean_nans(records)


def serialize_figure(fig) -> dict:
    """Convert a Plotly figure to a JSON-serializable dict."""
    return fig.to_dict()


def clean_nans(d: Any) -> Any:
    """Recursively replace NaN and inf with None in nested structures."""
    if isinstance(d, float):
        if math.isnan(d) or math.isinf(d):
            return None
        return d
    if isinstance(d, dict):
        return {k: clean_nans(v) for k, v in d.items()}
    if isinstance(d, list):
        return [clean_nans(item) for item in d]
    if isinstance(d, np.floating):
        val = float(d)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(d, np.integer):
        return int(d)
    return d
