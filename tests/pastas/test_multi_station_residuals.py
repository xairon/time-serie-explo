"""Tests for multi-station residual comparison."""
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
from dashboard.utils.pastas.multi_station_residuals import compute_regional_residuals


def test_returns_structure_with_no_siblings():
    model = MagicMock()
    dates = pd.date_range("2015-01-01", periods=365*3, freq="D")
    rng = np.random.default_rng(42)
    model.residuals.return_value = pd.Series(rng.normal(0, 0.1, 365*3), index=dates)

    engine = MagicMock(spec=Engine)
    mock_conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.mappings.return_value.first.return_value = None

    result = compute_regional_residuals(model, "TEST/001", "2015-01-01", "2017-12-31", engine)
    assert len(result["model_residual_zscore"]) > 0
    assert result["siblings"] == []
    assert result["correlation_with_siblings"] is None
