"""Tests for cross-correlation."""
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
from dashboard.utils.pastas.cross_correlation import compute_cross_correlation


def test_detects_lag():
    model = MagicMock()
    model.get_step_response.side_effect = Exception("no step response")
    engine = MagicMock(spec=Engine)

    # Create synthetic data with 2-month lag
    n = 120
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)
    precip = 60 + 20 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 5, n)
    piezo = np.roll(precip, 2) * 0.1 + 10  # 2-month lagged response

    df = pd.DataFrame({"mois": dates, "niveau_moyen": piezo, "precipitation_totale": precip})

    with patch.object(engine, 'connect') as mock_conn:
        mock_ctx = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pandas.read_sql", return_value=df):
            result = compute_cross_correlation(model, "TEST/001", "2010-01-01", "2019-12-31", engine)

    assert len(result["lags_months"]) > 0
    assert result["max_lag_months"] is not None
    assert result["max_correlation"] > 0.3
