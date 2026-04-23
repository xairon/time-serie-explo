"""Tests for input quality anomaly detection."""
from __future__ import annotations

import sys
import types
import unittest.mock as mock
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: inject minimal sqlalchemy stubs if the real package is absent.
# dashboard/utils/pastas/input_quality.py imports `from sqlalchemy import text`
# at the top level.  On the bare dev host the full dependency set is not
# installed, so we provide lightweight stand-ins.
# ---------------------------------------------------------------------------
_SQLALCHEMY_REAL = True
try:
    import sqlalchemy  # noqa: F401
    from sqlalchemy.engine import Engine  # noqa: F401
except ImportError:
    _SQLALCHEMY_REAL = False

    _sa_stub = types.ModuleType("sqlalchemy")
    _sa_stub.text = lambda s: mock.MagicMock()
    sys.modules["sqlalchemy"] = _sa_stub

    _engine_mod = types.ModuleType("sqlalchemy.engine")

    class _EngineStub:
        """Minimal stand-in for sqlalchemy.engine.Engine."""
        def connect(self):  # pragma: no cover
            pass

    Engine = _EngineStub  # type: ignore[assignment,misc]
    _engine_mod.Engine = Engine
    sys.modules["sqlalchemy.engine"] = _engine_mod

import numpy as np
import pandas as pd
from dashboard.utils.pastas.input_quality import detect_input_anomalies


def _make_engine():
    """Return a MagicMock engine with a working spec."""
    if _SQLALCHEMY_REAL:
        return MagicMock(spec=Engine)
    # When sqlalchemy is stubbed, use an unspecced mock so patch.object works.
    return MagicMock()


def test_detects_anomalies_in_synthetic_data():
    engine = _make_engine()

    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    df = pd.DataFrame({
        "mois": dates,
        "niveau_moyen": 10 + np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 0.1, n),
        "precipitation_totale": 60 + 20 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 5, n),
        "temperature_moyenne": 12 + 8 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 0.5, n),
        "evaporation_moyenne": 2 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 0.2, n),
    })
    # Inject clear anomalies
    df.loc[50, "niveau_moyen"] = 20.0  # extreme spike
    df.loc[80, "precipitation_totale"] = 300.0  # extreme precip

    with patch.object(engine, "connect") as mock_conn:
        mock_ctx = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        with patch("pandas.read_sql", return_value=df):
            result = detect_input_anomalies("TEST/001", engine)

    assert result["n_total"] > 0
    assert result["n_flagged"] > 0
    assert all("month" in f for f in result["flagged"])
    assert all("score" in f for f in result["flagged"])
    assert all("reason" in f for f in result["flagged"])


def test_short_series_returns_empty():
    engine = _make_engine()
    df = pd.DataFrame({
        "mois": pd.date_range("2020-01-01", periods=10, freq="MS"),
        "niveau_moyen": np.ones(10),
    })

    with patch.object(engine, "connect") as mock_conn:
        mock_ctx = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        with patch("pandas.read_sql", return_value=df):
            result = detect_input_anomalies("TEST/001", engine)

    assert result["n_flagged"] == 0
    assert result["flagged"] == []
