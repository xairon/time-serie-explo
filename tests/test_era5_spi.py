"""Unit tests for the mart-based ERA5 SPI/STI grid endpoints (Task A2).

/spi and /sti no longer fit a gamma / compute a z-score in the API — they read
the precomputed ``gold.fct_era5_indices_grid`` mart and only format + classify
the result. These tests cover the pure formatting helpers (``_rows_to_spi``,
``_rows_to_sti``) and the month-resolution helper (``_resolve_indices_month``),
following the repo convention (no DB touched — see test_era5_resolve_month_start.py).
"""
from datetime import date

from api.era5_anomaly import classify_index
from api.routers.observatory_era5 import _resolve_indices_month, _rows_to_spi, _rows_to_sti


# --- _rows_to_spi ---

def test_rows_to_spi_rounds_and_classifies():
    rows = [{"latitude": 45.0, "longitude": 2.0, "spi": -1.23456}]
    out = _rows_to_spi(rows)
    assert out == [{"latitude": 45.0, "longitude": 2.0, "spi": -1.235, "index_class": "BAS"}]


def test_rows_to_spi_drops_null_spi():
    rows = [
        {"latitude": 45.0, "longitude": 2.0, "spi": None},
        {"latitude": 46.0, "longitude": 3.0, "spi": 0.0},
    ]
    out = _rows_to_spi(rows)
    assert len(out) == 1
    assert out[0]["latitude"] == 46.0


def test_rows_to_spi_multiple_cells():
    rows = [
        {"latitude": 45.0, "longitude": 2.0, "spi": 2.0},
        {"latitude": 46.0, "longitude": 3.0, "spi": -2.0},
    ]
    out = _rows_to_spi(rows)
    classes = {r["latitude"]: r["index_class"] for r in out}
    assert classes[45.0] == "EXTREMEMENT_HAUT"
    assert classes[46.0] == "EXTREMEMENT_BAS"


# --- _rows_to_sti ---

def test_rows_to_sti_unrounded_and_classified():
    rows = [{"latitude": 48.0, "longitude": 2.0, "sti": 2.0}]
    out = _rows_to_sti(rows)
    assert out == [{"latitude": 48.0, "longitude": 2.0, "sti": 2.0, "index_class": "EXTREMEMENT_HAUT"}]


def test_rows_to_sti_drops_null_sti():
    rows = [{"latitude": 48.0, "longitude": 2.0, "sti": None}]
    assert _rows_to_sti(rows) == []


def test_rows_to_sti_preserves_full_precision():
    # Unlike _rows_to_spi, sti is NOT rounded (matches the historical compute_sti output).
    rows = [{"latitude": 48.0, "longitude": 2.0, "sti": 1.23456789}]
    out = _rows_to_sti(rows)
    assert out[0]["sti"] == 1.23456789


def test_uses_same_mckee_thresholds_as_classify_index():
    # SPI/STI share the IPS 7-class thresholds.
    assert classify_index(-1.9) == "EXTREMEMENT_BAS"
    assert classify_index(0.0) == "NORMAL"
    assert classify_index(1.9) == "EXTREMEMENT_HAUT"


# --- _resolve_indices_month ---

class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConn:
    """Returns a fixed MAX(month) for the single query the helper runs."""
    def __init__(self, max_month):
        self._max_month = max_month

    def execute(self, *_args, **_kwargs):
        return _FakeScalar(self._max_month)


def test_resolve_indices_month_empty_mart_returns_none():
    assert _resolve_indices_month(_FakeConn(None), None, 3) is None
    assert _resolve_indices_month(_FakeConn(None), date(2026, 5, 1), 3) is None


def test_resolve_indices_month_no_date_uses_latest():
    conn = _FakeConn(date(2026, 6, 1))
    assert _resolve_indices_month(conn, None, 3) == date(2026, 6, 1)


def test_resolve_indices_month_supplied_past_month_preserved():
    conn = _FakeConn(date(2026, 6, 1))
    assert _resolve_indices_month(conn, date(2024, 3, 10), 3) == date(2024, 3, 1)


def test_resolve_indices_month_supplied_future_month_clamped():
    conn = _FakeConn(date(2026, 6, 1))
    assert _resolve_indices_month(conn, date(2026, 12, 1), 3) == date(2026, 6, 1)
