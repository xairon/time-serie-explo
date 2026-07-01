"""Unit tests for _resolve_month_start — the STI/anomaly ending-month clamp.

Regression: when a date was supplied the router used it verbatim, so a partial
in-progress (or future) month was scored against a complete-month reference,
biasing the index on the most-visible month. It must clamp to the latest
complete month.
"""
from datetime import date

from api.routers.observatory_era5 import _resolve_month_start


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConn:
    """Returns a fixed max("time") date for the single query the helper runs."""
    def __init__(self, max_date):
        self._max_date = max_date

    def execute(self, *_args, **_kwargs):
        return _FakeScalar(self._max_date)


def test_empty_table_returns_none():
    assert _resolve_month_start(_FakeConn(None), None) is None
    assert _resolve_month_start(_FakeConn(None), date(2026, 5, 1)) is None


def test_no_date_uses_latest_complete_month():
    # max is mid-June → latest complete month is May
    assert _resolve_month_start(_FakeConn(date(2026, 6, 15)), None) == date(2026, 5, 1)


def test_supplied_partial_month_is_clamped_to_latest_complete():
    # user asks for June (partial) but only mid-June data exists → clamp to May
    conn = _FakeConn(date(2026, 6, 15))
    assert _resolve_month_start(conn, date(2026, 6, 20)) == date(2026, 5, 1)


def test_supplied_future_month_is_clamped():
    conn = _FakeConn(date(2026, 6, 30))  # June complete
    assert _resolve_month_start(conn, date(2026, 12, 1)) == date(2026, 6, 1)


def test_supplied_past_complete_month_is_preserved():
    conn = _FakeConn(date(2026, 6, 15))  # latest complete = May
    assert _resolve_month_start(conn, date(2024, 3, 10)) == date(2024, 3, 1)
