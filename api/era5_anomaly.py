"""Pure helpers for ERA5 temperature-anomaly window maths (no DB/Streamlit)."""
from __future__ import annotations

from datetime import date


def window_end_months(end_month: int, n: int) -> list[int]:
    """The n calendar months (1..12) ending at end_month, chronological, wrapping."""
    months = [((end_month - i - 1) % 12) + 1 for i in range(n)]
    return months[::-1]


def add_months(d: date, k: int) -> date:
    """First day of the month k months from d (k may be negative)."""
    total = (d.year * 12 + (d.month - 1)) + k
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)
