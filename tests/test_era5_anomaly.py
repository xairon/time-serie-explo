from datetime import date
from decimal import Decimal
import pytest
from api.era5_anomaly import window_end_months, add_months, latest_complete_month, compute_anomalies


def test_window_end_months_no_wrap():
    assert window_end_months(3, 3) == [1, 2, 3]
    assert window_end_months(12, 1) == [12]
    assert window_end_months(6, 6) == [1, 2, 3, 4, 5, 6]


def test_window_end_months_wraps_year_boundary():
    assert window_end_months(1, 3) == [11, 12, 1]
    assert window_end_months(2, 12) == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]


def test_add_months_forward_and_back():
    assert add_months(date(2024, 3, 15), 1) == date(2024, 4, 1)
    assert add_months(date(2024, 1, 10), -1) == date(2023, 12, 1)
    assert add_months(date(2024, 3, 1), -2) == date(2024, 1, 1)
    assert add_months(date(2024, 12, 1), 1) == date(2025, 1, 1)


def test_latest_complete_month_last_day():
    # March 31 = last day of March → March is complete
    assert latest_complete_month(date(2024, 3, 31)) == date(2024, 3, 1)


def test_latest_complete_month_mid_month():
    # March 15 → March incomplete → previous month = February
    assert latest_complete_month(date(2024, 3, 15)) == date(2024, 2, 1)


def test_latest_complete_month_jan_boundary():
    # Jan 1 (incomplete) → previous month = December of prior year
    assert latest_complete_month(date(2024, 1, 1)) == date(2023, 12, 1)


# --- compute_anomalies ---

def _make_clim(lat, lon, months_means):
    """Helper: build climatology rows for a given cell across multiple months."""
    return [{"latitude": lat, "longitude": lon, "mo": mo, "mean_c": mean} for mo, mean in months_means]


def test_compute_anomalies_incomplete_window_dropped():
    """A cell with n_months < window is excluded from output."""
    window = 3
    months = window_end_months(3, window)  # [1, 2, 3]
    clim = _make_clim(48.0, 2.0, [(1, 10.0), (2, 12.0), (3, 14.0)])
    # n_months = 2, which is < window=3
    rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 11.0, "n_months": 2}]
    result = compute_anomalies(rows, clim, months, window)
    assert result == []


def test_compute_anomalies_incomplete_normals_dropped():
    """A cell missing one ending-month climatology entry is excluded."""
    window = 3
    months = window_end_months(3, window)  # [1, 2, 3]
    # Only 2 months of climatology provided (missing month 3)
    clim = _make_clim(48.0, 2.0, [(1, 10.0), (2, 12.0)])
    rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 11.0, "n_months": 3}]
    result = compute_anomalies(rows, clim, months, window)
    assert result == []


def test_compute_anomalies_valid_cell_correct_anomaly():
    """A valid cell yields anomaly_c = window_mean − mean(ending-month normals)."""
    window = 3
    months = window_end_months(3, window)  # [1, 2, 3]
    # normals: 10, 12, 14 → mean = 12.0
    clim = _make_clim(48.0, 2.0, [(1, 10.0), (2, 12.0), (3, 14.0)])
    rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 15.0, "n_months": 3}]
    result = compute_anomalies(rows, clim, months, window)
    assert len(result) == 1
    assert result[0]["latitude"] == pytest.approx(48.0)
    assert result[0]["longitude"] == pytest.approx(2.0)
    assert result[0]["anomaly_c"] == pytest.approx(15.0 - 12.0)


def test_compute_anomalies_decimal_string_inputs():
    """Mixed Decimal/str latitude, longitude, mean_c still join and produce correct output (cached-string path)."""
    window = 3
    months = window_end_months(3, window)  # [1, 2, 3]
    # Simulate DB returning Decimal for lat/lon/mean_c, and str for lat/lon in rows
    clim = [
        {"latitude": Decimal("48.0"), "longitude": Decimal("2.0"), "mo": 1, "mean_c": Decimal("10.0")},
        {"latitude": Decimal("48.0"), "longitude": Decimal("2.0"), "mo": 2, "mean_c": Decimal("12.0")},
        {"latitude": Decimal("48.0"), "longitude": Decimal("2.0"), "mo": 3, "mean_c": Decimal("14.0")},
    ]
    rows = [
        {"latitude": "48.0", "longitude": "2.0", "window_mean": "15.0", "n_months": "3"}
    ]
    result = compute_anomalies(rows, clim, months, window)
    assert len(result) == 1
    assert result[0]["anomaly_c"] == pytest.approx(3.0)
