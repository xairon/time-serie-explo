from datetime import date
from decimal import Decimal
import math
import pytest
from api.era5_anomaly import window_end_months, add_months, latest_complete_month, compute_anomalies, compute_precip_anomalies, classify_index, compute_sti


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


# --- compute_precip_anomalies ---

def test_precip_anomaly_percent_basic():
    # one cell, window=3 ending month with normals [10,20,30]=60; observed sum=90 → +50%
    clim = [
        {"latitude": 48.0, "longitude": 2.0, "mo": 1, "mean_sum": 10.0},
        {"latitude": 48.0, "longitude": 2.0, "mo": 2, "mean_sum": 20.0},
        {"latitude": 48.0, "longitude": 2.0, "mo": 3, "mean_sum": 30.0},
    ]
    rows = [{"latitude": 48.0, "longitude": 2.0, "precip_sum": 90.0, "n_months": 3}]
    out = compute_precip_anomalies(rows, clim, [1, 2, 3], 3)
    assert len(out) == 1
    assert out[0]["anomaly"] == 50.0  # (90-60)/60*100


def test_precip_anomaly_drops_incomplete_and_zero_normal():
    clim = [{"latitude": 1.0, "longitude": 1.0, "mo": 1, "mean_sum": 0.0}]
    rows = [{"latitude": 1.0, "longitude": 1.0, "precip_sum": 5.0, "n_months": 1}]
    assert compute_precip_anomalies(rows, clim, [1], 1) == []  # normal<=0 dropped
    rows2 = [{"latitude": 1.0, "longitude": 1.0, "precip_sum": 5.0, "n_months": 0}]
    assert compute_precip_anomalies(rows2, [{"latitude":1.0,"longitude":1.0,"mo":1,"mean_sum":3.0}], [1], 1) == []  # n_months<window


# --- classify_index ---

def test_classify_index_none_returns_unknown():
    assert classify_index(None) == "UNKNOWN"


def test_classify_index_nan_returns_unknown():
    assert classify_index(float("nan")) == "UNKNOWN"


def test_classify_index_boundary_values():
    # Boundaries: lo <= z < hi, so the boundary value enters the UPPER class
    # (-inf, -1.75, EXTREMEMENT_BAS): values < -1.75
    # (-1.75, -1.28, TRES_BAS): -1.75 <= z < -1.28
    assert classify_index(-2.0) == "EXTREMEMENT_BAS"
    assert classify_index(-1.75) == "TRES_BAS"   # exactly at boundary → enters TRES_BAS
    assert classify_index(-1.5) == "TRES_BAS"
    assert classify_index(-1.28) == "BAS"         # exactly at boundary → enters BAS
    assert classify_index(-1.0) == "BAS"
    assert classify_index(-0.84) == "NORMAL"      # exactly at boundary → enters NORMAL
    assert classify_index(0.0) == "NORMAL"
    assert classify_index(0.84) == "HAUT"         # exactly at boundary → enters HAUT
    assert classify_index(1.0) == "HAUT"
    assert classify_index(1.28) == "TRES_HAUT"    # exactly at boundary → enters TRES_HAUT
    assert classify_index(1.5) == "TRES_HAUT"
    assert classify_index(1.75) == "EXTREMEMENT_HAUT"  # exactly at boundary → enters EXTREMEMENT_HAUT
    assert classify_index(2.0) == "EXTREMEMENT_HAUT"


# --- compute_sti ---

def test_compute_sti_basic_z2_extremement_haut():
    """mean=10, std=2, obs=14 → z=2.0 → EXTREMEMENT_HAUT."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 14.0, "n_months": 3}]
    reference = [{"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": 2.0, "n_years": 30}]
    result = compute_sti(window_rows, reference, 3)
    assert len(result) == 1
    assert result[0]["latitude"] == pytest.approx(48.0)
    assert result[0]["longitude"] == pytest.approx(2.0)
    assert result[0]["sti"] == pytest.approx(2.0)
    assert result[0]["index_class"] == "EXTREMEMENT_HAUT"


def test_compute_sti_std_zero_dropped():
    """std=0 → cell must be dropped."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 14.0, "n_months": 3}]
    reference = [{"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": 0.0, "n_years": 30}]
    assert compute_sti(window_rows, reference, 3) == []


def test_compute_sti_std_negative_dropped():
    """std<0 (should not happen but guard) → dropped."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 14.0, "n_months": 3}]
    reference = [{"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": -1.0, "n_years": 30}]
    assert compute_sti(window_rows, reference, 3) == []


def test_compute_sti_n_months_less_than_window_dropped():
    """n_months < window → cell must be dropped."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 14.0, "n_months": 2}]
    reference = [{"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": 2.0, "n_years": 30}]
    assert compute_sti(window_rows, reference, 3) == []


def test_compute_sti_missing_reference_dropped():
    """No reference entry for the cell → dropped."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 14.0, "n_months": 3}]
    assert compute_sti(window_rows, [], 3) == []


def test_compute_sti_window_mean_none_dropped():
    """window_mean is None → dropped."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": None, "n_months": 3}]
    reference = [{"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": 2.0, "n_years": 30}]
    assert compute_sti(window_rows, reference, 3) == []


def test_compute_sti_negative_z():
    """mean=10, std=2, obs=6 → z=-2.0 → EXTREMEMENT_BAS."""
    window_rows = [{"latitude": 48.0, "longitude": 2.0, "window_mean": 6.0, "n_months": 1}]
    reference = [{"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": 2.0, "n_years": 30}]
    result = compute_sti(window_rows, reference, 1)
    assert len(result) == 1
    assert result[0]["sti"] == pytest.approx(-2.0)
    assert result[0]["index_class"] == "EXTREMEMENT_BAS"


def test_compute_sti_multiple_cells():
    """Two cells, both valid, produce two results."""
    window_rows = [
        {"latitude": 48.0, "longitude": 2.0, "window_mean": 14.0, "n_months": 3},
        {"latitude": 49.0, "longitude": 3.0, "window_mean": 9.0, "n_months": 3},
    ]
    reference = [
        {"latitude": 48.0, "longitude": 2.0, "mean": 10.0, "std": 2.0, "n_years": 30},
        {"latitude": 49.0, "longitude": 3.0, "mean": 10.0, "std": 2.0, "n_years": 30},
    ]
    result = compute_sti(window_rows, reference, 3)
    assert len(result) == 2
    lats = {r["latitude"] for r in result}
    assert 48.0 in lats and 49.0 in lats
