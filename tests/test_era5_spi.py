"""Unit tests for compute_spi_grid — gamma-based ERA5 Standardized Precipitation Index."""
import numpy as np
from scipy import stats

from api.era5_anomaly import compute_spi_grid, classify_index


def _fit(sums):
    a, loc, scale = stats.gamma.fit(np.asarray(sums, dtype=float), floc=0)
    return {"a": a, "loc": loc, "scale": scale, "n_years": len(sums)}


def test_median_accumulation_is_near_zero_sigma():
    # Reference: a spread of 30 plausible 3-month precip totals (mm).
    ref_sums = [120, 90, 150, 110, 200, 80, 130, 170, 100, 140,
                160, 95, 145, 115, 185, 105, 125, 155, 135, 175,
                85, 165, 118, 132, 148, 108, 128, 158, 138, 178]
    ref = _fit(ref_sums)
    ref["latitude"] = 45.0
    ref["longitude"] = 2.0
    median = float(np.median(ref_sums))
    rows = [{"latitude": 45.0, "longitude": 2.0, "window_sum": median, "n_months": 3}]
    out = compute_spi_grid(rows, [ref], 3)
    assert len(out) == 1
    assert abs(out[0]["spi"]) < 0.4  # near the median → near 0 sigma
    assert out[0]["index_class"] == "NORMAL"


def test_low_accumulation_is_dry_negative_spi():
    ref_sums = [120, 90, 150, 110, 200, 80, 130, 170, 100, 140,
                160, 95, 145, 115, 185, 105, 125, 155, 135, 175]
    ref = _fit(ref_sums); ref["latitude"] = 45.0; ref["longitude"] = 2.0
    rows = [{"latitude": 45.0, "longitude": 2.0, "window_sum": 30.0, "n_months": 3}]
    out = compute_spi_grid(rows, [ref], 3)
    assert out[0]["spi"] < -0.84  # well below normal → drought class
    assert out[0]["index_class"] in ("BAS", "TRES_BAS", "EXTREMEMENT_BAS")


def test_drops_incomplete_window_and_nonpositive_and_missing_ref():
    ref = _fit([120, 90, 150, 110, 200, 80, 130, 170, 100, 140])
    ref["latitude"] = 45.0; ref["longitude"] = 2.0
    rows = [
        {"latitude": 45.0, "longitude": 2.0, "window_sum": 100.0, "n_months": 2},   # incomplete
        {"latitude": 45.0, "longitude": 2.0, "window_sum": 0.0, "n_months": 3},      # non-positive
        {"latitude": 45.0, "longitude": 2.0, "window_sum": None, "n_months": 3},     # null
        {"latitude": 99.0, "longitude": 9.0, "window_sum": 100.0, "n_months": 3},    # no ref
    ]
    assert compute_spi_grid(rows, [ref], 3) == []


def test_uses_same_mckee_thresholds_as_classify_index():
    # SPI shares the STI/IPS 7-class thresholds.
    assert classify_index(-1.9) == "EXTREMEMENT_BAS"
    assert classify_index(0.0) == "NORMAL"
    assert classify_index(1.9) == "EXTREMEMENT_HAUT"
