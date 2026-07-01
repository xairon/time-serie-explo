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


def test_mom_gamma_matches_moments_and_guards_degenerate():
    from api.routers.observatory_era5 import _mom_gamma, SPI_MIN_REF_YEARS
    vals = [float(x) for x in range(50, 50 + SPI_MIN_REF_YEARS + 5)]
    g = _mom_gamma(vals)
    assert g is not None
    a, loc, scale = g
    assert loc == 0.0 and a > 0 and scale > 0
    # method of moments: mean == a*scale (reconstruct the sample mean)
    mean = sum(vals) / len(vals)
    assert abs(a * scale - mean) < 1e-6
    # too few points, or zero variance → None
    assert _mom_gamma([100.0] * (SPI_MIN_REF_YEARS - 1)) is None
    assert _mom_gamma([100.0] * (SPI_MIN_REF_YEARS + 3)) is None  # var == 0


def _synthetic_by_cell(end_months=(3, 4, 5)):
    # one cell with 30 complete reference years of monthly values
    ym = {}
    for i, yr in enumerate(range(1991, 2021)):
        for mo in end_months:
            ym[(yr, mo)] = 10.0 + mo + i * 0.1
    return {(45.0, 2.0): ym}


def test_sti_ref_from_cells_produces_mean_std_over_30_years():
    from api.routers.observatory_era5 import _sti_ref_from_cells
    out = _sti_ref_from_cells(_synthetic_by_cell(), window=3, end_month=5)
    assert len(out) == 1
    r = out[0]
    assert r["n_years"] == 30 and r["std"] > 0
    assert set(r) == {"latitude", "longitude", "mean", "std", "n_years"}


def test_spi_ref_from_cells_yields_gamma_params():
    from api.routers.observatory_era5 import _spi_ref_from_cells
    out = _spi_ref_from_cells(_synthetic_by_cell(), window=3, end_month=5)
    assert len(out) == 1
    r = out[0]
    assert r["a"] > 0 and r["scale"] > 0 and r["loc"] == 0.0 and r["n_years"] == 30


def test_bulk_sti_ref_matches_per_key_algorithm():
    # The bulk helper must reproduce the per-key reference maths for a shared cell set.
    from api.routers.observatory_era5 import _sti_ref_from_cells
    by_cell = _synthetic_by_cell(end_months=(1, 2, 3, 4, 5, 6))
    # window=3 ending May uses only Mar/Apr/May → same as a cell holding just those
    full = _sti_ref_from_cells(by_cell, window=3, end_month=5)[0]
    trimmed = _sti_ref_from_cells(_synthetic_by_cell((3, 4, 5)), window=3, end_month=5)[0]
    assert abs(full["mean"] - trimmed["mean"]) < 1e-9
    assert abs(full["std"] - trimmed["std"]) < 1e-9
