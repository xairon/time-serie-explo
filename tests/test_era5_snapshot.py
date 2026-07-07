"""Unit tests for _rows_to_snapshot — the /snapshot duplicate-coordinate merge.

Regression: gold.era5_grid is a view over bronze.era5_france_timeseries, which
was never purged — unrounded float coordinates still exist there for
2026-01-21→2026-05-01 (~1.16M rows; only silver was purged/re-staged by the
upstream remediation). Commit 8231128 dropped the weighted AVG merge from
/snapshot on the (incorrect, at the time) assumption that upstream coordinates
were always clean. Without the merge, two float variants of the same 0.1° cell
(e.g. 47.09999999999994 and 47.1) surface as two separate cells instead of one
averaged cell — this test locks in the fix (no DB touched, follows the repo
convention — see test_era5_spi.py).
"""
from api.routers.observatory_era5 import _rows_to_snapshot


def test_merges_float_coordinate_doublons_into_one_cell():
    rows = [
        {
            "latitude": 47.09999999999994,
            "longitude": 2.0,
            "temperature_2m": 10.0,
            "total_precipitation": 1.0,
            "potential_evaporation": -1.0,
        },
        {
            "latitude": 47.1,
            "longitude": 2.0,
            "temperature_2m": 12.0,
            "total_precipitation": 3.0,
            "potential_evaporation": -3.0,
        },
    ]
    out = _rows_to_snapshot(rows)
    assert len(out) == 1
    cell = out[0]
    assert cell["latitude"] == 47.1
    assert cell["longitude"] == 2.0
    assert cell["temperature_2m"] == 11.0
    assert cell["total_precipitation"] == 2.0
    assert cell["potential_evaporation"] == -2.0


def test_distinct_cells_stay_separate():
    rows = [
        {"latitude": 45.0, "longitude": 1.0, "temperature_2m": 5.0,
         "total_precipitation": 0.0, "potential_evaporation": 0.0},
        {"latitude": 46.0, "longitude": 2.0, "temperature_2m": 6.0,
         "total_precipitation": 1.0, "potential_evaporation": -1.0},
    ]
    out = _rows_to_snapshot(rows)
    assert len(out) == 2
    keys = {(r["latitude"], r["longitude"]) for r in out}
    assert keys == {(45.0, 1.0), (46.0, 2.0)}


def test_null_variables_ignored_in_average():
    # One variant has a NULL temperature — average must ignore it (matches SQL AVG()).
    rows = [
        {"latitude": 47.09999999999994, "longitude": 2.0, "temperature_2m": None,
         "total_precipitation": 1.0, "potential_evaporation": -1.0},
        {"latitude": 47.1, "longitude": 2.0, "temperature_2m": 12.0,
         "total_precipitation": 3.0, "potential_evaporation": -3.0},
    ]
    out = _rows_to_snapshot(rows)
    assert len(out) == 1
    assert out[0]["temperature_2m"] == 12.0
    assert out[0]["total_precipitation"] == 2.0


def test_all_null_variable_stays_none():
    rows = [
        {"latitude": 45.0, "longitude": 1.0, "temperature_2m": None,
         "total_precipitation": None, "potential_evaporation": 2.0},
    ]
    out = _rows_to_snapshot(rows)
    assert out[0]["temperature_2m"] is None
    assert out[0]["total_precipitation"] is None
    assert out[0]["potential_evaporation"] == 2.0
