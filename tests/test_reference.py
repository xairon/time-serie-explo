import numpy as np
from dashboard.utils.reference import series_zscores, class_bounds_ngf, CLASS_CUTOFF_PCTL

# synthetic ascending grid 1..99 -> values 1.0..99.0
_GRID = [float(p) for p in range(1, 100)]


def test_class_bounds_ngf_matches_cutoffs():
    bounds = class_bounds_ngf(_GRID)
    assert len(bounds) == len(CLASS_CUTOFF_PCTL)
    # ascending
    assert all(bounds[i] <= bounds[i + 1] for i in range(len(bounds) - 1))
    # ~ equals the cutoff percentile value on a linear grid
    assert abs(bounds[0] - CLASS_CUTOFF_PCTL[0]) < 0.5


def test_series_zscores_sign():
    grid_by_month = {m: _GRID for m in range(1, 13)}
    z = series_zscores([("2020-06-15", 5.0), ("2020-07-15", 95.0)], grid_by_month)
    assert z[0] < 0 < z[1]


def test_series_zscores_handles_missing_month_grid():
    z = series_zscores([("2020-06-15", 50.0)], {6: None})
    assert z == [None]
