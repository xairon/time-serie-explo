"""Unit tests for QmnJ (L/s → m³/s) conversion in observatory_hydro router."""
from api.routers.observatory_hydro import (
    _FLOW_COLS_DIM,
    _QMNJ_MAX_VALID,
    _QMNJ_MIN_VALID,
    _convert_qmnj_row,
    _qmnj_to_m3_s,
)
from dashboard.utils.reference import value_to_zscore


def test_ssfi_value_and_grid_must_share_units():
    """Regression: SSFI was constant over time because the series value was
    converted to m³/s but interpolated against an L/s reference grid.

    The warehouse stores the hydro quantile grid in raw QmnJ (L/s), so the
    z-score must be computed with value and grid in the *same* unit.
    """
    # Reference grid in warehouse units (L/s), ascending ~400..1576
    grid_ls = [400.0 + i * 12 for i in range(99)]
    val_ls = 1500.0  # high within the grid → z should be clearly positive

    z_raw = value_to_zscore(val_ls, grid_ls)
    z_converted = value_to_zscore(val_ls / 1000.0, [g / 1000.0 for g in grid_ls])

    # Consistent units (both L/s, or both m³/s) give an identical, sensible z
    assert z_raw == z_converted
    assert z_raw > 1.0

    # The bug: value in m³/s vs grid in L/s floors every month to the same value
    z_bug = value_to_zscore(val_ls / 1000.0, grid_ls)
    assert z_bug < -2.0
    # ...and it is constant regardless of the input value (the reported symptom)
    assert z_bug == value_to_zscore(900.0 / 1000.0, grid_ls)


def test_qmnj_none_stays_none():
    assert _qmnj_to_m3_s(None) is None


def test_qmnj_converts_typical_flow():
    # Rhône à Tarascon: 1.7M L/s stored → 1700 m³/s displayed
    assert _qmnj_to_m3_s(1_696_425) == 1696.425
    assert _qmnj_to_m3_s(0) == 0.0
    assert _qmnj_to_m3_s(50_000) == 50.0


def test_qmnj_accepts_string_or_decimal():
    # SQLAlchemy may return Decimal; conversion must coerce via float()
    assert _qmnj_to_m3_s("1000") == 1.0


def test_qmnj_filters_high_sentinel():
    assert _qmnj_to_m3_s(999_999_872) is None
    assert _qmnj_to_m3_s(_QMNJ_MAX_VALID) is None
    assert _qmnj_to_m3_s(_QMNJ_MAX_VALID - 1) is not None


def test_qmnj_filters_low_sentinel():
    assert _qmnj_to_m3_s(-4_000_000) is None
    assert _qmnj_to_m3_s(-150_100) is None
    assert _qmnj_to_m3_s(-10_000) is None
    assert _qmnj_to_m3_s(_QMNJ_MIN_VALID) is None
    assert _qmnj_to_m3_s(_QMNJ_MIN_VALID + 1) is not None


def test_qmnj_keeps_small_negatives():
    # Tidal/sensor noise near zero: keep, do not drop
    assert _qmnj_to_m3_s(-1) == -0.001
    assert _qmnj_to_m3_s(-851) == -0.851


def test_convert_row_handles_missing_and_present_columns():
    row = {
        "resultat_moyen_global": 50_000,
        "resultat_min_global": None,
        "resultat_max_global": 999_999_872,  # sentinel
        "code_station": "X123",  # unrelated, untouched
    }
    out = _convert_qmnj_row(row, _FLOW_COLS_DIM)
    assert out is row  # in-place mutation contract
    assert out["resultat_moyen_global"] == 50.0
    assert out["resultat_min_global"] is None
    assert out["resultat_max_global"] is None  # filtered
    assert out["code_station"] == "X123"
    # Columns absent from row are silently skipped (no KeyError)
    assert "resultat_stddev_global" not in out
