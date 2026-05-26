"""Unit tests for QmnJ (L/s → m³/s) conversion in observatory_hydro router."""
from api.routers.observatory_hydro import (
    _FLOW_COLS_DIM,
    _QMNJ_MAX_VALID,
    _QMNJ_MIN_VALID,
    _convert_qmnj_row,
    _qmnj_to_m3_s,
)


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
