"""Unit tests for classify_index — the shared McKee/WMO 7-class classification used by
the ERA5 SPI/STI grid endpoints and the per-station SPI endpoints.

Was ``test_era5_anomaly.py``: that file also covered ``window_end_months``,
``add_months``, ``latest_complete_month``, ``compute_anomalies`` and
``compute_precip_anomalies`` — the pure helpers behind the on-the-fly ``/anomaly``
climatology scan. That endpoint (and the frontend's phantom "anomaly" overlay
variable it served) was removed in Task C1 (Lot 2), so those helpers were deleted
from api/era5_anomaly.py as dead code; only the classify_index tests survive, moved
here under a name that no longer references anomaly.
"""
from api.era5_anomaly import classify_index


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
