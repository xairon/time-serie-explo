"""Unit tests for ``station_spi_rows`` — the piezo/hydro station SPI formatter.

Task A2: ``/stations/{code}/spi`` (piezo + hydro) no longer fits a per-station
gamma distribution on the fly; it reads the precomputed SPI for the station's
mapped ERA5 grid cell (``gold.fct_era5_indices_grid``, via
``int_station_era5_mapping`` / ``int_hydro_station_era5_mapping``) and the
monthly precipitation total (``gold.fct_era5_monthly_grid``). This only formats
and classifies — no statistics are (re)computed here.
"""
from api.era5_anomaly import station_spi_rows


def test_station_spi_rows_basic_shape():
    rows = [{"mois": "2026-05-01", "value": 88.51436140772421, "spi": -1.482}]
    out = station_spi_rows(rows)
    assert out == [{"mois": "2026-05-01", "value": 88.5144, "spi": -1.482, "classification": "TRES_BAS"}]


def test_station_spi_rows_null_spi_yields_unknown_classification():
    # A month present in the monthly mart but missing from the indices mart
    # (LEFT JOIN miss) must still surface the precipitation value, spi=None.
    rows = [{"mois": "1950-01-01", "value": 40.0, "spi": None}]
    out = station_spi_rows(rows)
    assert out == [{"mois": "1950-01-01", "value": 40.0, "spi": None, "classification": "UNKNOWN"}]


def test_station_spi_rows_null_value_preserved():
    rows = [{"mois": "2026-01-01", "value": None, "spi": 0.5}]
    out = station_spi_rows(rows)
    assert out[0]["value"] is None
    assert out[0]["spi"] == 0.5
    assert out[0]["classification"] == "NORMAL"


def test_station_spi_rows_ordering_and_multiple_months():
    rows = [
        {"mois": "2026-01-01", "value": 10.0, "spi": 2.0},
        {"mois": "2026-02-01", "value": 5.0, "spi": -2.0},
    ]
    out = station_spi_rows(rows)
    assert [r["mois"] for r in out] == ["2026-01-01", "2026-02-01"]
    assert out[0]["classification"] == "EXTREMEMENT_HAUT"
    assert out[1]["classification"] == "EXTREMEMENT_BAS"


def test_station_spi_rows_empty_input():
    assert station_spi_rows([]) == []
