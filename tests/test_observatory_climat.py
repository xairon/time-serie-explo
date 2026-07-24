"""Unit tests for the observatory_climat router — pure helpers + param validation.

Follows the repo convention (see test_era5_spi.py): exercise the pure Python
transforms directly with plain dicts standing in for SQLAlchemy row mappings,
rather than hitting the real BRGM warehouse.
"""
from datetime import date

import pytest
from fastapi import HTTPException

from api.routers import observatory_climat
from api.routers.observatory_climat import (
    router,
    _parse_month,
    _parse_date,
    _round_cell,
    _build_drought_episodes,
    _build_range,
    _build_daily_points,
    _build_daily_range,
    _DAILY_PRECIP_SQL,
    _merge_point_series,
    _merge_compare_years,
    _MONTHLY_VARIABLES,
    _DAILY_TEMP_VARIABLES,
    WINDOWS,
)


def _rows(key, vals):
    return [{"month": date(2026, m, 1), key: v} for m, v in vals]


def test_router_mounts_all_eleven_climat_paths():
    paths = {r.path for r in router.routes}
    assert paths == {
        "/api/v1/observatory/climat/range",
        "/api/v1/observatory/climat/grid-monthly",
        "/api/v1/observatory/climat/grid-indices",
        "/api/v1/observatory/climat/daily-temp",
        "/api/v1/observatory/climat/daily-temp-range",
        "/api/v1/observatory/climat/daily-precip",
        "/api/v1/observatory/climat/daily-precip-range",
        "/api/v1/observatory/climat/point-series",
        "/api/v1/observatory/climat/point-episodes",
        "/api/v1/observatory/climat/compare-years",
        "/api/v1/observatory/climat/export-point.csv",
    }


class TestParseMonth:
    def test_parses_year_month(self):
        assert _parse_month("2026-06") == date(2026, 6, 1)

    def test_parses_full_iso_date_by_truncating_to_month(self):
        assert _parse_month("2026-06-15") == date(2026, 6, 1)

    def test_invalid_format_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            _parse_month("not-a-month")
        assert exc.value.status_code == 422

    def test_invalid_month_number_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            _parse_month("2026-13")
        assert exc.value.status_code == 422

    def test_empty_string_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            _parse_month("")
        assert exc.value.status_code == 422


class TestRoundCell:
    def test_rounds_to_nearest_tenth(self):
        assert _round_cell(47.42, 0.68) == (47.4, 0.7)

    def test_exact_multiple_is_unchanged(self):
        assert _round_cell(47.4, 0.7) == (47.4, 0.7)


class TestMonthlyVariables:
    def test_all_mart_columns_are_known_and_unique(self):
        # grid-monthly's whitelist must map onto real fct_era5_monthly_grid columns.
        expected_columns = {
            "temperature_moyenne", "temperature_min", "temperature_max",
            "precipitation_totale", "etp_totale", "bilan_hydrique",
        }
        assert set(_MONTHLY_VARIABLES.values()) == expected_columns


class TestParseDate:
    def test_parses_iso_date(self):
        assert _parse_date("2026-06-28") == date(2026, 6, 28)

    def test_invalid_format_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            _parse_date("not-a-date")
        assert exc.value.status_code == 422

    def test_year_month_only_raises_422(self):
        # Unlike _parse_month, /daily-temp requires a full YYYY-MM-DD day.
        with pytest.raises(HTTPException) as exc:
            _parse_date("2026-06")
        assert exc.value.status_code == 422

    def test_invalid_day_number_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            _parse_date("2026-06-31")
        assert exc.value.status_code == 422

    def test_empty_string_raises_422(self):
        with pytest.raises(HTTPException) as exc:
            _parse_date("")
        assert exc.value.status_code == 422


class TestDailyTempVariables:
    def test_maps_tx_tn_tmoy_to_daily_stats_columns(self):
        assert _DAILY_TEMP_VARIABLES == {"tmax": "t2m_max", "tmin": "t2m_min", "tmean": "t2m_mean"}


class TestBuildDailyPoints:
    def test_happy_path_formats_per_cell_value(self):
        rows = [
            {"latitude": 47.4, "longitude": 0.7, "value": 32.5},
            {"latitude": 47.5, "longitude": 0.7, "value": 30.1},
        ]
        out = _build_daily_points(rows)
        assert out == [
            {"latitude": 47.4, "longitude": 0.7, "value": 32.5},
            {"latitude": 47.5, "longitude": 0.7, "value": 30.1},
        ]

    def test_no_rows_for_the_date_yields_empty_list(self):
        # Mirrors /grid-monthly: no data yet for this date is not a 404.
        assert _build_daily_points([]) == []


class TestBuildDailyRange:
    def test_formats_bounds_as_iso_date_strings(self):
        out = _build_daily_range(date(2026, 5, 1), date(2026, 6, 30))
        assert out == {"min_date": "2026-05-01", "max_date": "2026-06-30"}

    def test_none_when_table_is_empty(self):
        out = _build_daily_range(None, None)
        assert out == {"min_date": None, "max_date": None}


class TestBuildRange:
    def test_formats_all_bounds_as_iso_month_strings(self):
        out = _build_range(date(2026, 6, 1), date(2026, 6, 1), date(2026, 7, 1), date(1950, 1, 1))
        assert out == {
            "min_month": "1950-01-01",
            "max_indices_month": "2026-06-01",
            "max_monthly_complete_month": "2026-06-01",
            "max_monthly_month": "2026-07-01",
        }

    def test_indices_and_monthly_maxima_can_differ(self):
        # The real-world case this endpoint exists for: a partial current month
        # already has a raw-monthly row but no SPI/STI yet.
        out = _build_range(date(2026, 6, 1), date(2026, 6, 1), date(2026, 7, 1), date(1950, 1, 1))
        assert out["max_indices_month"] != out["max_monthly_month"]

    def test_none_when_a_mart_is_empty(self):
        out = _build_range(None, None, None, None)
        assert out == {
            "min_month": None,
            "max_indices_month": None,
            "max_monthly_complete_month": None,
            "max_monthly_month": None,
        }


class TestBuildDroughtEpisodes:
    """Calendar-aware gaps-and-islands: a missing calendar month must break the
    episode instead of silently merging two distinct droughts."""

    def _spi(self, *month_spi_pairs):
        return [{"month": date(y, m, 1), "spi": v} for (y, m), v in month_spi_pairs]

    def test_consecutive_drought_months_form_one_episode(self):
        spi_rows = self._spi(
            ((2020, 1), -1.5), ((2020, 2), -2.0), ((2020, 3), -1.2)
        )
        out = _build_drought_episodes(spi_rows, [], [])
        assert len(out) == 1
        assert out[0]["debut"] == "2020-01-01"
        assert out[0]["fin"] == "2020-03-01"
        assert out[0]["duree_mois"] == 3
        assert out[0]["index_min"] == -2.0

    def test_missing_month_in_the_middle_splits_into_two_episodes(self):
        # 2020-03 is simply absent (e.g. its SPI was NULL and filtered upstream) —
        # the drought resumes in 04-05 but must NOT be merged with 01-02.
        spi_rows = self._spi(
            ((2020, 1), -1.5), ((2020, 2), -2.0),
            ((2020, 4), -1.8), ((2020, 5), -3.0),
        )
        out = _build_drought_episodes(spi_rows, [], [])
        assert len(out) == 2
        by_debut = {e["debut"]: e for e in out}
        assert by_debut["2020-01-01"]["fin"] == "2020-02-01"
        assert by_debut["2020-01-01"]["duree_mois"] == 2
        assert by_debut["2020-04-01"]["fin"] == "2020-05-01"
        assert by_debut["2020-04-01"]["duree_mois"] == 2

    def test_non_drought_month_present_also_breaks_the_episode(self):
        # 2020-03 IS present but spi >= -1 (not a drought month) — must break too.
        spi_rows = self._spi(
            ((2020, 1), -1.5), ((2020, 2), -2.0),
            ((2020, 3), 0.2),
            ((2020, 4), -1.8), ((2020, 5), -3.0),
        )
        out = _build_drought_episodes(spi_rows, [], [])
        assert len(out) == 2
        assert {e["duree_mois"] for e in out} == {2, 2}

    def test_no_drought_rows_returns_empty_list(self):
        spi_rows = self._spi(((2020, 1), 0.5), ((2020, 2), -0.3))
        assert _build_drought_episodes(spi_rows, [], []) == []

    def test_episodes_sorted_by_duration_descending(self):
        spi_rows = self._spi(
            ((2020, 1), -1.5),
            ((2020, 3), -2.0), ((2020, 4), -2.1), ((2020, 5), -1.9),
        )
        out = _build_drought_episodes(spi_rows, [], [])
        assert [e["duree_mois"] for e in out] == [3, 1]
        assert out[0]["debut"] == "2020-03-01"

    def test_deficit_cumule_mm_sums_precip_minus_normal_within_episode_range(self):
        spi_rows = self._spi(((2020, 1), -1.5), ((2020, 2), -2.0))
        monthly_rows = [
            {"mois": date(2020, 1, 1), "precipitation_totale": 10.0},
            {"mois": date(2020, 2, 1), "precipitation_totale": 5.0},
            {"mois": date(2020, 3, 1), "precipitation_totale": 999.0},  # outside range
        ]
        clim_rows = [
            {"mois_calendaire": 1, "precip_moyenne": 40.0},
            {"mois_calendaire": 2, "precip_moyenne": 35.0},
        ]
        out = _build_drought_episodes(spi_rows, monthly_rows, clim_rows)
        assert len(out) == 1
        # (10 - 40) + (5 - 35) = -60.0 — March is excluded despite the huge value.
        assert out[0]["deficit_cumule_mm"] == -60.0

    def test_missing_climatology_or_monthly_row_yields_zero_deficit(self):
        spi_rows = self._spi(((2020, 1), -1.5))
        out = _build_drought_episodes(spi_rows, [], [])
        assert out[0]["deficit_cumule_mm"] == 0.0

    def test_episodes_generic_over_spei(self):
        # 3 consecutive months < -1 → one episode, keyed by 'spei'
        rows = _rows("spei", [(4, -1.2), (5, -1.6), (6, -0.9), (7, -2.1)])
        eps = _build_drought_episodes(rows, [], [], index_key="spei")
        assert len(eps) == 2  # (apr-may) and (jul)
        assert eps[0]["duree_mois"] == 2
        assert eps[0]["index_min"] == -1.6

    def test_episodes_default_key_is_spi(self):
        rows = _rows("spi", [(4, -1.5), (5, -1.5)])
        eps = _build_drought_episodes(rows, [], [])
        assert eps[0]["index_min"] == -1.5


class TestMergePointSeries:
    def test_joins_monthly_climatology_and_indices_by_month(self):
        monthly_rows = [
            {
                "mois": date(2026, 6, 1), "temperature_moyenne": 18.5, "temperature_min": 12.0,
                "temperature_max": 25.0, "precipitation_totale": 45.2, "etp_totale": 80.1,
                "bilan_hydrique": -34.9, "nb_jours": 30, "mois_complet": True,
            }
        ]
        clim_rows = [{"mois_calendaire": 6, "precip_moyenne": 60.0, "temp_moyenne": 17.0}]
        indices_rows = [
            {"month": date(2026, 6, 1), "fenetre": 1, "spi": -0.5, "sti": 0.3, "spei": -0.4},
            {"month": date(2026, 6, 1), "fenetre": 3, "spi": -1.2, "sti": 0.1, "spei": -1.1},
        ]
        series = _merge_point_series(monthly_rows, clim_rows, indices_rows)
        assert len(series) == 1
        entry = series[0]
        assert entry["month"] == "2026-06-01"
        assert entry["precipitation_normale"] == 60.0
        assert entry["temperature_normale"] == 17.0
        assert entry["spi_1"] == -0.5
        assert entry["spi_3"] == -1.2
        assert entry["spi_6"] is None
        assert entry["sti_12"] is None
        assert entry["spei_1"] == -0.4
        assert entry["spei_3"] == -1.1
        assert entry["spei_6"] is None

    def test_missing_climatology_leaves_normals_none(self):
        monthly_rows = [
            {
                "mois": date(1950, 1, 1), "temperature_moyenne": 5.0, "temperature_min": 1.0,
                "temperature_max": 9.0, "precipitation_totale": 10.0, "etp_totale": 5.0,
                "bilan_hydrique": 5.0, "nb_jours": 31, "mois_complet": True,
            }
        ]
        series = _merge_point_series(monthly_rows, [], [])
        assert series[0]["precipitation_normale"] is None
        assert series[0]["temperature_normale"] is None
        for w in WINDOWS:
            assert series[0][f"spi_{w}"] is None
            assert series[0][f"sti_{w}"] is None
            assert series[0][f"spei_{w}"] is None

    def test_merge_point_series_includes_spei(self):
        monthly = [{"mois": date(2026, 6, 1),
                    "temperature_moyenne": 18.0, "temperature_min": 10.0,
                    "temperature_max": 26.0, "precipitation_totale": 40.0,
                    "etp_totale": 120.0, "bilan_hydrique": -80.0,
                    "nb_jours": 30, "mois_complet": True}]
        clim = []
        indices = [{"month": date(2026, 6, 1),
                    "fenetre": 3, "spi": -1.2, "sti": 0.5, "spei": -1.5}]
        out = _merge_point_series(monthly, clim, indices)
        assert out[0]["spei_3"] == -1.5
        assert out[0]["spei_1"] is None      # window absent → None, like spi/sti


class TestMergeCompareYears:
    def test_cumulative_sum_and_normal_accumulate_across_months(self):
        monthly_rows = [
            {"mois": date(1976, 1, 1), "precipitation_totale": 10.0},
            {"mois": date(1976, 2, 1), "precipitation_totale": 20.0},
        ]
        clim_rows = [
            {"mois_calendaire": 1, "precip_moyenne": 15.0},
            {"mois_calendaire": 2, "precip_moyenne": 25.0},
        ]
        out = _merge_compare_years(monthly_rows, clim_rows, [], [1976])
        cumul = out[1976]["cumul_mensuel"]
        assert cumul[0]["cumul"] == 10.0
        assert cumul[1]["cumul"] == 30.0
        assert cumul[0]["cumul_normal"] == 15.0
        assert cumul[1]["cumul_normal"] == 40.0

    def test_requested_year_with_no_data_still_present_and_empty(self):
        out = _merge_compare_years([], [], [], [1976, 2026])
        assert out[1976] == {"cumul_mensuel": [], "spi_3": []}
        assert out[2026] == {"cumul_mensuel": [], "spi_3": []}

    def test_spi_series_grouped_by_year(self):
        spi_rows = [
            {"month": date(2003, 1, 1), "spi": -1.5},
            {"month": date(2003, 2, 1), "spi": -1.8},
        ]
        out = _merge_compare_years([], [], spi_rows, [2003])
        assert out[2003]["spi_3"] == [{"mois": 1, "spi": -1.5}, {"mois": 2, "spi": -1.8}]


def test_assert_index_accepts_all_three():
    for ok in ("spi", "sti", "spei"):
        observatory_climat._assert_index(ok)  # must NOT raise


def test_assert_index_rejects_unknown():
    with pytest.raises(HTTPException) as exc:
        observatory_climat._assert_index("bogus")
    assert exc.value.status_code == 422


class TestParamValidationViaHttp:
    """Endpoint-level validation for bad query params (no DB call is reached)."""

    def test_grid_monthly_rejects_unknown_variable(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get("/api/v1/observatory/climat/grid-monthly", params={"month": "2026-06", "variable": "nope"})
        assert r.status_code == 422

    def test_grid_indices_rejects_unknown_index(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get("/api/v1/observatory/climat/grid-indices", params={"month": "2026-06", "index": "nope"})
        assert r.status_code == 422

    def test_grid_monthly_rejects_malformed_month(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get("/api/v1/observatory/climat/grid-monthly", params={"month": "not-a-month"})
        assert r.status_code == 422

    def test_compare_years_requires_at_least_one_year(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/compare-years",
            params={"lat": 47.4, "lon": 0.7, "years": ""},
        )
        assert r.status_code == 422

    def test_compare_years_rejects_non_integer_years(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/compare-years",
            params={"lat": 47.4, "lon": 0.7, "years": "abc,2003"},
        )
        assert r.status_code == 422

    def test_grid_indices_rejects_invalid_window(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/grid-indices", params={"month": "2026-06", "window": 4}
        )
        assert r.status_code == 422

    def test_point_episodes_rejects_invalid_window(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/point-episodes",
            params={"lat": 47.4, "lon": 0.7, "window": 4},
        )
        assert r.status_code == 422

    def test_daily_temp_rejects_unknown_variable(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/daily-temp",
            params={"date": "2026-06-28", "variable": "nope"},
        )
        assert r.status_code == 422

    def test_daily_temp_rejects_malformed_date(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/daily-temp",
            params={"date": "not-a-date", "variable": "tmax"},
        )
        assert r.status_code == 422

    def test_daily_precip_rejects_malformed_date(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/daily-precip",
            params={"date": "not-a-date"},
        )
        assert r.status_code == 422


class TestDailyPrecip:
    # test_build_daily_points_formats_cells, test_build_daily_points_empty_input_yields_empty_list
    # et test_build_daily_range_is_none_safe étaient des doublons de
    # TestBuildDailyPoints/TestBuildDailyRange (plus haut dans ce fichier) : /daily-precip
    # réutilise _build_daily_points/_build_daily_range telles quelles, déjà couvertes là-bas.
    def test_daily_precip_reads_silver_never_bronze(self):
        # bronze.era5_france_timeseries a les mêmes colonnes, la même plage et le
        # même nombre de lignes que silver, mais 22 985 mailles au lieu de 11 496
        # (doublons flottants ERA5). Y taper donnerait une grille désalignée sans
        # aucun signal visible.
        assert "silver.stg_era5_timeseries" in _DAILY_PRECIP_SQL
        assert "bronze" not in _DAILY_PRECIP_SQL

    def test_daily_precip_never_casts_the_partition_column(self):
        # Régression PERF, mesurée : un cast/une fonction sur `time` casse
        # l'exclusion de chunks TimescaleDB — 413 ms -> 69 032 ms (×167) sur les
        # 321 M de lignes de la table.
        assert "time::date" not in _DAILY_PRECIP_SQL
        assert "date(time)" not in _DAILY_PRECIP_SQL
        assert "CAST(time" not in _DAILY_PRECIP_SQL.replace(" ", "")
        # la forme correcte : borne basse >= et borne haute < jour+1
        assert "time >= :day" in _DAILY_PRECIP_SQL
        assert "INTERVAL '1 day'" in _DAILY_PRECIP_SQL
