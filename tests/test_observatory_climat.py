"""Unit tests for the observatory_climat router — pure helpers + param validation.

Follows the repo convention (see test_era5_spi.py): exercise the pure Python
transforms directly with plain dicts standing in for SQLAlchemy row mappings,
rather than hitting the real BRGM warehouse.
"""
from datetime import date

import pytest
from fastapi import HTTPException

from api.era5_anomaly import DROUGHT_SPI_THRESHOLD, _STI_THRESHOLDS
from api.routers.observatory_climat import (
    router,
    _parse_month,
    _parse_date,
    _round_cell,
    _build_situation_summary,
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


def test_router_mounts_all_twelve_climat_paths():
    paths = {r.path for r in router.routes}
    assert paths == {
        "/api/v1/observatory/climat/range",
        "/api/v1/observatory/climat/grid-monthly",
        "/api/v1/observatory/climat/grid-indices",
        "/api/v1/observatory/climat/daily-temp",
        "/api/v1/observatory/climat/daily-temp-range",
        "/api/v1/observatory/climat/daily-precip",
        "/api/v1/observatory/climat/daily-precip-range",
        "/api/v1/observatory/climat/situation-summary",
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


class TestBuildSituationSummary:
    def _rows(self, spis):
        return [{"era5_latitude": 45.0 + i * 0.1, "era5_longitude": 2.0, "spi": v} for i, v in enumerate(spis)]

    def test_empty_rows_yields_zeroed_summary(self):
        out = _build_situation_summary([], [], date(2026, 6, 1), 3)
        assert out["n_cells"] == 0
        assert out["median_spi"] is None
        assert out["is_driest_on_record"] is False
        assert out["top5_cellules_seches"] == []
        assert sum(out["classes_pct"].values()) == 0.0
        # No SPI computed for this month/window yet (e.g. the partial current
        # month) — must be flagged explicitly, not read as a real 0% drought.
        assert out["available"] is False

    def test_non_empty_rows_yields_available_true(self):
        out = _build_situation_summary(self._rows([-2.0, -0.5]), [], date(2026, 6, 1), 3)
        assert out["available"] is True

    def test_classes_pct_sums_to_100(self):
        spis = [-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, -1.2]
        out = _build_situation_summary(self._rows(spis), [], date(2026, 6, 1), 3)
        assert out["n_cells"] == 8
        assert abs(sum(out["classes_pct"].values()) - 100.0) < 1e-6

    def test_pct_secheresse_is_readable_off_the_class_bar(self):
        """Le % du bandeau doit se retrouver à l'œil sur la barre de synthèse :
        il vaut exactement la somme des 3 classes les plus sèches. L'ancien seuil
        -1.0 coupait au milieu de la classe BAS ([-1.28, -0.84)), ce qui rendait
        le % invérifiable contre la légende affichée juste à côté."""
        # -0.9 est « Modérément sec » (BAS) tout en étant > -1.0 : c'est lui qui
        # sépare l'ancien seuil du nouveau (ancien -> 33.33 %, nouveau -> 50 %).
        spis = [-2.0, -1.5, -0.9, -0.5, 0.0, 1.0]
        out = _build_situation_summary(self._rows(spis), [], date(2026, 6, 1), 3)
        cp = out["classes_pct"]
        dry_classes = cp["EXTREMEMENT_BAS"] + cp["TRES_BAS"] + cp["BAS"]
        # Tolérance : chaque classe est arrondie INDÉPENDAMMENT à 2 décimales, donc la
        # somme des arrondis dérive de l'arrondi de la somme (ici 16.67*3 = 50.01 vs
        # 50.0). L'écart est borné par 3 * 0.005 et invisible à l'affichage — mais il
        # existe, et l'égalité stricte serait fausse.
        assert out["pct_secheresse"] == pytest.approx(dry_classes, abs=0.02)
        assert out["pct_secheresse"] == pytest.approx(50.0)  # 3 cellules sur 6

    def test_pct_secheresse_excludes_the_normal_boundary_exactly(self):
        """-0.84 appartient à NORMAL (convention lo <= z < hi de classify_index),
        donc il ne compte PAS comme sécheresse. Sans ça, le % dépasserait la somme
        des classes sèches et l'invariant ci-dessus casserait au bord."""
        out = _build_situation_summary(self._rows([-0.84, -0.85]), [], date(2026, 6, 1), 3)
        assert out["classes_pct"]["NORMAL"] == 50.0
        assert out["classes_pct"]["BAS"] == 50.0
        assert out["pct_secheresse"] == 50.0  # seul -0.85 compte

    def test_drought_threshold_is_the_normal_class_lower_boundary(self):
        """Le seuil n'est pas un nombre libre : c'est la frontière basse de la
        classe NORMAL, donc une limite visible sur la légende. Ce test casse si
        quelqu'un déplace l'un sans l'autre."""
        normal_lo = next(lo for lo, _hi, cls in _STI_THRESHOLDS if cls == "NORMAL")
        assert DROUGHT_SPI_THRESHOLD == normal_lo

    def test_top5_returns_the_five_driest_cells_sorted_ascending(self):
        spis = [0.5, -3.0, -1.0, -2.5, 1.0, -0.2, -2.0]
        out = _build_situation_summary(self._rows(spis), [], date(2026, 6, 1), 3)
        top5_values = [c["spi"] for c in out["top5_cellules_seches"]]
        assert top5_values == sorted(spis)[:5]
        assert len(out["top5_cellules_seches"]) == 5

    def test_driest_on_record_when_no_year_was_as_dry(self):
        rows = self._rows([-2.5, -2.5])  # median -2.5
        year_rows = [
            {"yr": 2020, "median_spi": -1.0},
            {"yr": 2015, "median_spi": -0.5},
        ]
        out = _build_situation_summary(rows, year_rows, date(2026, 6, 1), 3)
        assert out["is_driest_on_record"] is True
        assert out["driest_since_year"] == 2015  # earliest year in the record

    def test_driest_since_year_finds_the_last_year_as_dry_or_drier(self):
        rows = self._rows([-1.0, -1.0])  # current median -1.0
        year_rows = [
            {"yr": 2020, "median_spi": -0.2},
            {"yr": 1990, "median_spi": -1.5},  # as dry or drier than current
            {"yr": 1976, "median_spi": -2.0},
        ]
        out = _build_situation_summary(rows, year_rows, date(2026, 6, 1), 3)
        assert out["is_driest_on_record"] is False
        # 1990 was as dry/drier -> driest since 1991 (the year right after)
        assert out["driest_since_year"] == 1991


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
        assert out[0]["spi_min"] == -2.0

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
            {"month": date(2026, 6, 1), "fenetre": 1, "spi": -0.5, "sti": 0.3},
            {"month": date(2026, 6, 1), "fenetre": 3, "spi": -1.2, "sti": 0.1},
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

    def test_situation_summary_rejects_invalid_window(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        r = client.get(
            "/api/v1/observatory/climat/situation-summary", params={"month": "2026-06", "window": 4}
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
