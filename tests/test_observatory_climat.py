"""Unit tests for the observatory_climat router — pure helpers + param validation.

Follows the repo convention (see test_era5_spi.py / test_era5_resolve_month_start.py):
exercise the pure Python transforms directly with plain dicts standing in for
SQLAlchemy row mappings, rather than hitting the real BRGM warehouse.
"""
from datetime import date

import pytest
from fastapi import HTTPException

from api.routers.observatory_climat import (
    router,
    _parse_month,
    _round_cell,
    _build_situation_summary,
    _build_drought_episodes,
    _merge_point_series,
    _merge_compare_years,
    _MONTHLY_VARIABLES,
    WINDOWS,
)


def test_router_mounts_all_seven_climat_paths():
    paths = {r.path for r in router.routes}
    assert paths == {
        "/api/v1/observatory/climat/grid-monthly",
        "/api/v1/observatory/climat/grid-indices",
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

    def test_classes_pct_sums_to_100(self):
        spis = [-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, -1.2]
        out = _build_situation_summary(self._rows(spis), [], date(2026, 6, 1), 3)
        assert out["n_cells"] == 8
        assert abs(sum(out["classes_pct"].values()) - 100.0) < 1e-6

    def test_pct_secheresse_counts_spi_below_minus_one(self):
        spis = [-2.0, -1.5, -0.5, 0.0]  # 2 of 4 are < -1
        out = _build_situation_summary(self._rows(spis), [], date(2026, 6, 1), 3)
        assert out["pct_secheresse"] == 50.0

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
