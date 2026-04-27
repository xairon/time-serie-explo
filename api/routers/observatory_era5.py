"""Observatory ERA5 router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

from datetime import date as DateType

from fastapi import APIRouter, Query
from sqlalchemy import create_engine, text

from api.config import settings
from dashboard.utils.cache import get_cached

router = APIRouter(prefix="/api/v1/observatory/era5", tags=["observatory-era5"])

GRID_TTL = 86400
SNAPSHOT_TTL = 86400
DATES_TTL = 86400
MONTHLY_TTL = 86400


def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


@router.get("/grid")
def get_era5_grid():
    def fetch():
        query = """
            SELECT era5_latitude, era5_longitude
            FROM gold.int_era5_grid_points
            ORDER BY era5_latitude, era5_longitude
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

    return get_cached("obs_era5_grid", {}, GRID_TTL, fetch)


@router.get("/snapshot")
def get_era5_snapshot(
    snapshot_date: DateType = Query(..., alias="date", description="Date for the ERA5 snapshot"),
):
    def fetch():
        query = """
            SELECT latitude, longitude,
                   temperature_2m, total_precipitation, potential_evaporation
            FROM gold.int_era5_for_stations
            WHERE era5_date = :snapshot_date
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"snapshot_date": snapshot_date})
                return [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

    return get_cached("obs_era5_snapshot", {"date": str(snapshot_date)}, SNAPSHOT_TTL, fetch)


@router.get("/dates")
def get_era5_dates():
    def fetch():
        query = """
            SELECT DISTINCT date_trunc('month', era5_date)::date AS month
            FROM gold.int_era5_for_stations
            ORDER BY month
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [str(row["month"]) for row in result.mappings()]
        finally:
            engine.dispose()

    return get_cached("obs_era5_dates", {}, DATES_TTL, fetch)


@router.get("/monthly")
def get_era5_monthly(
    month: DateType = Query(..., description="Month in YYYY-MM-DD format (first of month)"),
):
    def fetch():
        month_start = month
        if month.month == 12:
            month_end = DateType(month.year + 1, 1, 1)
        else:
            month_end = DateType(month.year, month.month + 1, 1)
        query = """
            SELECT latitude, longitude,
                   AVG(temperature_2m) AS temperature_2m,
                   SUM(total_precipitation) AS total_precipitation,
                   AVG(potential_evaporation) AS potential_evaporation
            FROM gold.int_era5_for_stations
            WHERE era5_date >= :month_start AND era5_date < :month_end
            GROUP BY latitude, longitude
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"month_start": month_start, "month_end": month_end})
                return [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

    return get_cached("obs_era5_monthly", {"month": str(month)}, MONTHLY_TTL, fetch)
