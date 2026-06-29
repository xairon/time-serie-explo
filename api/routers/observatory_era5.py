"""Observatory ERA5 router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

from datetime import date as DateType

from fastapi import APIRouter, Query
from sqlalchemy import text
from api.database import get_brgm_sync_engine

from api.config import settings
from dashboard.utils.cache import get_cached
from api.era5_anomaly import window_end_months, add_months

router = APIRouter(prefix="/api/v1/observatory/era5", tags=["observatory-era5"])

GRID_TTL = 86400
SNAPSHOT_TTL = 86400
DATES_TTL = 86400
MONTHLY_TTL = 86400
RANGE_TTL = 86400
CLIMATOLOGY_TTL = 604800  # 7 days — climatology is effectively static
ANOMALY_TTL = 86400


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
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_grid", {}, GRID_TTL, fetch)


@router.get("/snapshot")
def get_era5_snapshot(
    snapshot_date: DateType | None = Query(
        None, alias="date", description="ERA5 snapshot date; latest available day if omitted"
    ),
):
    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                d = snapshot_date
                if d is None:
                    d = conn.execute(
                        text("SELECT max(era5_date) FROM gold.int_era5_for_all_stations")
                    ).scalar()
                query = """
                    SELECT latitude, longitude,
                           temperature_2m, total_precipitation, potential_evaporation
                    FROM gold.int_era5_for_all_stations
                    WHERE era5_date = :d
                      AND (temperature_2m IS NOT NULL
                           OR total_precipitation IS NOT NULL
                           OR potential_evaporation IS NOT NULL)
                """
                result = conn.execute(text(query), {"d": d})
                return [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached(
        "obs_era5_snapshot",
        {"date": str(snapshot_date) if snapshot_date else "latest"},
        SNAPSHOT_TTL,
        fetch,
    )


@router.get("/dates")
def get_era5_dates():
    def fetch():
        query = """
            SELECT DISTINCT date_trunc('month', era5_date)::date AS month
            FROM gold.int_era5_for_all_stations
            ORDER BY month
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [str(row["month"]) for row in result.mappings()]
        finally:
            pass  # shared pooled engine; do not dispose

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
            FROM gold.int_era5_for_all_stations
            WHERE era5_date >= :month_start AND era5_date < :month_end
            GROUP BY latitude, longitude
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"month_start": month_start, "month_end": month_end})
                return [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_monthly", {"month": str(month)}, MONTHLY_TTL, fetch)


@router.get("/range")
def get_era5_range():
    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT min(era5_date) AS min_date, max(era5_date) AS max_date "
                        "FROM gold.int_era5_for_all_stations"
                    )
                ).mappings().first()
                return {"min_date": str(row["min_date"]), "max_date": str(row["max_date"])}
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_range", {}, RANGE_TTL, fetch)


def _era5_temp_climatology():
    """Per-cell long-term mean temperature for each calendar month (1950+)."""
    def fetch():
        query = """
            SELECT latitude, longitude,
                   EXTRACT(MONTH FROM era5_date)::int AS mo,
                   AVG(temperature_2m) AS mean_c
            FROM gold.int_era5_for_all_stations
            WHERE temperature_2m IS NOT NULL
            GROUP BY latitude, longitude, EXTRACT(MONTH FROM era5_date)
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_temp_climatology", {}, CLIMATOLOGY_TTL, fetch)


@router.get("/temp-anomaly")
def get_era5_temp_anomaly(
    anomaly_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    if window not in (1, 3, 6, 12):
        window = 3

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                d = anomaly_date
                if d is None:
                    d = conn.execute(
                        text("SELECT max(era5_date) FROM gold.int_era5_for_all_stations")
                    ).scalar()
                month_start = DateType(d.year, d.month, 1)
                win_start = add_months(month_start, -(window - 1))
                win_end = add_months(month_start, 1)
                rows = conn.execute(
                    text(
                        """
                        SELECT latitude, longitude, AVG(temperature_2m) AS window_mean
                        FROM gold.int_era5_for_all_stations
                        WHERE era5_date >= :win_start AND era5_date < :win_end
                          AND temperature_2m IS NOT NULL
                        GROUP BY latitude, longitude
                        """
                    ),
                    {"win_start": win_start, "win_end": win_end},
                ).mappings().all()

            # climatology (separately cached) → normal for the N ending months
            clim = _era5_temp_climatology()
            months = set(window_end_months(month_start.month, window))
            norm: dict[tuple, list] = {}
            for c in clim:
                if c["mo"] in months:
                    norm.setdefault((float(c["latitude"]), float(c["longitude"])), []).append(float(c["mean_c"]))

            out = []
            for r in rows:
                key = (float(r["latitude"]), float(r["longitude"]))
                vals = norm.get(key)
                if not vals or len(vals) < len(months) or r["window_mean"] is None:
                    continue
                normal = sum(vals) / len(vals)
                out.append(
                    {
                        "latitude": float(r["latitude"]),
                        "longitude": float(r["longitude"]),
                        "anomaly_c": float(r["window_mean"]) - normal,
                    }
                )
            return out
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached(
        "obs_era5_temp_anomaly",
        {"date": str(anomaly_date) if anomaly_date else "latest", "window": window},
        ANOMALY_TTL,
        fetch,
    )
