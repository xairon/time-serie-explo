"""Observatory ERA5 router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

import threading
from datetime import date as DateType, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import text
from api.database import get_brgm_sync_engine

from api.config import settings
from dashboard.utils.cache import get_cached, read_cached
from api.era5_anomaly import window_end_months, add_months, latest_complete_month, compute_anomalies, compute_precip_anomalies

router = APIRouter(prefix="/api/v1/observatory/era5", tags=["observatory-era5"])

GRID_TTL = 86400
SNAPSHOT_TTL = 86400
DATES_TTL = 86400
MONTHLY_TTL = 86400
RANGE_TTL = 86400
CLIMATOLOGY_TTL = 604800  # 7 days — climatology is effectively static
ANOMALY_TTL = 86400

# Single-flight guard: only one thread runs the ~71s climatology scan at a time.
# Concurrent cache misses acquire this lock and double-check before scanning.
_climatology_lock = threading.Lock()
# Separate single-flight guard for the precip climatology scan.
_precip_climatology_lock = threading.Lock()


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
                        text('SELECT max("time")::date FROM gold.era5_grid')
                    ).scalar()
                query = """
                    SELECT latitude, longitude,
                           temperature_2m, total_precipitation, potential_evaporation
                    FROM gold.era5_grid
                    WHERE "time" >= :d AND "time" < :d_next
                      AND (temperature_2m IS NOT NULL
                           OR total_precipitation IS NOT NULL
                           OR potential_evaporation IS NOT NULL)
                """
                result = conn.execute(text(query), {"d": d, "d_next": d + timedelta(days=1)})
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
            SELECT DISTINCT date_trunc('month', "time")::date AS month
            FROM gold.era5_grid
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
            FROM gold.era5_grid
            WHERE "time" >= :month_start AND "time" < :month_end
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
                        'SELECT min("time")::date AS min_date, max("time")::date AS max_date '
                        "FROM gold.era5_grid"
                    )
                ).mappings().first()
                return {"min_date": str(row["min_date"]), "max_date": str(row["max_date"])}
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_range", {}, RANGE_TTL, fetch)


def _era5_temp_climatology():
    """Per-cell long-term mean temperature for each calendar month (1950+)."""
    def fetch():
        # Single-flight: acquire the lock before running the ~71s full-table scan.
        # All concurrent cache misses serialize here; the first to enter runs the
        # scan, the rest double-check via read_cached and return immediately.
        with _climatology_lock:
            cached = read_cached("obs_era5_temp_climatology", {})
            if cached is not None:
                return cached

            # We hold the lock and the cache is empty — run the expensive scan.
            query = """
                SELECT latitude, longitude,
                       EXTRACT(MONTH FROM "time")::int AS mo,
                       AVG(temperature_2m) AS mean_c
                FROM gold.era5_grid
                WHERE temperature_2m IS NOT NULL
                GROUP BY latitude, longitude, EXTRACT(MONTH FROM "time")
            """
            engine = get_brgm_sync_engine()
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    return [dict(r._mapping) for r in result]
            finally:
                pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_temp_climatology", {}, CLIMATOLOGY_TTL, fetch)


def _era5_precip_climatology():
    """Per-cell long-term mean monthly precipitation sum for each calendar month (1950+).
    Computed as: for each (cell, calendar month), average the monthly SUM across all years."""
    def fetch():
        # Single-flight: acquire the lock before running the expensive full-table scan.
        # Concurrent cache misses serialize here; the first runner computes, the rest
        # double-check via read_cached and return immediately.
        with _precip_climatology_lock:
            cached = read_cached("obs_era5_precip_climatology", {})
            if cached is not None:
                return cached

            # We hold the lock and the cache is empty — run the expensive scan.
            query = """
                WITH monthly AS (
                    SELECT latitude, longitude,
                           date_trunc('month', "time") AS ym,
                           EXTRACT(MONTH FROM "time")::int AS mo,
                           SUM(total_precipitation) AS msum
                    FROM gold.era5_grid
                    WHERE total_precipitation IS NOT NULL
                    GROUP BY latitude, longitude,
                             date_trunc('month', "time"),
                             EXTRACT(MONTH FROM "time")
                )
                SELECT latitude, longitude, mo, AVG(msum) AS mean_sum
                FROM monthly
                GROUP BY latitude, longitude, mo
            """
            engine = get_brgm_sync_engine()
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    return [dict(r._mapping) for r in result]
            finally:
                pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_precip_climatology", {}, CLIMATOLOGY_TTL, fetch)


def _compute_temp_anomaly(conn, anomaly_date, window):
    """Shared computation for temperature anomaly used by both /temp-anomaly and /anomaly."""
    d = anomaly_date
    if d is None:
        d = conn.execute(
            text('SELECT max("time")::date FROM gold.era5_grid')
        ).scalar()
        if d is None:
            return None, None  # empty table guard
        month_start = latest_complete_month(d)
    else:
        month_start = DateType(d.year, d.month, 1)
    win_start = add_months(month_start, -(window - 1))
    win_end = add_months(month_start, 1)
    rows = conn.execute(
        text(
            """
            WITH monthly AS (
                SELECT latitude, longitude,
                       date_trunc('month', "time") AS ym,
                       AVG(temperature_2m) AS m_mean
                FROM gold.era5_grid
                WHERE "time" >= :win_start AND "time" < :win_end
                  AND temperature_2m IS NOT NULL
                GROUP BY latitude, longitude, date_trunc('month', "time")
            )
            SELECT latitude, longitude,
                   AVG(m_mean) AS window_mean,
                   COUNT(*) AS n_months
            FROM monthly
            GROUP BY latitude, longitude
            """
        ),
        {"win_start": win_start, "win_end": win_end},
    ).mappings().all()
    clim = _era5_temp_climatology()
    result = compute_anomalies(rows, clim, window_end_months(month_start.month, window), window)
    return result, month_start


@router.get("/anomaly")
def get_era5_anomaly(
    variable: str = Query("temperature", description="Variable: temperature or precipitation"),
    anomaly_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    if window not in (1, 3, 6, 12):
        window = 3
    if variable not in ("temperature", "precipitation"):
        variable = "temperature"

    # Build cache key at month granularity before touching the DB
    if anomaly_date is None:
        cache_month_key = "latest"
    else:
        cache_month_key = str(DateType(anomaly_date.year, anomaly_date.month, 1))

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                if variable == "temperature":
                    result, _ = _compute_temp_anomaly(conn, anomaly_date, window)
                    return result if result is not None else []

                # precipitation path
                d = anomaly_date
                if d is None:
                    d = conn.execute(
                        text('SELECT max("time")::date FROM gold.era5_grid')
                    ).scalar()
                    if d is None:
                        return []
                    month_start = latest_complete_month(d)
                else:
                    month_start = DateType(d.year, d.month, 1)
                win_start = add_months(month_start, -(window - 1))
                win_end = add_months(month_start, 1)
                rows = conn.execute(
                    text(
                        """
                        SELECT latitude, longitude,
                               SUM(total_precipitation) AS precip_sum,
                               COUNT(DISTINCT date_trunc('month', "time")) AS n_months
                        FROM gold.era5_grid
                        WHERE "time" >= :win_start AND "time" < :win_end
                          AND total_precipitation IS NOT NULL
                        GROUP BY latitude, longitude
                        """
                    ),
                    {"win_start": win_start, "win_end": win_end},
                ).mappings().all()
                clim = _era5_precip_climatology()
                return compute_precip_anomalies(
                    rows, clim, window_end_months(month_start.month, window), window
                )
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached(
        "obs_era5_anomaly",
        {"variable": variable, "month": cache_month_key, "window": window},
        ANOMALY_TTL,
        fetch,
    )


@router.get("/temp-anomaly")
def get_era5_temp_anomaly(
    anomaly_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    if window not in (1, 3, 6, 12):
        window = 3

    # M2: build cache key at month granularity before touching the DB
    if anomaly_date is None:
        cache_month_key = "latest"
    else:
        cache_month_key = str(DateType(anomaly_date.year, anomaly_date.month, 1))

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                d = anomaly_date
                if d is None:
                    d = conn.execute(
                        text('SELECT max("time")::date FROM gold.era5_grid')
                    ).scalar()
                    if d is None:  # M1: empty table guard
                        return []
                    month_start = latest_complete_month(d)  # I1: use latest complete month
                else:
                    month_start = DateType(d.year, d.month, 1)
                win_start = add_months(month_start, -(window - 1))
                win_end = add_months(month_start, 1)
                rows = conn.execute(
                    text(
                        """
                        WITH monthly AS (
                            SELECT latitude, longitude,
                                   date_trunc('month', "time") AS ym,
                                   AVG(temperature_2m) AS m_mean
                            FROM gold.era5_grid
                            WHERE "time" >= :win_start AND "time" < :win_end
                              AND temperature_2m IS NOT NULL
                            GROUP BY latitude, longitude, date_trunc('month', "time")
                        )
                        SELECT latitude, longitude,
                               AVG(m_mean) AS window_mean,
                               COUNT(*) AS n_months
                        FROM monthly
                        GROUP BY latitude, longitude
                        """
                    ),
                    {"win_start": win_start, "win_end": win_end},
                ).mappings().all()

            # climatology (separately cached) → normal for the N ending months
            clim = _era5_temp_climatology()
            return compute_anomalies(rows, clim, window_end_months(month_start.month, window), window)
        finally:
            pass  # shared pooled engine; do not dispose

    return get_cached(
        "obs_era5_temp_anomaly",
        {"month": cache_month_key, "window": window},
        ANOMALY_TTL,
        fetch,
    )
