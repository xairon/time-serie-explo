"""Observatory ERA5 router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date as DateType, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import text
from api.database import get_brgm_sync_engine

from api.config import settings
from dashboard.utils.cache import get_cached, read_cached, write_cached
from api.era5_anomaly import (
    window_end_months,
    add_months,
    latest_complete_month,
    compute_anomalies,
    compute_precip_anomalies,
    classify_index,
)

router = APIRouter(prefix="/api/v1/observatory/era5", tags=["observatory-era5"])

GRID_TTL = 86400
SNAPSHOT_TTL = 86400
DATES_TTL = 86400
MONTHLY_TTL = 86400
RANGE_TTL = 86400
CLIMATOLOGY_TTL = 604800  # 7 days — climatology is effectively static
ANOMALY_TTL = 86400
INDICES_TTL = 86400

# Single-flight guard: only one thread runs the ~71s climatology scan at a time.
# Concurrent cache misses acquire this lock and double-check before scanning.
# Still needed by /anomaly (temperature path), which is the one remaining consumer
# of the on-the-fly climatology scan — /spi and /sti now read the precomputed
# gold.fct_era5_indices_grid mart directly (no more per-request gamma fit).
_climatology_lock = threading.Lock()
# Separate single-flight guard for the precip climatology scan (/anomaly, precipitation path).
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
                if d is None:
                    # Empty grid table — nothing to snapshot (avoids d + timedelta TypeError).
                    return []
                # No weighted duplicate-coordinate merge needed here anymore: the
                # upstream ERA5 coordinates are clean + defensively rounded to 0.1°
                # (one row per cell/day), so a plain per-row rounding is equivalent
                # to the old AVG(...)-per-rounded-cell aggregation.
                query = """
                    SELECT round(latitude::numeric, 1) AS latitude,
                           round(longitude::numeric, 1) AS longitude,
                           temperature_2m,
                           total_precipitation,
                           potential_evaporation
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
    """Per-cell monthly snapshot, read from the precomputed ``fct_era5_monthly_grid`` mart
    (no more on-the-fly aggregation over the daily grid).

    ``potential_evaporation`` is reconstructed to match the historical field exactly:
    the mart's ``etp_totale`` is a POSITIVE monthly total (mm), while this field has
    always been a NEGATIVE daily-average rate (ERA5 raw-variable convention, consumed
    as-is by the frontend's water-balance calc: ``total_precipitation + potential_evaporation``).
    ``-(etp_totale / nb_jours)`` reproduces the old ``AVG(potential_evaporation)`` value bit-for-bit.
    """
    def fetch():
        month_start = DateType(month.year, month.month, 1)
        query = """
            SELECT era5_latitude AS latitude,
                   era5_longitude AS longitude,
                   temperature_moyenne AS temperature_2m,
                   precipitation_totale AS total_precipitation,
                   CASE WHEN nb_jours > 0 THEN -(etp_totale / nb_jours) END AS potential_evaporation
            FROM gold.fct_era5_monthly_grid
            WHERE mois = :month_start
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"month_start": month_start})
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
            # Scope to WMO standard reference period 1991-2020 (matches hydro IPS ref
            # per project convention; reduces scan from 321M→126M rows, ~130s).
            # Use raw lat/lon in SQL (no ::numeric cast → faster), then merge
            # float-doublon variants by rounding to 0.1° in Python with weighted AVG
            # so climatology keys match rounded window-query coords.
            query = """
                SELECT latitude, longitude,
                       EXTRACT(MONTH FROM "time")::int AS mo,
                       AVG(temperature_2m) AS mean_c,
                       COUNT(*) AS n
                FROM gold.era5_grid
                WHERE temperature_2m IS NOT NULL
                  AND "time" >= '1991-01-01' AND "time" < '2021-01-01'
                GROUP BY latitude, longitude, EXTRACT(MONTH FROM "time")
            """
            engine = get_brgm_sync_engine()
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    rows = result.fetchall()
                # Python-side: round to 0.1° and compute weighted average across doublon variants.
                acc: dict = defaultdict(lambda: {"sw": 0.0, "sn": 0})
                for r in rows:
                    key = (round(float(r[0]), 1), round(float(r[1]), 1), int(r[2]))
                    n = int(r[4])
                    acc[key]["sw"] += float(r[3]) * n
                    acc[key]["sn"] += n
                out = [
                    {"latitude": k[0], "longitude": k[1], "mo": k[2],
                     "mean_c": v["sw"] / v["sn"]}
                    for k, v in acc.items()
                ]
                # Publish under the lock so waiters see it immediately (see write_cached).
                write_cached("obs_era5_temp_climatology", {}, CLIMATOLOGY_TTL, out)
                return out
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
            # Scope to WMO standard reference period 1991-2020 (matches hydro IPS ref).
            # Use raw lat/lon in SQL (no ::numeric cast → faster), then merge
            # float-doublon variants by rounding to 0.1° in Python with weighted AVG.
            query = """
                WITH monthly AS (
                    SELECT latitude, longitude,
                           date_trunc('month', "time") AS ym,
                           EXTRACT(MONTH FROM "time")::int AS mo,
                           SUM(total_precipitation) AS msum
                    FROM gold.era5_grid
                    WHERE total_precipitation IS NOT NULL
                      AND "time" >= '1991-01-01' AND "time" < '2021-01-01'
                    GROUP BY latitude, longitude,
                             date_trunc('month', "time"),
                             EXTRACT(MONTH FROM "time")
                )
                SELECT latitude, longitude, mo,
                       AVG(msum) AS mean_sum,
                       COUNT(*) AS n_years
                FROM monthly
                GROUP BY latitude, longitude, mo
            """
            engine = get_brgm_sync_engine()
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    rows = result.fetchall()
                # Python-side: round to 0.1° and compute weighted average across doublon variants.
                acc: dict = defaultdict(lambda: {"sw": 0.0, "sn": 0})
                for r in rows:
                    key = (round(float(r[0]), 1), round(float(r[1]), 1), int(r[2]))
                    n = int(r[4])  # n_years
                    acc[key]["sw"] += float(r[3]) * n  # mean_sum * n_years
                    acc[key]["sn"] += n
                out = [
                    {"latitude": k[0], "longitude": k[1], "mo": k[2],
                     "mean_sum": v["sw"] / v["sn"]}
                    for k, v in acc.items()
                ]
                # Publish under the lock so waiters see it immediately (see write_cached).
                write_cached("obs_era5_precip_climatology", {}, CLIMATOLOGY_TTL, out)
                return out
            finally:
                pass  # shared pooled engine; do not dispose

    return get_cached("obs_era5_precip_climatology", {}, CLIMATOLOGY_TTL, fetch)


def _resolve_month_start(conn, d):
    """First day of the window-ending month, clamped to the latest COMPLETE month.

    A supplied date is clamped (``min`` against the latest complete month) so a
    partial in-progress or future month is never scored against the complete-month
    reference distribution — otherwise the most-visible month gets a biased index.
    Returns None if the grid table is empty.
    """
    max_date = conn.execute(
        text('SELECT max("time")::date FROM gold.era5_grid')
    ).scalar()
    if max_date is None:
        return None
    latest = latest_complete_month(max_date)
    if d is None:
        return latest
    return min(DateType(d.year, d.month, 1), latest)


def _compute_temp_anomaly(conn, anomaly_date, window):
    """Computation for the temperature-anomaly path of /anomaly."""
    month_start = _resolve_month_start(conn, anomaly_date)
    if month_start is None:
        return None, None  # empty table guard
    win_start = add_months(month_start, -(window - 1))
    win_end = add_months(month_start, 1)
    rows = conn.execute(
        text(
            """
            WITH monthly AS (
                SELECT round(latitude::numeric, 1) AS latitude,
                       round(longitude::numeric, 1) AS longitude,
                       date_trunc('month', "time") AS ym,
                       AVG(temperature_2m) AS m_mean
                FROM gold.era5_grid
                WHERE "time" >= :win_start AND "time" < :win_end
                  AND temperature_2m IS NOT NULL
                GROUP BY round(latitude::numeric, 1), round(longitude::numeric, 1),
                         date_trunc('month', "time")
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


def _resolve_indices_month(conn, requested_date, window: int):
    """Resolve the target month for /spi or /sti, clamped to the latest month present
    in ``gold.fct_era5_indices_grid`` for the given window (the mart only contains
    complete calendar months by construction — no need to look at the daily grid).

    Mirrors ``_resolve_month_start``'s "latest available if omitted / clamp if too
    recent" contract, but scoped to the indices mart. Returns None if the mart has no
    row yet for this window (e.g. before the first backfill).
    """
    max_month = conn.execute(
        text("SELECT max(month) FROM gold.fct_era5_indices_grid WHERE fenetre = :window"),
        {"window": window},
    ).scalar()
    if max_month is None:
        return None
    if requested_date is None:
        return max_month
    requested_month = DateType(requested_date.year, requested_date.month, 1)
    return min(requested_month, max_month)


def _rows_to_spi(rows) -> list[dict]:
    """Format raw ``fct_era5_indices_grid`` SPI rows into the API response shape.

    No statistics are (re)computed here — SPI is precomputed upstream in the mart
    (gamma fit on the 1991-2020 reference); this only rounds and classifies.
    """
    out = []
    for r in rows:
        if r["spi"] is None:
            continue
        z = round(float(r["spi"]), 3)
        out.append({
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "spi": z,
            "index_class": classify_index(z),
        })
    return out


def _rows_to_sti(rows) -> list[dict]:
    """Format raw ``fct_era5_indices_grid`` STI rows into the API response shape.

    No statistics are (re)computed here — STI is precomputed upstream in the mart;
    this only classifies (matches the historical unrounded ``sti`` field)."""
    out = []
    for r in rows:
        if r["sti"] is None:
            continue
        z = float(r["sti"])
        out.append({
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "sti": z,
            "index_class": classify_index(z),
        })
    return out


@router.get("/spi")
def get_era5_spi(
    spi_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    """Standardized Precipitation Index (SPI, McKee 1993) per grid cell.

    Returns ``[{latitude, longitude, spi, index_class}]``, read directly from the
    precomputed ``gold.fct_era5_indices_grid`` mart — no gamma fit / CDF projection
    happens in the API anymore (that runs once upstream, in the Dagster pipeline).
    """
    if window not in (1, 3, 6, 12):
        window = 3

    if spi_date is None:
        cache_month_key = "latest"
    else:
        cache_month_key = str(DateType(spi_date.year, spi_date.month, 1))

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                month_start = _resolve_indices_month(conn, spi_date, window)
                if month_start is None:
                    return []

                rows = conn.execute(
                    text(
                        """
                        SELECT era5_latitude AS latitude, era5_longitude AS longitude, spi
                        FROM gold.fct_era5_indices_grid
                        WHERE month = :month AND fenetre = :window
                        """
                    ),
                    {"month": month_start, "window": window},
                ).mappings().all()
        finally:
            pass  # shared pooled engine; do not dispose

        return _rows_to_spi(rows)

    return get_cached(
        "obs_era5_spi",
        {"window": window, "month": cache_month_key},
        INDICES_TTL,
        fetch,
    )


@router.get("/sti")
def get_era5_sti(
    sti_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    """Standardized Temperature Index (STI) per grid cell.

    Returns ``[{latitude, longitude, sti, index_class}]``, read directly from the
    precomputed ``gold.fct_era5_indices_grid`` mart — no z-score computation happens
    in the API anymore (that runs once upstream, in the Dagster pipeline).
    """
    if window not in (1, 3, 6, 12):
        window = 3

    if sti_date is None:
        cache_month_key = "latest"
    else:
        cache_month_key = str(DateType(sti_date.year, sti_date.month, 1))

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                month_start = _resolve_indices_month(conn, sti_date, window)
                if month_start is None:
                    return []

                rows = conn.execute(
                    text(
                        """
                        SELECT era5_latitude AS latitude, era5_longitude AS longitude, sti
                        FROM gold.fct_era5_indices_grid
                        WHERE month = :month AND fenetre = :window
                        """
                    ),
                    {"month": month_start, "window": window},
                ).mappings().all()
        finally:
            pass  # shared pooled engine; do not dispose

        return _rows_to_sti(rows)

    return get_cached(
        "obs_era5_sti",
        {"window": window, "month": cache_month_key},
        INDICES_TTL,
        fetch,
    )


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
                    if result is not None:
                        # Remap anomaly_c to generic anomaly field for /anomaly endpoint
                        result = [{"latitude": r["latitude"], "longitude": r["longitude"], "anomaly": r["anomaly_c"]} for r in result]
                    return result if result is not None else []

                # precipitation path
                month_start = _resolve_month_start(conn, anomaly_date)
                if month_start is None:
                    return []
                win_start = add_months(month_start, -(window - 1))
                win_end = add_months(month_start, 1)
                rows = conn.execute(
                    text(
                        # Weighted-merge the ERA5 float-coordinate doublons per (rounded
                        # cell, month) before accumulating, consistent with the climatology
                        # builder — a raw SUM would double-count duplicated cells.
                        """
                        WITH per_variant AS (
                            SELECT latitude, longitude,
                                   date_trunc('month', "time") AS ym,
                                   SUM(total_precipitation) AS m_sum,
                                   COUNT(*) AS n
                            FROM gold.era5_grid
                            WHERE "time" >= :win_start AND "time" < :win_end
                              AND total_precipitation IS NOT NULL
                            GROUP BY latitude, longitude, date_trunc('month', "time")
                        ),
                        monthly AS (
                            SELECT round(latitude::numeric, 1) AS latitude,
                                   round(longitude::numeric, 1) AS longitude,
                                   ym,
                                   SUM(m_sum * n) / NULLIF(SUM(n), 0) AS m_merged
                            FROM per_variant
                            GROUP BY round(latitude::numeric, 1), round(longitude::numeric, 1), ym
                        )
                        SELECT latitude, longitude,
                               SUM(m_merged) AS precip_sum,
                               COUNT(*) AS n_months
                        FROM monthly
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

