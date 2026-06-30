"""Observatory ERA5 router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date as DateType, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import text
from api.database import get_brgm_sync_engine

from api.config import settings
from dashboard.utils.cache import get_cached, read_cached
from api.era5_anomaly import window_end_months, add_months, latest_complete_month, compute_anomalies, compute_precip_anomalies, compute_sti

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
# Single-flight guard for the STI reference scan.
_sti_reference_lock = threading.Lock()


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
                    SELECT round(latitude::numeric, 1) AS latitude,
                           round(longitude::numeric, 1) AS longitude,
                           AVG(temperature_2m) AS temperature_2m,
                           AVG(total_precipitation) AS total_precipitation,
                           AVG(potential_evaporation) AS potential_evaporation
                    FROM gold.era5_grid
                    WHERE "time" >= :d AND "time" < :d_next
                      AND (temperature_2m IS NOT NULL
                           OR total_precipitation IS NOT NULL
                           OR potential_evaporation IS NOT NULL)
                    GROUP BY round(latitude::numeric, 1), round(longitude::numeric, 1)
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
            SELECT round(latitude::numeric, 1) AS latitude,
                   round(longitude::numeric, 1) AS longitude,
                   AVG(temperature_2m) AS temperature_2m,
                   SUM(total_precipitation) AS total_precipitation,
                   AVG(potential_evaporation) AS potential_evaporation
            FROM gold.era5_grid
            WHERE "time" >= :month_start AND "time" < :month_end
            GROUP BY round(latitude::numeric, 1), round(longitude::numeric, 1)
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
                return [
                    {"latitude": k[0], "longitude": k[1], "mo": k[2],
                     "mean_c": v["sw"] / v["sn"]}
                    for k, v in acc.items()
                ]
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
                return [
                    {"latitude": k[0], "longitude": k[1], "mo": k[2],
                     "mean_sum": v["sw"] / v["sn"]}
                    for k, v in acc.items()
                ]
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


def _era5_sti_reference(window: int, end_month: int) -> list[dict]:
    """Per-cell reference distribution for STI: mean and std of N-month window means over 1991-2020.

    For each reference year in 1991-2020, computes the N-month window mean temperature ending at
    ``end_month`` per cell, then returns AVG and STDDEV_SAMP over years.

    Year-boundary windows (e.g. window=3, end_month=1 → months Nov/Dec/Jan): each row is
    assigned to the year of its window's END via CASE WHEN mo > end_month THEN yr+1 ELSE yr.

    Coords rounded to 0.1° in Python (same doublon-merge strategy as the temp climatology).
    Cached keyed (window, end_month), own single-flight lock, TTL 7 days.
    """
    cache_key = "obs_era5_sti_ref"
    cache_params = {"window": window, "end_month": end_month}

    def fetch():
        with _sti_reference_lock:
            cached = read_cached(cache_key, cache_params)
            if cached is not None:
                return cached

            months = window_end_months(end_month, window)
            # Earliest data needed: reference year 1991 starting (window-1) months before end_month
            ref_start = add_months(DateType(1991, end_month, 1), -(window - 1))

            # Step 1: raw SQL — per (latitude, longitude, year, month) → monthly mean + count.
            # Raw lat/lon (no ::numeric cast) for speed; Python-side rounding handles doublons.
            query = """
                SELECT latitude, longitude,
                       EXTRACT(YEAR FROM "time")::int AS yr,
                       EXTRACT(MONTH FROM "time")::int AS mo,
                       AVG(temperature_2m) AS m_mean,
                       COUNT(*) AS n
                FROM gold.era5_grid
                WHERE temperature_2m IS NOT NULL
                  AND "time" >= :ref_start AND "time" < '2021-01-01'
                  AND EXTRACT(MONTH FROM "time")::int = ANY(:months)
                GROUP BY latitude, longitude,
                         EXTRACT(YEAR FROM "time")::int,
                         EXTRACT(MONTH FROM "time")::int
            """
            engine = get_brgm_sync_engine()
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query), {"ref_start": ref_start, "months": months})
                    rows = result.fetchall()
            finally:
                pass  # shared pooled engine; do not dispose

            # Step 2: round to 0.1°, weighted-avg doublon variants per (rounded_lat, rounded_lon, yr, mo)
            acc1: dict = defaultdict(lambda: {"sw": 0.0, "sn": 0})
            for r in rows:
                la = round(float(r[0]), 1)
                lo = round(float(r[1]), 1)
                yr = int(r[2])
                mo = int(r[3])
                m_mean = float(r[4])
                n = int(r[5])
                acc1[(la, lo, yr, mo)]["sw"] += m_mean * n
                acc1[(la, lo, yr, mo)]["sn"] += n

            # Step 3: assign ending year (wy) and collect monthly means per (la, lo, wy)
            # mo > end_month means this row belongs to the PREVIOUS calendar year of the window's end
            acc2: dict = defaultdict(list)
            for (la, lo, yr, mo), v in acc1.items():
                merged_mean = v["sw"] / v["sn"]
                wy = yr + 1 if mo > end_month else yr
                acc2[(la, lo, wy)].append(merged_mean)

            # Step 4: filter complete windows in the 1991-2020 reference period, compute wmean per year
            acc3: dict = defaultdict(list)
            for (la, lo, wy), month_means in acc2.items():
                if wy < 1991 or wy > 2020:
                    continue
                if len(month_means) != window:
                    continue  # incomplete window (boundary year or missing data)
                wmean = sum(month_means) / len(month_means)
                acc3[(la, lo)].append(wmean)

            # Step 5: AVG + STDDEV_SAMP (sample std) over reference years per cell
            out = []
            for (la, lo), wmeans in acc3.items():
                n_years = len(wmeans)
                if n_years < 2:
                    continue  # need at least 2 for sample std
                mean_val = sum(wmeans) / n_years
                variance = sum((w - mean_val) ** 2 for w in wmeans) / (n_years - 1)
                std_val = variance ** 0.5
                out.append({
                    "latitude": la,
                    "longitude": lo,
                    "mean": mean_val,
                    "std": std_val,
                    "n_years": n_years,
                })
            return out

    return get_cached(cache_key, cache_params, CLIMATOLOGY_TTL, fetch)


def _warm_era5_sti_default():
    """Warm the STI reference for window=3 + latest complete month. Used at startup/re-warm."""
    try:
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            max_date = conn.execute(
                text('SELECT max("time")::date FROM gold.era5_grid')
            ).scalar()
        if max_date is None:
            return None
        end_month = latest_complete_month(max_date).month
        return _era5_sti_reference(3, end_month)
    except Exception:
        raise


@router.get("/sti")
def get_era5_sti(
    sti_date: DateType | None = Query(
        None, alias="date", description="Window end date; latest available month if omitted"
    ),
    window: int = Query(3, description="Window length in months (1, 3, 6 or 12)"),
):
    """Standardized Temperature Index (STI) per grid cell.

    Returns ``[{latitude, longitude, sti, index_class}]`` using the SPI method applied to
    temperature: z = (observed_window_mean − reference_mean) / reference_std,
    classified into 7 McKee/WMO classes (±0.84 / ±1.28 / ±1.75 thresholds).

    Reference distribution: 1991-2020, per (cell, window, end-calendar-month).
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
                d = sti_date
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
        finally:
            pass  # shared pooled engine; do not dispose

        ref = _era5_sti_reference(window, month_start.month)
        return compute_sti(rows, ref, window)

    return get_cached(
        "obs_era5_sti",
        {"window": window, "month": cache_month_key},
        ANOMALY_TTL,
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
                        SELECT round(latitude::numeric, 1) AS latitude,
                               round(longitude::numeric, 1) AS longitude,
                               SUM(total_precipitation) AS precip_sum,
                               COUNT(DISTINCT date_trunc('month', "time")) AS n_months
                        FROM gold.era5_grid
                        WHERE "time" >= :win_start AND "time" < :win_end
                          AND total_precipitation IS NOT NULL
                        GROUP BY round(latitude::numeric, 1), round(longitude::numeric, 1)
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
