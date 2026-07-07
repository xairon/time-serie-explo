"""Observatory ERA5 router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

from collections import defaultdict
from datetime import date as DateType, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import text
from api.database import get_brgm_sync_engine

from api.config import settings
from dashboard.utils.cache import get_cached
from api.era5_anomaly import classify_index

router = APIRouter(prefix="/api/v1/observatory/era5", tags=["observatory-era5"])

GRID_TTL = 86400
SNAPSHOT_TTL = 86400
DATES_TTL = 86400
MONTHLY_TTL = 86400
RANGE_TTL = 86400
INDICES_TTL = 86400


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


def _rows_to_snapshot(rows) -> list[dict]:
    """Merge duplicate-coordinate rows into one row per rounded 0.1° grid cell.

    CONFIRMED (verified against the live warehouse): ``gold.era5_grid`` is a view
    over ``bronze.era5_france_timeseries``, which was never purged — unrounded
    float coordinates still exist there for 2026-01-21→2026-05-01 (~1.16M rows;
    only SILVER was purged/re-staged by the upstream remediation). Two float
    variants of the same cell (e.g. ``47.09999999999994`` and ``47.1``) must be
    merged into a single row, averaging the three variables per cell — otherwise
    /snapshot returns duplicate/overlapping cells for any date in that window
    (and defensively, for any future ingestion regression that reintroduces
    unrounded coordinates). This reproduces the pre-8231128
    ``GROUP BY round(...) / AVG(...)`` SQL behaviour, but merges in Python so the
    logic can be unit-tested without touching the database.
    """
    fields = ("temperature_2m", "total_precipitation", "potential_evaporation")
    acc: dict = defaultdict(lambda: {f: [] for f in fields})
    for r in rows:
        key = (round(float(r["latitude"]), 1), round(float(r["longitude"]), 1))
        bucket = acc[key]
        for f in fields:
            v = r[f]
            if v is not None:
                bucket[f].append(float(v))
    return [
        {
            "latitude": lat,
            "longitude": lon,
            **{f: (sum(vals) / len(vals) if vals else None) for f, vals in bucket.items()},
        }
        for (lat, lon), bucket in acc.items()
    ]


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
                # Duplicate-coordinate merge IS required here: gold.era5_grid reads
                # bronze (never purged), which still has unrounded float coordinates
                # for jan-avr 2026 (and potentially future ingestion regressions).
                # See _rows_to_snapshot docstring for the verified detail.
                query = """
                    SELECT latitude,
                           longitude,
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
                return _rows_to_snapshot([dict(r._mapping) for r in result])
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


def _resolve_indices_month(conn, requested_date, window: int):
    """Resolve the target month for /spi or /sti, clamped to the latest month present
    in ``gold.fct_era5_indices_grid`` for the given window (the mart only contains
    complete calendar months by construction — no need to look at the daily grid).

    Uses a "latest available if omitted / clamp if too recent" contract scoped to
    the indices mart. Returns None if the mart has no row yet for this window (e.g.
    before the first backfill).
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
