"""Observatory Climat router — sync SQLAlchemy SELECTs against the ERA5 grid marts.

No statistics are recomputed here: everything (monthly aggregates, 1991-2020
normals, SPI/STI) already lives in the BRGM warehouse gold marts
(``fct_era5_monthly_grid``, ``fct_era5_climatology_grid``, ``fct_era5_indices_grid``).
This router is a thin read layer with simple aggregates (per-cell values,
gaps-and-islands) and a 24h Redis cache, mirroring ``observatory_era5.py``.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import date as DateType
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import text

from api.database import get_brgm_sync_engine
from api.era5_anomaly import classify_index
from dashboard.utils.cache import get_cached

router = APIRouter(prefix="/api/v1/observatory/climat", tags=["observatory-climat"])

GRID_TTL = 86400  # 24h, per Global Constraints

WINDOWS = (1, 3, 6, 12)

_MONTHLY_VARIABLES = {
    "temperature": "temperature_moyenne",
    "temperature_min": "temperature_min",
    "temperature_max": "temperature_max",
    "precipitation": "precipitation_totale",
    "etp": "etp_totale",
    "bilan_hydrique": "bilan_hydrique",
}

# TRUE daily statistics (24 hourly steps, not the 00h UTC instant used by the
# monthly/index marts) from silver.stg_era5_daily_temp_stats — a Silver staging
# table (plain latitude/longitude columns, unlike the gold marts' era5_latitude/
# era5_longitude), populated by a nightly J-7 ingestion + an independent
# 1950-2025 backfill running in the sibling data-pipeline repo. Coverage is
# partial — always query /daily-temp-range for the actual min/max date.
_DAILY_TEMP_VARIABLES = {"tmax": "t2m_max", "tmin": "t2m_min", "tmean": "t2m_mean"}
DAILY_TEMP_RANGE_TTL = 3600  # 1h — the date window moves daily (nightly J-7 ingestion), not monthly


def _num(v):
    """Cast a numeric/Decimal DB value to float, preserving None."""
    return float(v) if v is not None else None


def _parse_month(value: str) -> DateType:
    """Parse a ``YYYY-MM`` (or ``YYYY-MM-DD``) string into the first-of-month date."""
    parts = value.split("-")
    try:
        if len(parts) not in (2, 3):
            raise ValueError
        year, month = int(parts[0]), int(parts[1])
        return DateType(year, month, 1)
    except (ValueError, IndexError):
        raise HTTPException(422, f"Mois invalide : {value!r} (attendu YYYY-MM)")


def _parse_date(value: str) -> DateType:
    """Parse a ``YYYY-MM-DD`` string into a date. Stricter than ``_parse_month`` —
    the daily grid has no "truncate to the 1st" convenience, a full day is required."""
    try:
        return DateType.fromisoformat(value)
    except (ValueError, TypeError):
        raise HTTPException(422, f"Date invalide : {value!r} (attendu YYYY-MM-DD)")


def _round_cell(lat: float, lon: float) -> tuple[float, float]:
    """Round lat/lon to the nearest 0.1° — grid cells are exact 0.1 multiples."""
    return round(lat, 1), round(lon, 1)


def _validate_window(window: int) -> None:
    """Reject an unknown window with 422, like variable/index/month/years elsewhere
    in this router — a silent fallback to 3 would hide a client-side bug."""
    if window not in WINDOWS:
        raise HTTPException(422, f"Fenêtre invalide : {window!r} (attendu {WINDOWS})")


_GRID_INDICES = ("spi", "sti", "spei")


def _assert_index(index: str) -> None:
    """Reject an unknown grid index with 422 — mirrors ``_validate_window``."""
    if index not in _GRID_INDICES:
        raise HTTPException(422, f"Indice inconnu : {index!r} (attendu {_GRID_INDICES})")


def _build_range(max_indices, max_monthly_complete, max_monthly, min_monthly) -> dict:
    """Pure formatting for GET /range: month bounds as ISO ``YYYY-MM-DD`` strings
    (None-safe — e.g. before the first backfill).

    Two distinct maxima matter because the two marts have different grain:
    ``fct_era5_indices_grid`` only ever contains complete calendar months, while
    ``fct_era5_monthly_grid`` may include a partial current month (flagged via
    ``mois_complet``). Seeding the Climat page's default month or an SPI/STI
    MonthStepper bound from the wrong one lands on a month with no indices at
    all — empty map, misleading 0% drought banner. Raw variables may still step
    into the partial month (``max_monthly_month``), it just needs to be flagged
    per-cell (``mois_complet``) in the UI.
    """
    return {
        "min_month": str(min_monthly) if min_monthly else None,
        "max_indices_month": str(max_indices) if max_indices else None,
        "max_monthly_complete_month": str(max_monthly_complete) if max_monthly_complete else None,
        "max_monthly_month": str(max_monthly) if max_monthly else None,
    }


def _build_daily_points(rows) -> list[dict]:
    """Pure formatting: per-cell ``{latitude, longitude, value}`` rows for one
    date (température ou pluie — le formatage est identique). Empty input yields
    an empty list — mirrors ``/grid-monthly``'s "no rows for this month"
    convention (no 404: with the backfill still running and a J-7 ingestion lag,
    "no data yet for this date" is an expected response, not an error)."""
    return [
        {
            "latitude": _num(r["latitude"]),
            "longitude": _num(r["longitude"]),
            "value": _num(r["value"]),
        }
        for r in rows
    ]


def _build_daily_range(min_date, max_date) -> dict:
    """Pure formatting for daily-layer range endpoints (température ou pluie —
    le formatage est identique): date bounds as ISO ``YYYY-MM-DD`` strings,
    None-safe (e.g. before the nightly ingestion's first row lands)."""
    return {
        "min_date": str(min_date) if min_date else None,
        "max_date": str(max_date) if max_date else None,
    }


def _merge_point_series(monthly_rows, clim_rows, indices_rows) -> list[dict]:
    """Pure merge: monthly grid rows + calendar-month normal (window-1 climatology)
    + SPI/STI for the 4 standard windows, joined on month."""
    normals = {
        int(r["mois_calendaire"]): {"precip": _num(r["precip_moyenne"]), "temp": _num(r["temp_moyenne"])}
        for r in clim_rows
    }

    indices_by_month: dict = defaultdict(dict)
    for r in indices_rows:
        key = str(r["month"])
        fen = int(r["fenetre"])
        indices_by_month[key][f"spi_{fen}"] = _num(r["spi"])
        indices_by_month[key][f"sti_{fen}"] = _num(r["sti"])
        indices_by_month[key][f"spei_{fen}"] = _num(r["spei"])

    series = []
    for r in monthly_rows:
        month_key = str(r["mois"])
        normal = normals.get(r["mois"].month, {})
        entry = {
            "month": month_key,
            "temperature_moyenne": _num(r["temperature_moyenne"]),
            "temperature_min": _num(r["temperature_min"]),
            "temperature_max": _num(r["temperature_max"]),
            "precipitation_totale": _num(r["precipitation_totale"]),
            "etp_totale": _num(r["etp_totale"]),
            "bilan_hydrique": _num(r["bilan_hydrique"]),
            "nb_jours": int(r["nb_jours"]) if r["nb_jours"] is not None else None,
            "mois_complet": bool(r["mois_complet"]) if r["mois_complet"] is not None else None,
            "precipitation_normale": normal.get("precip"),
            "temperature_normale": normal.get("temp"),
        }
        idx = indices_by_month.get(month_key, {})
        for fen in WINDOWS:
            entry[f"spi_{fen}"] = idx.get(f"spi_{fen}")
            entry[f"sti_{fen}"] = idx.get(f"sti_{fen}")
            entry[f"spei_{fen}"] = idx.get(f"spei_{fen}")
        series.append(entry)
    return series


def _build_drought_episodes(spi_rows, monthly_rows, clim_rows) -> list[dict]:
    """Pure gaps-and-islands grouping: consecutive CALENDAR months with SPI < -1

    Le seuil -1.0 est VOLONTAIRE : c'est la définition WMO de l'ÉVÉNEMENT de
    sécheresse (« le SPI est continûment négatif et atteint -1.0 ou moins »). Un
    standard reconnu, à ne pas remplacer par une convention maison.
    form one drought episode.

    Island key = ``month_ordinal - rank`` where ``rank`` is the 1-based position
    of the row once sorted by month among the drought-only rows — the calendar
    equivalent of the SQL idiom
    ``month - ROW_NUMBER() OVER (ORDER BY month) * INTERVAL '1 month'``.
    Consecutive calendar months keep the same key; any missing month (an absent
    row, or a NULL SPI filtered out upstream) shifts the key and therefore
    breaks the episode — un mois manquant casse l'épisode, robuste aux trous de
    série (a mid-series gap no longer silently merges two distinct droughts).

    Args:
        spi_rows: dicts ``{month, spi}`` for one cell/window, ``spi`` may be
            non-drought or already spi-not-null filtered upstream.
        monthly_rows: dicts ``{mois, precipitation_totale}`` for the same cell,
            used to compute the cumulative rainfall deficit per episode.
        clim_rows: dicts ``{mois_calendaire, precip_moyenne}`` (window-1
            climatology), the calendar-month normal subtracted from
            ``precipitation_totale``.
    """
    normal_by_month = {int(r["mois_calendaire"]): _num(r["precip_moyenne"]) for r in clim_rows}
    precip_by_month = {r["mois"]: _num(r["precipitation_totale"]) for r in monthly_rows}

    drought = sorted(
        (
            {"month": r["month"], "spi": float(r["spi"])}
            for r in spi_rows
            if r["spi"] is not None and float(r["spi"]) < -1.0
        ),
        key=lambda r: r["month"],
    )

    groups: dict = defaultdict(list)
    for rank, row in enumerate(drought, start=1):
        month_ordinal = row["month"].year * 12 + row["month"].month
        groups[month_ordinal - rank].append(row)

    episodes = []
    for members in groups.values():
        months = [m["month"] for m in members]
        debut, fin = min(months), max(months)
        deficit_terms = []
        for month, precip in precip_by_month.items():
            if not (debut <= month <= fin):
                continue
            normal = normal_by_month.get(month.month)
            if precip is not None and normal is not None:
                deficit_terms.append(precip - normal)
        episodes.append(
            {
                "debut": str(debut),
                "fin": str(fin),
                "duree_mois": len(members),
                "spi_min": round(min(m["spi"] for m in members), 3),
                "deficit_cumule_mm": round(sum(deficit_terms), 1),
            }
        )
    episodes.sort(key=lambda e: (-e["duree_mois"], e["debut"]))
    return episodes


def _merge_compare_years(monthly_rows, clim_rows, spi_rows, year_list) -> dict:
    """Pure merge: per-year cumulative monthly rainfall (vs. normal) + 3-month SPI series."""
    normal_by_month = {int(r["mois_calendaire"]): _num(r["precip_moyenne"]) or 0.0 for r in clim_rows}

    by_year: dict = defaultdict(lambda: {"cumul_mensuel": [], "spi_3": []})
    cumul_by_year: dict = defaultdict(float)
    cumul_normal_by_year: dict = defaultdict(float)
    for r in monthly_rows:
        yr = r["mois"].year
        mo = r["mois"].month
        precip = _num(r["precipitation_totale"]) or 0.0
        cumul_by_year[yr] += precip
        cumul_normal_by_year[yr] += normal_by_month.get(mo, 0.0)
        by_year[yr]["cumul_mensuel"].append(
            {
                "mois": mo,
                "precipitation": round(precip, 1),
                "cumul": round(cumul_by_year[yr], 1),
                "cumul_normal": round(cumul_normal_by_year[yr], 1),
            }
        )
    for r in spi_rows:
        yr = r["month"].year
        by_year[yr]["spi_3"].append({"mois": r["month"].month, "spi": _num(r["spi"])})

    return {yr: by_year.get(yr, {"cumul_mensuel": [], "spi_3": []}) for yr in year_list}


@router.get("/range")
def get_climat_range():
    """Month bounds for the Climat page (24h cache). See ``_build_range`` for why
    the indices max and the raw-monthly max are reported separately instead of
    reusing ``/era5/range`` (daily-grid based, reflects the partial current
    month)."""

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            max_indices = conn.execute(
                text("SELECT max(month) FROM gold.fct_era5_indices_grid")
            ).scalar()
            max_monthly_complete = conn.execute(
                text("SELECT max(mois) FROM gold.fct_era5_monthly_grid WHERE mois_complet = true")
            ).scalar()
            max_monthly = conn.execute(
                text("SELECT max(mois) FROM gold.fct_era5_monthly_grid")
            ).scalar()
            min_monthly = conn.execute(
                text("SELECT min(mois) FROM gold.fct_era5_monthly_grid")
            ).scalar()
        return _build_range(max_indices, max_monthly_complete, max_monthly, min_monthly)

    return get_cached("obs_climat_range", {}, GRID_TTL, fetch)


@router.get("/grid-monthly")
def get_grid_monthly(
    month: str = Query(..., description="Mois au format YYYY-MM"),
    variable: str = Query(
        "temperature",
        description="temperature|temperature_min|temperature_max|precipitation|etp|bilan_hydrique",
    ),
):
    """Per-cell value of one monthly variable from ``fct_era5_monthly_grid``."""
    if variable not in _MONTHLY_VARIABLES:
        raise HTTPException(
            422, f"Variable inconnue : {variable!r} (attendu {sorted(_MONTHLY_VARIABLES)})"
        )
    month_start = _parse_month(month)
    column = _MONTHLY_VARIABLES[variable]

    def fetch():
        query = f"""
            SELECT era5_latitude, era5_longitude, {column} AS value, mois_complet
            FROM gold.fct_era5_monthly_grid
            WHERE mois = :month
        """
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"month": month_start}).mappings().all()
        return [
            {
                "latitude": _num(r["era5_latitude"]),
                "longitude": _num(r["era5_longitude"]),
                "value": _num(r["value"]),
                "mois_complet": bool(r["mois_complet"]) if r["mois_complet"] is not None else None,
            }
            for r in rows
        ]

    return get_cached(
        "obs_climat_grid_monthly", {"month": str(month_start), "variable": variable}, GRID_TTL, fetch
    )


@router.get("/grid-indices")
def get_grid_indices(
    month: str = Query(..., description="Mois au format YYYY-MM"),
    window: int = Query(3, description="Fenêtre en mois (1, 3, 6 ou 12)"),
    index: str = Query("spi", description="spi, sti ou spei"),
):
    """Per-cell SPI, STI or SPEI value from ``fct_era5_indices_grid`` for the given month/window."""
    _validate_window(window)
    _assert_index(index)
    month_start = _parse_month(month)

    def fetch():
        query = f"""
            SELECT era5_latitude, era5_longitude, {index} AS value
            FROM gold.fct_era5_indices_grid
            WHERE month = :month AND fenetre = :window
        """
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"month": month_start, "window": window}).mappings().all()
        out = []
        for r in rows:
            if r["value"] is None:
                continue
            v = float(r["value"])
            out.append(
                {
                    "latitude": _num(r["era5_latitude"]),
                    "longitude": _num(r["era5_longitude"]),
                    index: v,
                    "index_class": classify_index(v),
                }
            )
        return out

    return get_cached(
        "obs_climat_grid_indices",
        {"month": str(month_start), "window": window, "index": index},
        GRID_TTL,
        fetch,
    )


@router.get("/daily-temp")
def get_daily_temp(
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
    variable: str = Query(..., description="tmax|tmin|tmean"),
):
    """Per-cell TRUE daily temperature statistic (ERA5, computed over 24 hourly
    steps — not the 00h UTC instant used by the monthly/index marts) from
    ``silver.stg_era5_daily_temp_stats`` for one exact date. No rows for the date
    -> empty list, same convention as ``/grid-monthly`` (not a 404: coverage is
    partial while the 1950-2025 backfill runs)."""
    if variable not in _DAILY_TEMP_VARIABLES:
        raise HTTPException(
            422, f"Variable inconnue : {variable!r} (attendu {sorted(_DAILY_TEMP_VARIABLES)})"
        )
    day = _parse_date(date)
    column = _DAILY_TEMP_VARIABLES[variable]

    def fetch():
        query = f"""
            SELECT latitude, longitude, {column} AS value
            FROM silver.stg_era5_daily_temp_stats
            -- jamais de fonction/cast sur la colonne de partition (time) : casse
            -- l'exclusion de chunks TimescaleDB (cf. CLAUDE.md du repo entrepôt)
            WHERE time >= :day AND time < CAST(:day AS date) + INTERVAL '1 day'
        """
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"day": day}).mappings().all()
        return _build_daily_points(rows)

    return get_cached("obs_climat_daily_temp", {"date": str(day), "variable": variable}, GRID_TTL, fetch)


@router.get("/daily-temp-range")
def get_daily_temp_range():
    """Date bounds for the daily-temperature layer (1h cache — the window moves
    daily as the J-7 nightly ingestion advances, unlike the monthly ``/range``'s
    24h cache which only moves once a month)."""

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            # min(time)::date, jamais min(time::date) : caster la colonne de
            # partition empêche l'exclusion de chunks TimescaleDB (même
            # anti-pattern que /daily-precip-range, douze lignes plus bas).
            # Équivalent car date() est monotone non décroissante : min(dates) ==
            # date(min(time)), idem pour max. Mesuré sur cette table
            # (stg_era5_daily_temp_stats) : ~18-19 s -> ~0,2-0,6 s.
            min_date = conn.execute(
                text("SELECT min(time)::date FROM silver.stg_era5_daily_temp_stats")
            ).scalar()
            max_date = conn.execute(
                text("SELECT max(time)::date FROM silver.stg_era5_daily_temp_stats")
            ).scalar()
        return _build_daily_range(min_date, max_date)

    return get_cached("obs_climat_daily_temp_range", {}, DAILY_TEMP_RANGE_TTL, fetch)


# SQL sorti en constante pour être testable : deux régressions coûteuses s'y
# cachent (la source silver vs bronze, et l'absence de cast sur `time`), et un
# test qui lit la chaîne les attrape sans toucher l'entrepôt.
_DAILY_PRECIP_SQL = """
    SELECT latitude, longitude, total_precipitation AS value
    FROM silver.stg_era5_timeseries
    -- jamais de fonction/cast sur la colonne de partition (time) : casse
    -- l'exclusion de chunks TimescaleDB. Mesuré sur cette table (321 M lignes) :
    -- ce prédicat = 413 ms, un cast sur la colonne elle-meme = 69 032 ms (x167).
    -- silver, jamais l'autre couche : celle-ci porte 22 985 mailles au lieu de
    -- 11 496 (doublons flottants ERA5) et serait désalignée de la carte.
    WHERE time >= :day AND time < CAST(:day AS date) + INTERVAL '1 day'
"""


@router.get("/daily-precip")
def get_daily_precip(
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
):
    """Cumul de précipitation journalier par maille (ERA5, mm) depuis
    ``silver.stg_era5_timeseries`` pour une date exacte. Aucune ligne pour la date
    -> liste vide, même convention que ``/daily-temp`` et ``/grid-monthly`` (pas de
    404 : la couverture avance avec l'ingestion, "pas encore de données" est une
    réponse attendue)."""
    day = _parse_date(date)

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(_DAILY_PRECIP_SQL), {"day": day}).mappings().all()
        return _build_daily_points(rows)

    return get_cached("obs_climat_daily_precip", {"date": str(day)}, GRID_TTL, fetch)


@router.get("/daily-precip-range")
def get_daily_precip_range():
    """Bornes de dates de la couche pluie journalière. Endpoint distinct de
    ``/daily-temp-range`` parce que les couvertures DIVERGENT (mesuré : température
    -> 2026-07-10, pluie -> 2026-07-12) : réutiliser celle de la température
    masquerait les jours de pluie les plus récents.

    ``min(time)::date`` et non ``min(time::date)`` : caster la colonne empêche
    d'utiliser l'index/les métadonnées de chunk — mesuré 224 ms contre 3 036 ms."""

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            min_date = conn.execute(
                text("SELECT min(time)::date FROM silver.stg_era5_timeseries")
            ).scalar()
            max_date = conn.execute(
                text("SELECT max(time)::date FROM silver.stg_era5_timeseries")
            ).scalar()
        return _build_daily_range(min_date, max_date)

    return get_cached("obs_climat_daily_precip_range", {}, DAILY_TEMP_RANGE_TTL, fetch)


@router.get("/point-series")
def get_point_series(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    from_month: Optional[str] = Query(None, alias="from", description="Mois de début YYYY-MM"),
    to_month: Optional[str] = Query(None, alias="to", description="Mois de fin YYYY-MM"),
):
    """Full monthly series for the nearest grid cell (rounded to 0.1°): monthly
    variables + calendar-month normal (window-1 climatology) + SPI/STI for the 4
    standard windows."""
    cell_lat, cell_lon = _round_cell(lat, lon)
    from_start = _parse_month(from_month) if from_month else None
    to_start = _parse_month(to_month) if to_month else None

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT 1 FROM gold.fct_era5_monthly_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon LIMIT 1"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).first()
            if exists is None:
                return None

            monthly_query = (
                "SELECT mois, temperature_moyenne, temperature_min, temperature_max,"
                " precipitation_totale, etp_totale, bilan_hydrique, nb_jours, mois_complet"
                " FROM gold.fct_era5_monthly_grid"
                " WHERE era5_latitude = :lat AND era5_longitude = :lon"
            )
            m_params: dict = {"lat": cell_lat, "lon": cell_lon}
            if from_start is not None:
                monthly_query += " AND mois >= :from_month"
                m_params["from_month"] = from_start
            if to_start is not None:
                monthly_query += " AND mois <= :to_month"
                m_params["to_month"] = to_start
            monthly_query += " ORDER BY mois"
            monthly_rows = conn.execute(text(monthly_query), m_params).mappings().all()

            clim_rows = conn.execute(
                text(
                    "SELECT mois_calendaire, precip_moyenne, temp_moyenne"
                    " FROM gold.fct_era5_climatology_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon AND fenetre = 1"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).mappings().all()

            indices_query = (
                "SELECT month, fenetre, spi, sti, spei FROM gold.fct_era5_indices_grid"
                " WHERE era5_latitude = :lat AND era5_longitude = :lon"
            )
            i_params: dict = {"lat": cell_lat, "lon": cell_lon}
            if from_start is not None:
                indices_query += " AND month >= :from_month"
                i_params["from_month"] = from_start
            if to_start is not None:
                indices_query += " AND month <= :to_month"
                i_params["to_month"] = to_start
            indices_rows = conn.execute(text(indices_query), i_params).mappings().all()

        series = _merge_point_series(monthly_rows, clim_rows, indices_rows)
        return {"cell": {"latitude": cell_lat, "longitude": cell_lon}, "series": series}

    result = get_cached(
        "obs_climat_point_series",
        {
            "lat": cell_lat,
            "lon": cell_lon,
            "from": str(from_start) if from_start else None,
            "to": str(to_start) if to_start else None,
        },
        GRID_TTL,
        fetch,
    )
    if result is None:
        raise HTTPException(404, f"Cellule climatique introuvable pour lat={lat}, lon={lon}")
    return result


@router.get("/point-episodes")
def get_point_episodes(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    window: int = Query(3, description="Fenêtre en mois (1, 3, 6 ou 12)"),
):
    """Drought episodes at the nearest cell: consecutive CALENDAR months with
    SPI < -1 for the given window. The SQL below only fetches raw rows —
    grouping is CALENDAR-aware gaps-and-islands done in ``_build_drought_episodes``
    (see its docstring for the island-keying idiom), so a missing month never
    silently merges two distinct episodes."""
    _validate_window(window)
    cell_lat, cell_lon = _round_cell(lat, lon)

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT 1 FROM gold.fct_era5_indices_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon AND fenetre = :window LIMIT 1"
                ),
                {"lat": cell_lat, "lon": cell_lon, "window": window},
            ).first()
            if exists is None:
                return None
            spi_rows = conn.execute(
                text(
                    "SELECT month, spi FROM gold.fct_era5_indices_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon"
                    " AND fenetre = :window AND spi IS NOT NULL ORDER BY month"
                ),
                {"lat": cell_lat, "lon": cell_lon, "window": window},
            ).mappings().all()
            monthly_rows = conn.execute(
                text(
                    "SELECT mois, precipitation_totale FROM gold.fct_era5_monthly_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).mappings().all()
            clim_rows = conn.execute(
                text(
                    "SELECT mois_calendaire, precip_moyenne FROM gold.fct_era5_climatology_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon AND fenetre = 1"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).mappings().all()
        return _build_drought_episodes(spi_rows, monthly_rows, clim_rows)

    result = get_cached(
        "obs_climat_point_episodes", {"lat": cell_lat, "lon": cell_lon, "window": window}, GRID_TTL, fetch
    )
    if result is None:
        raise HTTPException(404, f"Cellule climatique introuvable pour lat={lat}, lon={lon}")
    return result


@router.get("/compare-years")
def get_compare_years(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    years: str = Query(..., description="Années séparées par des virgules, ex: 1976,2003,2026"),
):
    """Per-year monthly rainfall cumulative sum (vs. the calendar-month normal) and
    3-month SPI series, for the requested years at the nearest grid cell."""
    cell_lat, cell_lon = _round_cell(lat, lon)
    try:
        year_list = sorted({int(y.strip()) for y in years.split(",") if y.strip()})
    except ValueError:
        raise HTTPException(422, f"Années invalides : {years!r} (attendu une liste séparée par des virgules)")
    if not year_list:
        raise HTTPException(422, "Au moins une année est requise")
    if len(year_list) > 15:
        raise HTTPException(422, "Trop d'années demandées (max 15)")

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT 1 FROM gold.fct_era5_monthly_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon LIMIT 1"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).first()
            if exists is None:
                return None
            monthly_rows = conn.execute(
                text(
                    "SELECT mois, precipitation_totale FROM gold.fct_era5_monthly_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon"
                    " AND EXTRACT(YEAR FROM mois) = ANY(:years) ORDER BY mois"
                ),
                {"lat": cell_lat, "lon": cell_lon, "years": year_list},
            ).mappings().all()
            clim_rows = conn.execute(
                text(
                    "SELECT mois_calendaire, precip_moyenne FROM gold.fct_era5_climatology_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon AND fenetre = 1"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).mappings().all()
            spi_rows = conn.execute(
                text(
                    "SELECT month, spi FROM gold.fct_era5_indices_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon AND fenetre = 3"
                    " AND EXTRACT(YEAR FROM month) = ANY(:years) ORDER BY month"
                ),
                {"lat": cell_lat, "lon": cell_lon, "years": year_list},
            ).mappings().all()

        years_data = _merge_compare_years(monthly_rows, clim_rows, spi_rows, year_list)
        return {
            "cell": {"latitude": cell_lat, "longitude": cell_lon},
            "years": {str(yr): data for yr, data in years_data.items()},
        }

    result = get_cached(
        "obs_climat_compare_years", {"lat": cell_lat, "lon": cell_lon, "years": year_list}, GRID_TTL, fetch
    )
    if result is None:
        raise HTTPException(404, f"Cellule climatique introuvable pour lat={lat}, lon={lon}")
    return result


@router.get("/export-point.csv")
def export_point_csv(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
):
    """CSV export (monthly variables + SPI/STI for the 4 windows) for the nearest cell."""
    cell_lat, cell_lon = _round_cell(lat, lon)

    def fetch():
        engine = get_brgm_sync_engine()
        with engine.connect() as conn:
            monthly_rows = conn.execute(
                text(
                    "SELECT mois, temperature_moyenne, temperature_min, temperature_max,"
                    " precipitation_totale, etp_totale, bilan_hydrique, nb_jours, mois_complet"
                    " FROM gold.fct_era5_monthly_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon ORDER BY mois"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).mappings().all()
            if not monthly_rows:
                return None
            indices_rows = conn.execute(
                text(
                    "SELECT month, fenetre, spi, sti, spei FROM gold.fct_era5_indices_grid"
                    " WHERE era5_latitude = :lat AND era5_longitude = :lon"
                ),
                {"lat": cell_lat, "lon": cell_lon},
            ).mappings().all()

        indices_by_month: dict = defaultdict(dict)
        for r in indices_rows:
            key = str(r["month"])
            fen = int(r["fenetre"])
            indices_by_month[key][f"spi_{fen}"] = _num(r["spi"])
            indices_by_month[key][f"sti_{fen}"] = _num(r["sti"])
            indices_by_month[key][f"spei_{fen}"] = _num(r["spei"])

        buf = io.StringIO()
        fieldnames = (
            [
                "mois", "temperature_moyenne", "temperature_min", "temperature_max",
                "precipitation_totale", "etp_totale", "bilan_hydrique", "nb_jours", "mois_complet",
            ]
            + [f"spi_{w}" for w in WINDOWS]
            + [f"sti_{w}" for w in WINDOWS]
            + [f"spei_{w}" for w in WINDOWS]
        )
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in monthly_rows:
            month_key = str(r["mois"])
            row = {
                "mois": month_key,
                "temperature_moyenne": _num(r["temperature_moyenne"]),
                "temperature_min": _num(r["temperature_min"]),
                "temperature_max": _num(r["temperature_max"]),
                "precipitation_totale": _num(r["precipitation_totale"]),
                "etp_totale": _num(r["etp_totale"]),
                "bilan_hydrique": _num(r["bilan_hydrique"]),
                "nb_jours": r["nb_jours"],
                "mois_complet": r["mois_complet"],
            }
            idx = indices_by_month.get(month_key, {})
            for w in WINDOWS:
                row[f"spi_{w}"] = idx.get(f"spi_{w}")
                row[f"sti_{w}"] = idx.get(f"sti_{w}")
                row[f"spei_{w}"] = idx.get(f"spei_{w}")
            writer.writerow(row)
        return buf.getvalue()

    body = get_cached("obs_climat_export_point", {"lat": cell_lat, "lon": cell_lon}, GRID_TTL, fetch)
    if body is None:
        raise HTTPException(404, f"Cellule climatique introuvable pour lat={lat}, lon={lon}")
    fname = f"climat_{cell_lat}_{cell_lon}.csv"
    return Response(
        # utf-8-sig adds a BOM so Excel decodes accents correctly (not Windows-1252).
        content=body.encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
