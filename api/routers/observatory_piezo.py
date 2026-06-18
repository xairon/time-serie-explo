"""Observatory piezo router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import text
from api.database import get_brgm_sync_engine
from sqlalchemy.exc import ProgrammingError
from dashboard.utils.station_export import build_station_csv

from api.config import settings
from api.schemas.observatory import (
    PiezoBdlisaSiblings,
    PiezoDaily,
    PiezoMonthly,
    PiezoPercentiles,
    PiezoSPI,
    PiezoSPLI,
    PiezoStation,
    PiezoYearly,
)
from dashboard.utils.cache import get_cached
from dashboard.utils.drought import compute_spi, _classify
from dashboard.utils.reference import value_to_zscore, class_bounds_ngf

router = APIRouter(prefix="/api/v1/observatory/piezo", tags=["observatory-piezo"])

LIST_TTL = 3600
DETAIL_TTL = 3600
DAILY_TTL = 21600
MONTHLY_TTL = 43200
YEARLY_TTL = 86400
PERCENTILES_TTL = 86400
SPLI_TTL = 86400
SIBLINGS_TTL = 3600

_BDLISA_SYSTEM_RE = re.compile(r"^(\d{3}[A-Z]{2})")


def _bdlisa_primary(codes_bdlisa: str | None) -> str | None:
    """Return the primary (first) BDLISA entity code from a possibly comma-joined string."""
    if not codes_bdlisa:
        return None
    first = codes_bdlisa.split(",")[0].strip()
    return first or None


def _bdlisa_system_prefix(codes_bdlisa: str | None) -> str | None:
    """Return the BDLISA system-level prefix (3 digits + 2 letters) of the primary code.

    '101AC01' -> '101AC' ; '101AC' -> '101AC' ; None/'' -> None.
    Falls back to the primary code unchanged if it doesn't match the BDLISA shape.
    """
    primary = _bdlisa_primary(codes_bdlisa)
    if not primary:
        return None
    m = _BDLISA_SYSTEM_RE.match(primary)
    return m.group(1) if m else primary


def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


# ---------------------------------------------------------------------------
# GET /stations
# ---------------------------------------------------------------------------

@router.get("/stations", response_model=list[PiezoStation])
def list_stations(
    min_observations: Optional[int] = Query(None, ge=0),
    last_measurement_after: Optional[date] = Query(None),
    code_departement: Optional[str] = Query(None, min_length=1, max_length=3),
    bbox: Optional[str] = Query(None, description="min_lon,min_lat,max_lon,max_lat"),
    search: Optional[str] = Query(None, min_length=2, max_length=100),
):
    params = {
        "min_observations": min_observations,
        "last_measurement_after": last_measurement_after,
        "code_departement": code_departement,
        "bbox": bbox,
        "search": search,
    }

    def fetch():
        conditions = ["1=1"]
        bind: dict = {}

        if min_observations is not None:
            conditions.append("nb_mesures_total >= :min_obs")
            bind["min_obs"] = min_observations
        if last_measurement_after is not None:
            conditions.append("derniere_mesure >= :last_after")
            bind["last_after"] = last_measurement_after
        if code_departement is not None:
            conditions.append("code_departement = :dept")
            bind["dept"] = code_departement
        if bbox is not None:
            try:
                parts = bbox.split(",")
                if len(parts) != 4:
                    raise ValueError
                min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
            except ValueError:
                raise HTTPException(400, "Format bbox invalide")
            conditions.append("latitude BETWEEN :min_lat AND :max_lat")
            conditions.append("longitude BETWEEN :min_lon AND :max_lon")
            bind.update(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        if search is not None:
            conditions.append("(code_bss ILIKE :search OR nom_commune ILIKE :search)")
            bind["search"] = f"%{search}%"

        where = " AND ".join(conditions)
        query = f"""
            SELECT code_bss, bss_id, latitude, longitude, nom_commune,
                   code_departement, nom_departement, codes_bdlisa,
                   altitude_station, date_debut_mesure, date_fin_mesure,
                   nb_mesures_total, nb_mois_total, premiere_mesure, derniere_mesure,
                   niveau_moyen_global, niveau_min_absolu, niveau_max_absolu,
                   niveau_stddev_global, amplitude_totale, profondeur_moyenne_globale,
                   temperature_moyenne_globale, precipitation_moyenne_mensuelle,
                   derniere_annee, niveau_derniere_annee,
                   slope_precipitation, niveau_alerte
            FROM gold.dim_piezo_stations
            WHERE {where}
            ORDER BY code_bss
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(row._mapping) for row in result]
        finally:
            pass  # shared pooled engine; do not dispose

        return rows

    return get_cached("obs_piezo_list", params, LIST_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_bss}
# ---------------------------------------------------------------------------

@router.get("/stations/{code_bss:path}/percentiles", response_model=PiezoPercentiles)
def get_percentiles(code_bss: str):
    def fetch():
        query = """
            SELECT
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY niveau_nappe_eau) AS p90
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = :code AND niveau_nappe_eau IS NOT NULL
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                row = result.mappings().first()
        finally:
            pass  # shared pooled engine; do not dispose
        if not row or row["p10"] is None:
            raise HTTPException(404, f"Aucune donnée pour la station piézométrique {code_bss}")
        return dict(row)

    return get_cached("obs_piezo_pctl", {"code_bss": code_bss}, PERCENTILES_TTL, fetch)


@router.get("/stations/{code_bss:path}/daily", response_model=list[PiezoDaily])
def get_daily(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(3650, ge=1, le=36500),
):
    params = {"code_bss": code_bss, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT date, niveau_nappe_eau, profondeur_nappe,
                   temperature_2m, total_precipitation, potential_evaporation
            FROM gold.hubeau_daily_chroniques
            WHERE code_bss = :code
        """
        bind: dict = {"code": code_bss}
        if start_date is not None:
            query += " AND date >= :start_date"
            bind["start_date"] = start_date
        if end_date is not None:
            query += " AND date <= :end_date"
            bind["end_date"] = end_date
        query += " ORDER BY date LIMIT :limit"
        bind["limit"] = limit

        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
        finally:
            pass  # shared pooled engine; do not dispose
        return rows

    return get_cached("obs_piezo_daily", params, DAILY_TTL, fetch)


@router.get("/stations/{code_bss:path}/monthly", response_model=list[PiezoMonthly])
def get_monthly(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(600, ge=1, le=1200),
):
    params = {"code_bss": code_bss, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT mois, niveau_moyen, niveau_min, niveau_max, amplitude_mensuelle,
                   temperature_moyenne, precipitation_totale, evaporation_moyenne,
                   nb_jours_mesures, niveau_moy_mobile_3m, niveau_moy_mobile_12m,
                   precipitation_moy_mobile_12m, variation_niveau_vs_mois_prec,
                   variation_niveau_vs_annee_prec
            FROM gold.fct_monthly_chroniques
            WHERE code_bss = :code
        """
        bind: dict = {"code": code_bss}
        if start_date is not None:
            query += " AND mois >= :start_date"
            bind["start_date"] = start_date
        if end_date is not None:
            query += " AND mois <= :end_date"
            bind["end_date"] = end_date
        query += " ORDER BY mois LIMIT :limit"
        bind["limit"] = limit

        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
        finally:
            pass  # shared pooled engine; do not dispose
        return rows

    return get_cached("obs_piezo_monthly", params, MONTHLY_TTL, fetch)


@router.get("/stations/{code_bss:path}/yearly", response_model=list[PiezoYearly])
def get_yearly(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=200),
):
    params = {"code_bss": code_bss, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT annee, niveau_moyen_annuel, niveau_min_annuel, niveau_max_annuel,
                   amplitude_annuelle, temperature_moyenne_annuelle,
                   precipitation_totale_annuelle, bilan_hydrique_annuel,
                   nb_jours_mesures_annuel, percentile_niveau_historique,
                   classification_niveau_annuel, niveau_moy_mobile_5ans
            FROM gold.fct_yearly_stats
            WHERE code_bss = :code
        """
        bind: dict = {"code": code_bss}
        if start_date is not None:
            query += " AND annee >= :start_year"
            bind["start_year"] = start_date.year
        if end_date is not None:
            query += " AND annee <= :end_year"
            bind["end_year"] = end_date.year
        query += " ORDER BY annee LIMIT :limit"
        bind["limit"] = limit

        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
        finally:
            pass  # shared pooled engine; do not dispose
        return rows

    return get_cached("obs_piezo_yearly", params, YEARLY_TTL, fetch)


@router.get("/stations/{code_bss:path}/spli", response_model=list[PiezoSPLI])
def get_spli(code_bss: str):
    """Compute SPLI (IPS) from fixed reference grid (gold.station_reference_stats)."""

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                # 1. Load monthly level series
                result = conn.execute(
                    text(
                        "SELECT mois, niveau_moyen FROM gold.fct_monthly_chroniques"
                        " WHERE code_bss = :code AND niveau_moyen IS NOT NULL ORDER BY mois"
                    ),
                    {"code": code_bss},
                )
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
                    return []

        finally:
            pass  # shared pooled engine; do not dispose

        # 2. Load fixed reference grid in a separate connection so a missing table
        #    (pre-materialization) doesn't poison the main query connection.
        grid_by_month: dict[int, list[float] | None] = {}
        engine2 = get_brgm_sync_engine()
        try:
            with engine2.connect() as conn2:
                ref_result = conn2.execute(
                    text(
                        "SELECT month, quantile_grid FROM gold.station_reference_stats"
                        " WHERE type='piezo' AND code=:code"
                    ),
                    {"code": code_bss},
                )
                for r in ref_result.mappings():
                    g = r["quantile_grid"]
                    if isinstance(g, str):
                        import json
                        g = json.loads(g)
                    grid_by_month[int(r["month"])] = g
        except ProgrammingError:
            # Table not yet created (pre-materialization) — return empty series
            pass
        finally:
            pass  # shared pooled engine; do not dispose

        # If no reference grid exists yet, return empty (table not yet materialized)
        if not grid_by_month:
            return []

        # 3. Compute z-score and classification per monthly value
        import pandas as pd
        out = []
        for r in rows:
            mois_dt = pd.to_datetime(str(r["mois"]))
            val = float(r["niveau_moyen"])
            m = mois_dt.month
            z = value_to_zscore(val, grid_by_month.get(m))
            out.append({
                "mois": mois_dt.strftime("%Y-%m-%d"),
                "value": round(val, 4),
                "spli": z,
                "classification": _classify(z),
            })
        return out

    return get_cached("obs_piezo_spli", {"code_bss": code_bss}, SPLI_TTL, fetch)


@router.get("/stations/{code_bss:path}/spi", response_model=list[PiezoSPI])
def get_spi(code_bss: str):
    """Compute Standardized Precipitation Index (SPI) from monthly precipitation."""

    def fetch():
        query = """
            SELECT mois, precipitation_totale
            FROM gold.fct_monthly_chroniques
            WHERE code_bss = :code AND precipitation_totale IS NOT NULL
            ORDER BY mois
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                        {"code": code_bss},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
                    return []
        finally:
            pass  # shared pooled engine; do not dispose

        months = [str(r["mois"]) for r in rows]
        values = [float(r["precipitation_totale"]) if r["precipitation_totale"] is not None else None for r in rows]
        return compute_spi(months, values)

    return get_cached("obs_piezo_spi", {"code_bss": code_bss}, SPLI_TTL, fetch)


@router.get("/stations/{code_bss:path}/siblings", response_model=PiezoBdlisaSiblings)
def get_siblings(code_bss: str, level: str = Query("nappe", pattern="^(nappe|systeme)$")):
    """Other piezometers in the same BDLISA entity (nappe) or system (préfixe)."""

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT codes_bdlisa FROM gold.dim_piezo_stations WHERE code_bss = :code"),
                    {"code": code_bss},
                ).mappings().first()
                if row is None:
                    raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")

                codes_bdlisa = row["codes_bdlisa"]
                primary = _bdlisa_primary(codes_bdlisa)
                if not primary:
                    return {
                        "level": level,
                        "code_bdlisa": None,
                        "non_rattachee": True,
                        "nb_stations": 1,
                        "siblings": [],
                    }

                if level == "systeme":
                    match = _bdlisa_system_prefix(codes_bdlisa)
                    where = "s.codes_bdlisa LIKE :pat AND s.code_bss != :code"
                    params = {"pat": f"{match}%", "code": code_bss}
                else:
                    match = primary
                    where = "s.codes_bdlisa = :pat AND s.code_bss != :code"
                    params = {"pat": match, "code": code_bss}

                query = f"""
                    SELECT s.code_bss, s.nom_commune, s.codes_bdlisa,
                           sci.index_class AS classification
                    FROM gold.dim_piezo_stations s
                    LEFT JOIN gold.station_current_index sci
                      ON sci.type = 'piezo' AND sci.code = s.code_bss
                    WHERE {where}
                    ORDER BY s.code_bss
                    LIMIT 50
                """
                result = conn.execute(text(query), params)
                siblings = [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

        return {
            "level": level,
            "code_bdlisa": match,
            "non_rattachee": False,
            "nb_stations": len(siblings) + 1,
            "siblings": [
                {
                    "code_bss": s["code_bss"],
                    "nom_commune": s.get("nom_commune"),
                    "codes_bdlisa": s.get("codes_bdlisa"),
                    "classification": s.get("classification"),
                }
                for s in siblings
            ],
        }

    return get_cached(
        "obs_piezo_siblings",
        {"code_bss": code_bss, "level": level},
        SIBLINGS_TTL,
        fetch,
    )


@router.get("/stations/{code_bss:path}/export.csv")
def export_csv(
    code_bss: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
):
    """Export station metadata + daily chronique + monthly IPS as a CSV file.

    Optional start_date / end_date bound the daily chronique (empty = full history).
    """
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        meta = conn.execute(
            text(
                "SELECT code_bss AS code, nom_commune, code_departement,"
                " nom_departement, codes_bdlisa, latitude, longitude"
                " FROM gold.dim_piezo_stations WHERE code_bss = :code"
            ),
            {"code": code_bss},
        ).mappings().first()
        if meta is None:
            raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
        daily_query = (
            "SELECT date, niveau_nappe_eau, profondeur_nappe, temperature_2m,"
            " total_precipitation, potential_evaporation"
            " FROM gold.hubeau_daily_chroniques WHERE code_bss = :code"
        )
        daily_bind: dict = {"code": code_bss}
        if start_date is not None:
            daily_query += " AND date >= :start_date"
            daily_bind["start_date"] = start_date
        if end_date is not None:
            daily_query += " AND date <= :end_date"
            daily_bind["end_date"] = end_date
        daily_query += " ORDER BY date"
        daily = [dict(r) for r in conn.execute(text(daily_query), daily_bind).mappings()]

    index_rows: list[dict] = []
    engine2 = get_brgm_sync_engine()
    try:
        with engine2.connect() as conn2:
            index_rows = [
                dict(r) for r in conn2.execute(
                    text(
                        "SELECT month, z, index_class, flag FROM gold.fct_monthly_index"
                        " WHERE type = 'piezo' AND code = :code ORDER BY month"
                    ),
                    {"code": code_bss},
                ).mappings()
            ]
    except ProgrammingError:
        index_rows = []  # table not yet materialized

    body = build_station_csv(
        "piezo", {**dict(meta), "generated_on": date.today().isoformat()}, daily, index_rows
    )
    fname = f"{code_bss.replace('/', '_')}_{date.today().isoformat()}.csv"
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/stations/{code_bss:path}", response_model=PiezoStation)
def get_station(code_bss: str):
    def fetch():
        # Main station query — keeps all existing fields; threshold computation
        # is done via a second query against gold.station_reference_stats instead
        # of the old LATERAL percentile_cont subquery.
        query = """
            SELECT s.code_bss, s.bss_id, s.latitude, s.longitude, s.nom_commune,
                   s.code_departement, s.nom_departement, s.codes_bdlisa,
                   s.altitude_station, s.date_debut_mesure, s.date_fin_mesure,
                   s.nb_mesures_total, s.nb_mois_total, s.premiere_mesure, s.derniere_mesure,
                   s.niveau_moyen_global, s.niveau_min_absolu, s.niveau_max_absolu,
                   s.niveau_stddev_global, s.amplitude_totale, s.profondeur_moyenne_globale,
                   s.temperature_moyenne_globale, s.precipitation_moyenne_mensuelle,
                   s.derniere_annee, s.niveau_derniere_annee,
                   s.slope_precipitation, s.niveau_alerte,
                   sci.index_name, sci.index_value, sci.index_class,
                   sci.ref_month AS index_ref_month,
                   sci.baseline_start AS index_baseline_start,
                   sci.baseline_end AS index_baseline_end,
                   lm.ref_value AS index_ref_value,
                   lm.month_median AS index_month_median
            FROM gold.dim_piezo_stations s
            LEFT JOIN gold.station_current_index sci ON sci.type = 'piezo' AND sci.code = s.code_bss
            LEFT JOIN LATERAL (
                SELECT m.niveau_moyen AS ref_value,
                       (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY m2.niveau_moyen)
                        FROM gold.fct_monthly_chroniques m2
                        WHERE m2.code_bss = s.code_bss AND m2.niveau_moyen IS NOT NULL
                          AND EXTRACT(MONTH FROM m2.mois) = EXTRACT(MONTH FROM m.mois)) AS month_median
                FROM gold.fct_monthly_chroniques m
                WHERE m.code_bss = s.code_bss AND m.niveau_moyen IS NOT NULL
                ORDER BY m.mois DESC LIMIT 1
            ) lm ON true
            WHERE s.code_bss = :code
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_bss})
                row = result.mappings().first()
                if not row:
                    raise HTTPException(404, f"Station piézométrique {code_bss} introuvable")
                out = dict(row)

        finally:
            pass  # shared pooled engine; do not dispose

        # Fetch fixed reference grid for the reference calendar month in a separate
        # connection so a missing table (pre-materialization) doesn't poison the above.
        import pandas as pd
        reference_flag = None
        index_class_bounds = None
        ref_month_val = out.get("index_ref_month")
        ref_m = pd.to_datetime(str(ref_month_val)).month if ref_month_val is not None else None
        if ref_m is not None:
            engine2 = get_brgm_sync_engine()
            try:
                with engine2.connect() as conn2:
                    ref_row = conn2.execute(
                        text(
                            "SELECT quantile_grid, flag FROM gold.station_reference_stats"
                            " WHERE type='piezo' AND code=:code AND month=:month"
                        ),
                        {"code": code_bss, "month": ref_m},
                    ).mappings().first()
                    if ref_row is not None:
                        reference_flag = ref_row["flag"]
                        g = ref_row["quantile_grid"]
                        if isinstance(g, str):
                            import json
                            g = json.loads(g)
                        index_class_bounds = class_bounds_ngf(g)
            except ProgrammingError:
                pass  # Table not yet created (pre-materialization)
            finally:
                pass  # shared pooled engine; do not dispose

        out["reference_flag"] = reference_flag
        out["index_class_bounds"] = index_class_bounds
        # Keep backward-compat: threshold_values still populated from class bounds
        out["index_threshold_values"] = index_class_bounds
        return out

    return get_cached("obs_piezo_detail", {"code_bss": code_bss}, DETAIL_TTL, fetch)
