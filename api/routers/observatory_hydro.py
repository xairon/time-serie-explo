"""Observatory hydro router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from api.database import get_brgm_sync_engine
from sqlalchemy.exc import ProgrammingError

from api.config import settings
from api.schemas.observatory import (
    HydroDaily,
    HydroMonthly,
    HydroPercentiles,
    HydroSiteSiblings,
    HydroSPI,
    HydroSSFI,
    HydroStation,
    HydroYearly,
)
from dashboard.utils.cache import get_cached
from dashboard.utils.drought import compute_spi, _classify
from dashboard.utils.reference import value_to_zscore, class_bounds_ngf

router = APIRouter(prefix="/api/v1/observatory/hydro", tags=["observatory-hydro"])

LIST_TTL = 3600
DETAIL_TTL = 3600
DAILY_TTL = 21600
MONTHLY_TTL = 43200
YEARLY_TTL = 86400
PERCENTILES_TTL = 86400
SSFI_TTL = 86400
SIBLINGS_TTL = 3600

# BRGM gold tier stores raw Hub'Eau QmnJ values in L/s with sentinels for missing data
# (e.g. -4e6, -1.5e5, 1e9). Frontend and downstream consumers expect m³/s. Convert at this
# boundary so the rest of the platform sees consistent SI units. Heights (H) would pass
# through unchanged but no H stations exist in the current dataset.
_QMNJ_MAX_VALID = 1e8
_QMNJ_MIN_VALID = -1e4

_FLOW_COLS_DIM = (
    "resultat_moyen_global", "resultat_min_global", "resultat_max_global",
    "resultat_stddev_global", "resultat_moyen_dern_annee",
)
_FLOW_COLS_DAILY = ("resultat_obs_elab",)
_FLOW_COLS_MONTHLY = (
    "resultat_moyen", "resultat_min", "resultat_max", "amplitude_mensuelle",
    "resultat_moy_mobile_3m", "resultat_moy_mobile_12m",
    "variation_resultat_vs_mois_prec", "variation_resultat_vs_annee_prec",
)
_FLOW_COLS_YEARLY = (
    "resultat_moyen_annuel", "resultat_min_annuel", "resultat_max_annuel", "amplitude_annuelle",
)
_FLOW_COLS_PERCENTILES = ("p10", "p25", "p75", "p90")


def _qmnj_to_m3_s(value):
    """Convert Hub'Eau QmnJ (L/s) to m³/s, mapping sentinels to None."""
    if value is None:
        return None
    v = float(value)
    if v >= _QMNJ_MAX_VALID or v <= _QMNJ_MIN_VALID:
        return None
    return v / 1000.0


def _convert_qmnj_row(row: dict, columns) -> dict:
    for col in columns:
        if col in row:
            row[col] = _qmnj_to_m3_s(row[col])
    return row

def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


# ---------------------------------------------------------------------------
# GET /stations
# ---------------------------------------------------------------------------

@router.get("/stations", response_model=list[HydroStation])
def list_stations(
    min_observations: Optional[int] = Query(None, ge=0),
    last_measurement_after: Optional[date] = Query(None),
    code_departement: Optional[str] = Query(None, min_length=1, max_length=3),
    grandeur_hydro: Optional[str] = Query(None),
    bbox: Optional[str] = Query(None, description="min_lon,min_lat,max_lon,max_lat"),
    search: Optional[str] = Query(None, min_length=2, max_length=100),
):
    params = {
        "min_observations": min_observations,
        "last_measurement_after": last_measurement_after,
        "code_departement": code_departement,
        "grandeur_hydro": grandeur_hydro,
        "bbox": bbox,
        "search": search,
    }

    def fetch():
        conditions = ["1=1"]
        bind: dict = {}

        if min_observations is not None:
            conditions.append("nb_jours_total >= :min_obs")
            bind["min_obs"] = min_observations
        if last_measurement_after is not None:
            conditions.append("derniere_mesure >= :last_after")
            bind["last_after"] = last_measurement_after
        if code_departement is not None:
            conditions.append("code_departement = :dept")
            bind["dept"] = code_departement
        if grandeur_hydro is not None:
            conditions.append("grandeur_hydro_principale = :grandeur_hydro")
            bind["grandeur_hydro"] = grandeur_hydro
        if bbox is not None:
            try:
                parts = bbox.split(",")
                if len(parts) != 4:
                    raise ValueError
                min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
            except ValueError:
                raise HTTPException(400, "Format bbox invalide")
            conditions.append("latitude_station BETWEEN :min_lat AND :max_lat")
            conditions.append("longitude_station BETWEEN :min_lon AND :max_lon")
            bind.update(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        if search is not None:
            conditions.append("(code_station ILIKE :search OR libelle_station ILIKE :search OR nom_cours_eau ILIKE :search)")
            bind["search"] = f"%{search}%"

        where = " AND ".join(conditions)
        query = f"""
            SELECT code_station, libelle_station, code_site, libelle_site,
                   code_cours_eau, nom_cours_eau, code_departement, nom_departement,
                   date_ouverture_station, longitude_station, latitude_station,
                   grandeur_hydro_principale, premiere_mesure, derniere_mesure,
                   nb_jours_total, nb_mois_total, resultat_moyen_global,
                   resultat_min_global, resultat_max_global, resultat_stddev_global,
                   annee_dernier_bilan, resultat_moyen_dern_annee
            FROM gold.dim_hydro_stations
            WHERE {where}
            ORDER BY code_station
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

        for r in rows:
            _convert_qmnj_row(r, _FLOW_COLS_DIM)
        return rows

    return get_cached("obs_hydro_list", params, LIST_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/percentiles
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/percentiles", response_model=HydroPercentiles)
def get_percentiles(code_station: str):
    def fetch():
        # Match the in-Python sentinel filter (_qmnj_to_m3_s) so percentiles
        # are not skewed by Hub'Eau placeholders like -4e6 or 1e9 L/s.
        query = """
            SELECT
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY resultat_obs_elab) AS p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY resultat_obs_elab) AS p25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY resultat_obs_elab) AS p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY resultat_obs_elab) AS p90
            FROM gold.hydro_daily_chroniques
            WHERE code_station = :code
              AND resultat_obs_elab IS NOT NULL
              AND resultat_obs_elab > :min_valid
              AND resultat_obs_elab < :max_valid
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"code": code_station, "min_valid": _QMNJ_MIN_VALID, "max_valid": _QMNJ_MAX_VALID},
                )
                row = result.mappings().first()
        finally:
            pass  # shared pooled engine; do not dispose
        if not row or row["p10"] is None:
            raise HTTPException(404, f"Aucune donnée pour la station hydrométrique {code_station}")
        return _convert_qmnj_row(dict(row), _FLOW_COLS_PERCENTILES)

    return get_cached("obs_hydro_pctl", {"code_station": code_station}, PERCENTILES_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/daily
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/daily", response_model=list[HydroDaily])
def get_daily(
    code_station: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(3650, ge=1, le=36500),
):
    params = {"code_station": code_station, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT date, resultat_obs_elab, grandeur_hydro_elab,
                   temperature_2m, total_precipitation, potential_evaporation
            FROM gold.hydro_daily_chroniques
            WHERE code_station = :code
        """
        bind: dict = {"code": code_station}
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
                        text("SELECT 1 FROM gold.dim_hydro_stations WHERE code_station = :code"),
                        {"code": code_station},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
        finally:
            pass  # shared pooled engine; do not dispose
        for r in rows:
            if r.get("grandeur_hydro_elab") != "H":
                _convert_qmnj_row(r, _FLOW_COLS_DAILY)
        return rows

    return get_cached("obs_hydro_daily", params, DAILY_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/monthly
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/monthly", response_model=list[HydroMonthly])
def get_monthly(
    code_station: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(600, ge=1, le=1200),
):
    params = {"code_station": code_station, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT mois, resultat_moyen, resultat_min, resultat_max, amplitude_mensuelle,
                   temperature_moyenne, precipitation_totale, evaporation_moyenne,
                   nb_jours_mesures, resultat_moy_mobile_3m, resultat_moy_mobile_12m,
                   precipitation_moy_mobile_12m, variation_resultat_vs_mois_prec,
                   variation_resultat_vs_annee_prec
            FROM gold.fct_monthly_hydro
            WHERE code_station = :code
        """
        bind: dict = {"code": code_station}
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
                        text("SELECT 1 FROM gold.dim_hydro_stations WHERE code_station = :code"),
                        {"code": code_station},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
        finally:
            pass  # shared pooled engine; do not dispose
        for r in rows:
            _convert_qmnj_row(r, _FLOW_COLS_MONTHLY)
        return rows

    return get_cached("obs_hydro_monthly", params, MONTHLY_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/yearly
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/yearly", response_model=list[HydroYearly])
def get_yearly(
    code_station: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=200),
):
    params = {"code_station": code_station, "start_date": start_date, "end_date": end_date, "limit": limit}

    def fetch():
        query = """
            SELECT annee, resultat_moyen_annuel, resultat_min_annuel, resultat_max_annuel,
                   amplitude_annuelle, temperature_moyenne_annuelle,
                   precipitation_totale_annuelle, nb_jours_mesures_annuel,
                   percentile_resultat_historique, classification_resultat_annuel
            FROM gold.fct_yearly_hydro
            WHERE code_station = :code
        """
        bind: dict = {"code": code_station}
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
                        text("SELECT 1 FROM gold.dim_hydro_stations WHERE code_station = :code"),
                        {"code": code_station},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
        finally:
            pass  # shared pooled engine; do not dispose
        for r in rows:
            _convert_qmnj_row(r, _FLOW_COLS_YEARLY)
        return rows

    return get_cached("obs_hydro_yearly", params, YEARLY_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/ssfi
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/ssfi", response_model=list[HydroSSFI])
def get_ssfi(code_station: str):
    """Compute SSFI from fixed reference grid (gold.station_reference_stats)."""

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                # 1. Load monthly flow series (positive only; sentinels already filtered)
                result = conn.execute(
                    text(
                        "SELECT mois, resultat_moyen FROM gold.fct_monthly_hydro"
                        " WHERE code_station = :code AND resultat_moyen IS NOT NULL"
                        " AND resultat_moyen > 0 AND resultat_moyen < :max_v ORDER BY mois"
                    ),
                    {"code": code_station, "max_v": _QMNJ_MAX_VALID},
                )
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_hydro_stations WHERE code_station = :code"),
                        {"code": code_station},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
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
                        " WHERE type='hydro' AND code=:code"
                    ),
                    {"code": code_station},
                )
                for r in ref_result.mappings():
                    g = r["quantile_grid"]
                    if isinstance(g, str):
                        import json
                        g = json.loads(g)
                    # The warehouse stores the grid in raw QmnJ (L/s); the series
                    # value is converted to m³/s below, so convert the grid too —
                    # otherwise every month floors to the same z-score (constant SSFI).
                    if g is not None:
                        g = [v / 1000.0 for v in g]
                    grid_by_month[int(r["month"])] = g
        except ProgrammingError:
            pass  # Table not yet created (pre-materialization)
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
            val_raw = r["resultat_moyen"]
            val = _qmnj_to_m3_s(val_raw)
            if val is None:
                continue
            m = mois_dt.month
            z = value_to_zscore(val, grid_by_month.get(m))
            out.append({
                "mois": mois_dt.strftime("%Y-%m-%d"),
                "value": round(val, 4),
                "ssfi": z,
                "classification": _classify(z),
            })
        return out

    return get_cached("obs_hydro_ssfi", {"code_station": code_station}, SSFI_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/spi
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/spi", response_model=list[HydroSPI])
def get_spi(code_station: str):
    """Compute Standardized Precipitation Index (SPI) from monthly precipitation."""

    def fetch():
        query = """
            SELECT mois, precipitation_totale
            FROM gold.fct_monthly_hydro
            WHERE code_station = :code AND precipitation_totale IS NOT NULL
            ORDER BY mois
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_station})
                rows = [dict(r._mapping) for r in result]
                if not rows:
                    exists = conn.execute(
                        text("SELECT 1 FROM gold.dim_hydro_stations WHERE code_station = :code"),
                        {"code": code_station},
                    ).first()
                    if exists is None:
                        raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
                    return []
        finally:
            pass  # shared pooled engine; do not dispose

        months = [str(r["mois"]) for r in rows]
        values = [float(r["precipitation_totale"]) if r["precipitation_totale"] is not None else None for r in rows]
        return compute_spi(months, values)

    return get_cached("obs_hydro_spi", {"code_station": code_station}, SSFI_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/siblings
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/siblings", response_model=HydroSiteSiblings)
def get_siblings(code_station: str, level: str = Query("site", pattern="^(site|cours_eau)$")):
    """Other hydro stations at the same hydrometric site or on the same watercourse."""

    def fetch():
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT code_site, libelle_site, code_cours_eau, nom_cours_eau "
                        "FROM gold.dim_hydro_stations WHERE code_station = :code"
                    ),
                    {"code": code_station},
                ).mappings().first()
                if not row:
                    raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")

                if level == "cours_eau":
                    group_val = row["code_cours_eau"]
                    if not group_val:
                        raise HTTPException(404, f"Aucun cours d'eau pour la station {code_station}")
                    where = "code_cours_eau = :grp AND code_station != :code"
                else:
                    group_val = row["code_site"]
                    if not group_val:
                        raise HTTPException(404, f"Aucun code de site pour la station {code_station}")
                    where = "code_site = :grp AND code_station != :code"

                query = f"""
                    SELECT s.code_station, s.libelle_station, s.grandeur_hydro_principale,
                           sci.index_class AS classification, s.derniere_mesure
                    FROM gold.dim_hydro_stations s
                    LEFT JOIN gold.station_current_index sci
                      ON sci.type = 'hydro' AND sci.code = s.code_station
                    WHERE {where}
                    ORDER BY s.code_station
                    LIMIT 50
                """
                result = conn.execute(text(query), {"grp": group_val, "code": code_station})
                siblings = [dict(r._mapping) for r in result]
        finally:
            pass  # shared pooled engine; do not dispose

        return {
            "code_site": row["code_site"],
            "libelle_site": row["libelle_site"],
            "nom_cours_eau": row["nom_cours_eau"],
            "nb_stations": len(siblings) + 1,
            "level": level,
            "siblings": [
                {
                    "code_station": s["code_station"],
                    "libelle_station": s.get("libelle_station"),
                    "grandeur_hydro_principale": s.get("grandeur_hydro_principale"),
                    "classification": s.get("classification"),
                    "derniere_mesure": s.get("derniere_mesure"),
                }
                for s in siblings
            ],
        }

    return get_cached(
        "obs_hydro_siblings",
        {"code_station": code_station, "level": level},
        SIBLINGS_TTL,
        fetch,
    )


# ---------------------------------------------------------------------------
# GET /stations/{code_station}
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}", response_model=HydroStation)
def get_station(code_station: str):
    def fetch():
        # Main station query — threshold computation moved to a second query against
        # gold.station_reference_stats instead of the old LATERAL percentile_cont subquery.
        query = """
            SELECT s.code_station, s.libelle_station, s.code_site, s.libelle_site,
                   s.code_cours_eau, s.nom_cours_eau, s.code_departement, s.nom_departement,
                   s.date_ouverture_station, s.longitude_station, s.latitude_station,
                   s.grandeur_hydro_principale, s.premiere_mesure, s.derniere_mesure,
                   s.nb_jours_total, s.nb_mois_total, s.resultat_moyen_global,
                   s.resultat_min_global, s.resultat_max_global, s.resultat_stddev_global,
                   s.annee_dernier_bilan, s.resultat_moyen_dern_annee,
                   sci.index_name, sci.index_value, sci.index_class,
                   sci.ref_month AS index_ref_month,
                   sci.baseline_start AS index_baseline_start,
                   sci.baseline_end AS index_baseline_end,
                   lm.ref_value AS index_ref_value,
                   lm.month_median AS index_month_median
            FROM gold.dim_hydro_stations s
            LEFT JOIN gold.station_current_index sci ON sci.type = 'hydro' AND sci.code = s.code_station
            LEFT JOIN LATERAL (
                SELECT m.resultat_moyen AS ref_value,
                       (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY m2.resultat_moyen)
                        FROM gold.fct_monthly_hydro m2
                        WHERE m2.code_station = s.code_station
                          AND m2.grandeur_hydro_elab = s.grandeur_hydro_principale
                          AND m2.resultat_moyen IS NOT NULL AND m2.resultat_moyen < 1e8
                          AND EXTRACT(MONTH FROM m2.mois) = EXTRACT(MONTH FROM m.mois)) AS month_median
                FROM gold.fct_monthly_hydro m
                WHERE m.code_station = s.code_station
                  AND m.grandeur_hydro_elab = s.grandeur_hydro_principale
                  AND m.resultat_moyen IS NOT NULL AND m.resultat_moyen < 1e8
                ORDER BY m.mois DESC LIMIT 1
            ) lm ON true
            WHERE s.code_station = :code
        """
        engine = get_brgm_sync_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_station})
                row = result.mappings().first()
                if not row:
                    raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
                out = _convert_qmnj_row(
                    dict(row),
                    _FLOW_COLS_DIM + ("index_ref_value", "index_month_median"),
                )

        finally:
            pass  # shared pooled engine; do not dispose

        # Fetch fixed reference grid in a separate connection so a missing table
        # (pre-materialization) doesn't poison the main query connection.
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
                            " WHERE type='hydro' AND code=:code AND month=:month"
                        ),
                        {"code": code_station, "month": ref_m},
                    ).mappings().first()
                    if ref_row is not None:
                        reference_flag = ref_row["flag"]
                        g = ref_row["quantile_grid"]
                        if isinstance(g, str):
                            import json
                            g = json.loads(g)
                        raw_bounds = class_bounds_ngf(g)
                        # Convert bounds from L/s warehouse units to m³/s
                        if raw_bounds is not None:
                            index_class_bounds = [_qmnj_to_m3_s(v) for v in raw_bounds]
            except ProgrammingError:
                pass  # Table not yet created (pre-materialization)
            finally:
                pass  # shared pooled engine; do not dispose

        out["reference_flag"] = reference_flag
        out["index_class_bounds"] = index_class_bounds
        # Keep backward-compat: threshold_values still populated from class bounds
        out["index_threshold_values"] = index_class_bounds
        return out

    return get_cached("obs_hydro_detail", {"code_station": code_station}, DETAIL_TTL, fetch)


