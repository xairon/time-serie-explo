"""Observatory hydro router — sync SQLAlchemy against BRGM data warehouse."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

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
from dashboard.utils.drought import compute_spi, compute_ssfi

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

ClassificationType = Literal[
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL", "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT"
]


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
    classification: Optional[list[ClassificationType]] = Query(None),
    code_departement: Optional[str] = Query(None, min_length=1, max_length=3),
    grandeur_hydro: Optional[str] = Query(None),
    bbox: Optional[str] = Query(None, description="min_lon,min_lat,max_lon,max_lat"),
    search: Optional[str] = Query(None, min_length=2, max_length=100),
):
    params = {
        "min_observations": min_observations,
        "last_measurement_after": last_measurement_after,
        "classification": classification,
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
        if classification is not None:
            conditions.append("classification_resultat_dern_annee = ANY(:classification)")
            bind["classification"] = list(classification)
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
                   annee_dernier_bilan, resultat_moyen_dern_annee,
                   classification_resultat_dern_annee, percentile_resultat_dern_annee
            FROM gold.dim_hydro_stations
            WHERE {where}
            ORDER BY code_station
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                rows = [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

        if classification is not None:
            rows = [r for r in rows if r.get("classification_resultat_dern_annee") in classification]

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
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(query),
                    {"code": code_station, "min_valid": _QMNJ_MIN_VALID, "max_valid": _QMNJ_MAX_VALID},
                )
                row = result.mappings().first()
        finally:
            engine.dispose()
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

        engine = create_engine(_brgm_url())
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
            engine.dispose()
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

        engine = create_engine(_brgm_url())
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
            engine.dispose()
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

        engine = create_engine(_brgm_url())
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
            engine.dispose()
        for r in rows:
            _convert_qmnj_row(r, _FLOW_COLS_YEARLY)
        return rows

    return get_cached("obs_hydro_yearly", params, YEARLY_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/ssfi
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/ssfi", response_model=list[HydroSSFI])
def get_ssfi(code_station: str):
    """Compute Standardized Streamflow Index (SSFI) from monthly data."""

    def fetch():
        query = """
            SELECT mois, resultat_moyen
            FROM gold.fct_monthly_hydro
            WHERE code_station = :code AND resultat_moyen IS NOT NULL
            ORDER BY mois
        """
        engine = create_engine(_brgm_url())
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
            engine.dispose()

        months = [str(r["mois"]) for r in rows]
        values = [_qmnj_to_m3_s(r["resultat_moyen"]) for r in rows]
        return compute_ssfi(months, values)

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
        engine = create_engine(_brgm_url())
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
            engine.dispose()

        months = [str(r["mois"]) for r in rows]
        values = [float(r["precipitation_totale"]) if r["precipitation_totale"] is not None else None for r in rows]
        return compute_spi(months, values)

    return get_cached("obs_hydro_spi", {"code_station": code_station}, SSFI_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}/siblings
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}/siblings", response_model=HydroSiteSiblings)
def get_siblings(code_station: str):
    """Return other hydro stations at the same hydrometric site."""

    def fetch():
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT code_site, libelle_site, nom_cours_eau FROM gold.dim_hydro_stations WHERE code_station = :code"),
                    {"code": code_station},
                ).mappings().first()
                if not row:
                    raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
                code_site = row["code_site"]
                if not code_site:
                    raise HTTPException(404, f"Aucun code de site pour la station {code_station}")

                query = """
                    SELECT code_station, libelle_station, grandeur_hydro_principale,
                           classification_resultat_dern_annee, derniere_mesure
                    FROM gold.dim_hydro_stations
                    WHERE code_site = :site AND code_station != :code
                    ORDER BY code_station
                """
                result = conn.execute(text(query), {"site": code_site, "code": code_station})
                siblings = [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

        return {
            "code_site": code_site,
            "libelle_site": row["libelle_site"],
            "nom_cours_eau": row["nom_cours_eau"],
            "nb_stations": len(siblings) + 1,
            "siblings": [
                {
                    "code_station": s["code_station"],
                    "libelle_station": s.get("libelle_station"),
                    "grandeur_hydro_principale": s.get("grandeur_hydro_principale"),
                    "classification": s.get("classification_resultat_dern_annee"),
                    "derniere_mesure": s.get("derniere_mesure"),
                }
                for s in siblings
            ],
        }

    return get_cached("obs_hydro_siblings", {"code_station": code_station}, SIBLINGS_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stations/{code_station}
# ---------------------------------------------------------------------------

@router.get("/stations/{code_station}", response_model=HydroStation)
def get_station(code_station: str):
    def fetch():
        query = """
            SELECT s.code_station, s.libelle_station, s.code_site, s.libelle_site,
                   s.code_cours_eau, s.nom_cours_eau, s.code_departement, s.nom_departement,
                   s.date_ouverture_station, s.longitude_station, s.latitude_station,
                   s.grandeur_hydro_principale, s.premiere_mesure, s.derniere_mesure,
                   s.nb_jours_total, s.nb_mois_total, s.resultat_moyen_global,
                   s.resultat_min_global, s.resultat_max_global, s.resultat_stddev_global,
                   s.annee_dernier_bilan, s.resultat_moyen_dern_annee,
                   s.classification_resultat_dern_annee, s.percentile_resultat_dern_annee,
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
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"code": code_station})
                row = result.mappings().first()
        finally:
            engine.dispose()
        if not row:
            raise HTTPException(404, f"Station hydrométrique {code_station} introuvable")
        return _convert_qmnj_row(dict(row), _FLOW_COLS_DIM + ("index_ref_value", "index_month_median"))

    return get_cached("obs_hydro_detail", {"code_station": code_station}, DETAIL_TTL, fetch)


