"""Observatory common router — GeoJSON, alerts, national/department stats, timeline."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from api.config import settings
from api.schemas.observatory import Alert, DepartmentStats, NationalStats
from dashboard.utils.cache import get_cached

router = APIRouter(prefix="/api/v1/observatory", tags=["observatory-common"])

GEOJSON_TTL = 3600
ALERTS_TTL = 3600
STATS_TTL = 21600
TIMELINE_TTL = 86400

SeverityType = Literal[
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT"
]


def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )


# ---------------------------------------------------------------------------
# GET /stations/geojson
# ---------------------------------------------------------------------------

@router.get("/stations/geojson")
def get_stations_geojson(
    type: Optional[Literal["piezo", "hydro", "all"]] = Query("all"),
):
    params = {"type": type}

    def fetch():
        features = []
        want_piezo = type in (None, "all", "piezo")
        want_hydro = type in (None, "all", "hydro")

        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                if want_piezo:
                    piezo_result = conn.execute(text("""
                        SELECT code_bss AS code, 'piezo' AS type,
                               latitude, longitude, nom_commune AS commune,
                               code_departement, nom_departement AS departement,
                               classification_derniere_annee AS classification,
                               codes_bdlisa, derniere_mesure,
                               nb_mesures_total, nb_mois_total
                        FROM gold.dim_piezo_stations
                        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                    """))
                    for row in piezo_result.mappings():
                        r = dict(row)
                        nb_mois = int(r["nb_mois_total"]) if r.get("nb_mois_total") else 0
                        fiab = "fiable" if nb_mois >= 120 else ("indicatif" if nb_mois >= 60 else "insuffisant")
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
                            "properties": {
                                "code": r["code"], "type": r["type"],
                                "classification": r["classification"],
                                "commune": r["commune"], "departement": r["departement"],
                                "code_departement": r["code_departement"],
                                "codes_bdlisa": r.get("codes_bdlisa"),
                                "derniere_mesure": str(r["derniere_mesure"]) if r["derniere_mesure"] else None,
                                "nb_observations": r["nb_mesures_total"],
                                "fiabilite": fiab,
                            },
                        })

                if want_hydro:
                    hydro_result = conn.execute(text("""
                        SELECT code_station AS code, 'hydro' AS type,
                               latitude_station AS latitude, longitude_station AS longitude,
                               libelle_station AS commune,
                               code_departement, nom_departement AS departement,
                               classification_resultat_dern_annee AS classification,
                               LEFT(code_cours_eau, 1) AS code_district, code_site, derniere_mesure,
                               nb_jours_total, nb_mois_total
                        FROM gold.dim_hydro_stations
                        WHERE latitude_station IS NOT NULL AND longitude_station IS NOT NULL
                    """))
                    for row in hydro_result.mappings():
                        r = dict(row)
                        nb_mois = int(r["nb_mois_total"]) if r.get("nb_mois_total") else 0
                        fiab = "fiable" if nb_mois >= 120 else ("indicatif" if nb_mois >= 60 else "insuffisant")
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
                            "properties": {
                                "code": r["code"], "type": r["type"],
                                "classification": r["classification"],
                                "commune": r["commune"], "departement": r["departement"],
                                "code_departement": r["code_departement"],
                                "code_district": r.get("code_district"),
                                "code_site": r.get("code_site"),
                                "derniere_mesure": str(r["derniere_mesure"]) if r["derniere_mesure"] else None,
                                "nb_observations": r["nb_jours_total"],
                                "fiabilite": fiab,
                            },
                        })
        finally:
            engine.dispose()

        return {"type": "FeatureCollection", "features": features}

    return get_cached("obs_stations_geojson", params, GEOJSON_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /alerts
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=list[Alert])
def list_alerts(
    severity: Optional[list[SeverityType]] = Query(None),
    type: Optional[Literal["piezo", "hydro"]] = Query(None),
    code_departement: Optional[str] = Query(None, min_length=1, max_length=3),
    active_only: bool = Query(True),
):
    severity_list = severity if severity is not None else ["TRES_BAS", "TRES_HAUT"]
    params = {
        "severity": severity_list,
        "type": type,
        "code_departement": code_departement,
        "active_only": active_only,
    }

    def fetch():
        bind: dict = {}
        recent_cutoff = date.today() - timedelta(days=90)
        parts = []

        if type is None or type == "piezo":
            conds = ["1=1"]
            conds.append("s.classification_derniere_annee = ANY(:severity)")
            bind["severity"] = severity_list
            if code_departement:
                conds.append("s.code_departement = :dept")
            if active_only:
                conds.append("s.derniere_mesure >= :recent_cutoff")
                bind["recent_cutoff"] = recent_cutoff

            parts.append(f"""
                SELECT s.code_bss AS code, 'piezo' AS type,
                       s.latitude, s.longitude,
                       s.nom_commune AS commune, s.code_departement, s.nom_departement AS departement,
                       s.classification_derniere_annee AS classification, s.derniere_mesure,
                       cs.alerte_depuis_annee, cs.nb_annees_consecutives
                FROM gold.dim_piezo_stations s
                LEFT JOIN LATERAL (
                    SELECT min(y.annee) AS alerte_depuis_annee,
                           count(*) AS nb_annees_consecutives
                    FROM (
                        SELECT annee, classification_niveau_annuel,
                               annee - ROW_NUMBER() OVER (ORDER BY annee) AS grp
                        FROM gold.fct_yearly_stats
                        WHERE code_bss = s.code_bss
                          AND classification_niveau_annuel IN ('TRES_BAS', 'BAS')
                    ) y
                    WHERE y.grp = (
                        SELECT annee - ROW_NUMBER() OVER (ORDER BY annee)
                        FROM gold.fct_yearly_stats
                        WHERE code_bss = s.code_bss
                          AND classification_niveau_annuel IN ('TRES_BAS', 'BAS')
                        ORDER BY annee DESC LIMIT 1
                    )
                ) cs ON true
                WHERE {" AND ".join(conds)}
            """)

        if type is None or type == "hydro":
            conds = ["1=1"]
            if "severity" not in bind:
                bind["severity"] = severity_list
            conds.append("s.classification_resultat_dern_annee = ANY(:severity)")
            if code_departement:
                conds.append("s.code_departement = :dept")
            if active_only:
                conds.append("s.derniere_mesure >= :recent_cutoff")
                bind["recent_cutoff"] = recent_cutoff

            parts.append(f"""
                SELECT s.code_station AS code, 'hydro' AS type,
                       s.latitude_station AS latitude, s.longitude_station AS longitude,
                       s.libelle_station AS commune, s.code_departement, s.nom_departement AS departement,
                       s.classification_resultat_dern_annee AS classification, s.derniere_mesure,
                       cs.alerte_depuis_annee, cs.nb_annees_consecutives
                FROM gold.dim_hydro_stations s
                LEFT JOIN LATERAL (
                    SELECT min(y.annee) AS alerte_depuis_annee,
                           count(*) AS nb_annees_consecutives
                    FROM (
                        SELECT annee, classification_resultat_annuel,
                               annee - ROW_NUMBER() OVER (ORDER BY annee) AS grp
                        FROM gold.fct_yearly_hydro
                        WHERE code_station = s.code_station
                          AND classification_resultat_annuel IN ('TRES_BAS', 'BAS')
                    ) y
                    WHERE y.grp = (
                        SELECT annee - ROW_NUMBER() OVER (ORDER BY annee)
                        FROM gold.fct_yearly_hydro
                        WHERE code_station = s.code_station
                          AND classification_resultat_annuel IN ('TRES_BAS', 'BAS')
                        ORDER BY annee DESC LIMIT 1
                    )
                ) cs ON true
                WHERE {" AND ".join(conds)}
            """)

        if not parts:
            return []

        if code_departement:
            bind["dept"] = code_departement

        union = " UNION ALL ".join(parts)
        query = f"{union} ORDER BY classification, code"

        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), bind)
                return [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

    return get_cached("obs_alerts", params, ALERTS_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stats/national
# ---------------------------------------------------------------------------

@router.get("/stats/national", response_model=NationalStats)
def get_national_stats():
    def fetch():
        recent_cutoff = date.today() - timedelta(days=90)
        query = """
            WITH piezo AS (
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE classification_derniere_annee = 'EXTREMEMENT_BAS') AS extremement_bas,
                       count(*) FILTER (WHERE classification_derniere_annee = 'TRES_BAS') AS tres_bas,
                       count(*) FILTER (WHERE classification_derniere_annee = 'BAS') AS bas,
                       count(*) FILTER (WHERE classification_derniere_annee = 'NORMAL') AS normal,
                       count(*) FILTER (WHERE classification_derniere_annee = 'HAUT') AS haut,
                       count(*) FILTER (WHERE classification_derniere_annee = 'TRES_HAUT') AS tres_haut,
                       count(*) FILTER (WHERE classification_derniere_annee = 'EXTREMEMENT_HAUT') AS extremement_haut,
                       count(*) FILTER (WHERE classification_derniere_annee IS NULL) AS no_class
                FROM gold.dim_piezo_stations
                WHERE derniere_mesure >= :recent_cutoff
            ),
            hydro AS (
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'EXTREMEMENT_BAS') AS extremement_bas,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'TRES_BAS') AS tres_bas,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'BAS') AS bas,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'NORMAL') AS normal,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'HAUT') AS haut,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'TRES_HAUT') AS tres_haut,
                       count(*) FILTER (WHERE classification_resultat_dern_annee = 'EXTREMEMENT_HAUT') AS extremement_haut
                FROM gold.dim_hydro_stations
                WHERE derniere_mesure >= :recent_cutoff
            )
            SELECT p.total AS total_piezo,
                   p.extremement_bas AS piezo_extremement_bas,
                   p.tres_bas AS piezo_tres_bas, p.bas AS piezo_bas,
                   p.normal AS piezo_normal, p.haut AS piezo_haut,
                   p.tres_haut AS piezo_tres_haut,
                   p.extremement_haut AS piezo_extremement_haut,
                   p.no_class AS piezo_no_class,
                   h.total AS total_hydro,
                   h.extremement_bas AS hydro_extremement_bas,
                   h.tres_bas AS hydro_tres_bas, h.bas AS hydro_bas,
                   h.normal AS hydro_normal, h.haut AS hydro_haut,
                   h.tres_haut AS hydro_tres_haut,
                   h.extremement_haut AS hydro_extremement_haut
            FROM piezo p CROSS JOIN hydro h
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"recent_cutoff": recent_cutoff})
                return dict(result.mappings().fetchone())
        finally:
            engine.dispose()

    return get_cached("obs_national_stats", {}, STATS_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /stats/departments
# ---------------------------------------------------------------------------

@router.get("/stats/departments", response_model=list[DepartmentStats])
def get_department_stats():
    def fetch():
        recent_cutoff = date.today() - timedelta(days=90)
        query = """
            WITH piezo AS (
                SELECT code_departement, nom_departement,
                       count(*) AS nb_piezo,
                       count(*) FILTER (WHERE classification_derniere_annee IN ('EXTREMEMENT_BAS', 'TRES_BAS')) AS tres_bas
                FROM gold.dim_piezo_stations
                WHERE code_departement IS NOT NULL AND derniere_mesure >= :recent_cutoff
                GROUP BY code_departement, nom_departement
            ),
            hydro AS (
                SELECT code_departement, count(*) AS nb_hydro
                FROM gold.dim_hydro_stations
                WHERE code_departement IS NOT NULL AND derniere_mesure >= :recent_cutoff
                GROUP BY code_departement
            ),
            trends AS (
                SELECT code_departement, avg(variation_annuelle_m) AS avg_variation
                FROM gold.agg_station_trends
                WHERE saison = 'annuel'
                GROUP BY code_departement
            )
            SELECT p.code_departement, p.nom_departement,
                   p.nb_piezo, COALESCE(h.nb_hydro, 0) AS nb_hydro,
                   ROUND(p.tres_bas::numeric / NULLIF(p.nb_piezo, 0) * 100, 1) AS pct_tres_bas,
                   t.avg_variation
            FROM piezo p
            LEFT JOIN hydro h ON p.code_departement = h.code_departement
            LEFT JOIN trends t ON p.code_departement = t.code_departement
            ORDER BY p.code_departement
        """
        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"recent_cutoff": recent_cutoff})
                return [dict(r._mapping) for r in result]
        finally:
            engine.dispose()

    return get_cached("obs_department_stats", {}, STATS_TTL, fetch)


# ---------------------------------------------------------------------------
# GET /classifications/timeline
# ---------------------------------------------------------------------------

@router.get("/classifications/timeline")
def get_classification_timeline():
    """Monthly classification timeline for all stations.

    Uses calendar-month percentile ranking: each month's value is compared
    against all values for the *same calendar month* across all years.

    Returns compact format: periods[] + stations dict with integer arrays.
    Classification codes: 0=EXTREMEMENT_BAS, 1=TRES_BAS, 2=BAS, 3=NORMAL,
    4=HAUT, 5=TRES_HAUT, 6=EXTREMEMENT_HAUT, 7=UNKNOWN.
    """

    def fetch():
        piezo_query = """
            WITH ranked AS (
                SELECT code_bss AS code,
                       TO_CHAR(mois, 'YYYY-MM') AS period,
                       PERCENT_RANK() OVER (
                           PARTITION BY code_bss, EXTRACT(MONTH FROM mois)
                           ORDER BY niveau_moyen
                       ) AS pctile
                FROM gold.fct_monthly_chroniques
                WHERE niveau_moyen IS NOT NULL AND mois >= '2000-01-01'
            )
            SELECT code, period,
                CASE
                    WHEN pctile < 0.05 THEN 0
                    WHEN pctile < 0.10 THEN 1
                    WHEN pctile < 0.25 THEN 2
                    WHEN pctile < 0.75 THEN 3
                    WHEN pctile < 0.90 THEN 4
                    WHEN pctile < 0.95 THEN 5
                    ELSE 6
                END AS cls
            FROM ranked
            ORDER BY code, period
        """
        hydro_query = """
            WITH ranked AS (
                SELECT code_station AS code,
                       TO_CHAR(mois, 'YYYY-MM') AS period,
                       PERCENT_RANK() OVER (
                           PARTITION BY code_station, EXTRACT(MONTH FROM mois)
                           ORDER BY resultat_moyen
                       ) AS pctile
                FROM gold.fct_monthly_hydro
                WHERE resultat_moyen IS NOT NULL AND mois >= '2000-01-01'
            )
            SELECT code, period,
                CASE
                    WHEN pctile < 0.05 THEN 0
                    WHEN pctile < 0.10 THEN 1
                    WHEN pctile < 0.25 THEN 2
                    WHEN pctile < 0.75 THEN 3
                    WHEN pctile < 0.90 THEN 4
                    WHEN pctile < 0.95 THEN 5
                    ELSE 6
                END AS cls
            FROM ranked
            ORDER BY code, period
        """

        periods_set: set[str] = set()
        station_periods: dict[str, dict[str, int]] = {}

        engine = create_engine(_brgm_url())
        try:
            with engine.connect() as conn:
                for row in conn.execute(text(piezo_query)).mappings():
                    periods_set.add(row["period"])
                    station_periods.setdefault(row["code"], {})[row["period"]] = row["cls"]

                for row in conn.execute(text(hydro_query)).mappings():
                    periods_set.add(row["period"])
                    station_periods.setdefault(row["code"], {})[row["period"]] = row["cls"]
        finally:
            engine.dispose()

        periods = sorted(periods_set)

        stations = {
            code: [vals.get(p, 7) for p in periods]
            for code, vals in station_periods.items()
        }

        return {"periods": periods, "stations": stations}

    return get_cached("obs_timeline", {}, TIMELINE_TTL, fetch)
