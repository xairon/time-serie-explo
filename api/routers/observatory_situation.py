"""Observatory situation router — territory + national 'météo des nappes' verdicts.

Aggregates per-station fixed-reference indices (IPS/SSFI) into a situation class
+ trend per territory, reusing pure helpers in dashboard.utils.territory_situation.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Query
from sqlalchemy import text

from api.data.territories_fr import (
    DEPARTMENT_NAMES, REGION_NAMES, region_of,
)
from api.database import get_brgm_sync_engine
from api.schemas.observatory import NationalSituation, TerritorySituation
from dashboard.utils.cache import get_cached
from dashboard.utils.territory_situation import (
    aggregate_situation, aggregate_trend,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/observatory", tags=["observatory-situation"])

SITUATION_TTL = 21600
RELIABLE_MIN_MOIS = 60

_STATION_SQL = {
    "piezo": """
        SELECT s.code_departement AS dept,
               sci.index_value AS z_latest,
               rs.flag AS flag,
               lag.niveau_moyen AS lag_value,
               glag.quantile_grid AS lag_grid
        FROM gold.dim_piezo_stations s
        JOIN gold.station_current_index sci
          ON sci.type = 'piezo' AND sci.code = s.code_bss
        LEFT JOIN gold.station_reference_stats rs
          ON rs.type = 'piezo' AND rs.code = s.code_bss
         AND rs.month = EXTRACT(MONTH FROM sci.ref_month)::int
        LEFT JOIN gold.fct_monthly_chroniques lag
          ON lag.code_bss = s.code_bss
         AND lag.mois = (date_trunc('month', sci.ref_month) - INTERVAL '3 months')::date
        LEFT JOIN gold.station_reference_stats glag
          ON glag.type = 'piezo' AND glag.code = s.code_bss
         AND glag.month = EXTRACT(MONTH FROM (sci.ref_month - INTERVAL '3 months'))::int
        WHERE s.code_departement IS NOT NULL
          AND s.nb_mois_total >= :min_mois
          AND sci.index_class IS NOT NULL AND sci.index_class <> 'UNKNOWN'
    """,
    "hydro": """
        SELECT s.code_departement AS dept,
               sci.index_value AS z_latest,
               rs.flag AS flag,
               lag.resultat_moyen AS lag_value,
               glag.quantile_grid AS lag_grid
        FROM gold.dim_hydro_stations s
        JOIN gold.station_current_index sci
          ON sci.type = 'hydro' AND sci.code = s.code_station
        LEFT JOIN gold.station_reference_stats rs
          ON rs.type = 'hydro' AND rs.code = s.code_station
         AND rs.month = EXTRACT(MONTH FROM sci.ref_month)::int
        LEFT JOIN gold.fct_monthly_hydro lag
          ON lag.code_station = s.code_station
         AND lag.mois = (date_trunc('month', sci.ref_month) - INTERVAL '3 months')::date
        LEFT JOIN gold.station_reference_stats glag
          ON glag.type = 'hydro' AND glag.code = s.code_station
         AND glag.month = EXTRACT(MONTH FROM (sci.ref_month - INTERVAL '3 months'))::int
        WHERE s.code_departement IS NOT NULL
          AND s.nb_mois_total >= :min_mois
          AND sci.index_class IS NOT NULL AND sci.index_class <> 'UNKNOWN'
    """,
}


def _fetch_station_rows(type_: str) -> list[tuple]:
    """-> list of (dept, z_latest, delta_z, flag) for all candidate stations."""
    from dashboard.utils.reference import value_to_zscore

    engine = get_brgm_sync_engine()
    out: list[tuple] = []
    with engine.connect() as conn:
        result = conn.execute(text(_STATION_SQL[type_]), {"min_mois": RELIABLE_MIN_MOIS})
        for r in result.mappings():
            z_latest = r["z_latest"]
            delta_z = None
            if z_latest is not None and r["lag_value"] is not None and r["lag_grid"]:
                z_lag = value_to_zscore(float(r["lag_value"]), list(r["lag_grid"]))
                if z_lag is not None:
                    delta_z = float(z_latest) - z_lag
            out.append((r["dept"], z_latest, delta_z, r["flag"]))
    return out


_STATION_SQL_WITH_CODE = {
    "piezo": _STATION_SQL["piezo"].replace(
        "SELECT s.code_departement AS dept,",
        "SELECT s.code_bss AS code, s.code_departement AS dept,", 1),
    "hydro": _STATION_SQL["hydro"].replace(
        "SELECT s.code_departement AS dept,",
        "SELECT s.code_station AS code, s.code_departement AS dept,", 1),
}


def _fetch_station_rows_with_code(type_: str) -> list[tuple]:
    """-> list of (code, z_latest, delta_z, flag)."""
    from dashboard.utils.reference import value_to_zscore
    engine = get_brgm_sync_engine()
    out: list[tuple] = []
    with engine.connect() as conn:
        result = conn.execute(text(_STATION_SQL_WITH_CODE[type_]), {"min_mois": RELIABLE_MIN_MOIS})
        for r in result.mappings():
            z_latest = r["z_latest"]
            delta_z = None
            if z_latest is not None and r["lag_value"] is not None and r["lag_grid"]:
                z_lag = value_to_zscore(float(r["lag_value"]), list(r["lag_grid"]))
                if z_lag is not None:
                    delta_z = float(z_latest) - z_lag
            out.append((r["code"], z_latest, delta_z, r["flag"]))
    return out


def _key_rows_by_sector(rows, code_to_sector, meta):
    """rows: (code, z, dz, flag) -> keyed rows (sector_id, nom, z, dz, flag)."""
    keyed = []
    for code, z, dz, flag in rows:
        sid = code_to_sector.get(code)
        if sid is None:
            continue
        keyed.append((sid, meta.get(sid, {}).get("nom") or f"Secteur {sid}", z, dz, flag))
    return keyed


def _eligible_rows_to_territories(rows, level, type_, min_eligible: int | None = None) -> list[dict]:
    """rows: iterable of (territory_code, territory_name, z, delta_z, flag).

    Groups by territory_code, excludes provisoire/UNKNOWN from the verdict but
    counts them, and produces a TerritorySituation-shaped dict per territory.
    min_eligible: seuil de stations pour rendre un verdict (défaut MIN_ELIGIBLE=3 ;
    2 pour le réseau officiel MétéEAU, aligné sur les indicateurs globaux BRGM à 2-20 stations).
    """
    groups: dict[str, dict] = {}
    for code, name, z, delta_z, flag in rows:
        g = groups.setdefault(code, {"name": name, "z": [], "dz": [], "prov": 0})
        if flag in ("normale", "adaptee") and z is not None:
            g["z"].append(float(z))
            if delta_z is not None:
                g["dz"].append(delta_z)
        else:
            g["prov"] += 1

    from dashboard.utils.territory_situation import MIN_ELIGIBLE
    if min_eligible is None:
        min_eligible = MIN_ELIGIBLE

    territories = []
    for code, g in groups.items():
        sit = aggregate_situation(g["z"], min_eligible=min_eligible)
        territories.append({
            "level": level, "code": code, "name": g["name"], "type": type_,
            "situation_class": sit["situation_class"],
            "trend": aggregate_trend(g["dz"]),
            "pct_below_normal": sit["pct_below_normal"],
            "n_eligible": sit["n_eligible"],
            "n_provisoire": g["prov"],
            "distribution": {k: v for k, v in sit["distribution"].items() if v},
            "insufficient": sit["insufficient"],
            "outlook": None,
        })
    return territories


def _fetch_month_rows_with_code(type_: str, month: str) -> list[tuple]:
    """Past-month rows (code, z, delta_z, flag) from gold.fct_monthly_index.

    z = index at `month`; delta_z = z(month) - z(month-3) read from the same table.
    """
    sql = text("""
        WITH cur AS (
            SELECT code, z, flag FROM gold.fct_monthly_index
            WHERE type = :t AND month = date_trunc('month', CAST(:m AS date))
              AND index_class <> 'UNKNOWN'
        ),
        prev AS (
            SELECT code, z FROM gold.fct_monthly_index
            WHERE type = :t AND month = (date_trunc('month', CAST(:m AS date)) - INTERVAL '3 months')
        )
        SELECT cur.code, cur.z AS z, cur.flag,
               (cur.z - prev.z) AS delta_z
        FROM cur LEFT JOIN prev ON prev.code = cur.code
    """)
    engine = get_brgm_sync_engine()
    out = []
    with engine.connect() as conn:
        for r in conn.execute(sql, {"t": type_, "m": f"{month}-01"}).mappings():
            out.append((r["code"], r["z"], r["delta_z"], r["flag"]))
    return out


_CLASS_TO_IDX = {c: i for i, c in enumerate([
    "EXTREMEMENT_BAS", "TRES_BAS", "BAS", "NORMAL", "HAUT", "TRES_HAUT", "EXTREMEMENT_HAUT"])}
_TREND_CODE = {"baisse": -1, "stable": 0, "hausse": 1}

# Réseau officiel MétéEAU Nappes (450 indicateurs ponctuels du bulletin BRGM,
# seed gold.ref_stations_meteeau_bsn) : 431 piézomètres (codes BSS ancien/nouveau)
# + 19 sources karstiques suivies en débit (codes hydro). network=meteeau
# restreint l'agrégation des secteurs à ce réseau pour coller aux cartes BRGM.
_official_codes_cache: set[str] | None = None


def _official_codes() -> set[str] | None:
    """Codes (tous formats) du réseau officiel MétéEAU ; None si le seed est absent."""
    global _official_codes_cache
    if _official_codes_cache is not None:
        return _official_codes_cache
    engine = get_brgm_sync_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT code_bss, code_bss_nouveau FROM gold.ref_stations_meteeau_bsn"
            )).fetchall()
        _official_codes_cache = {c for row in rows for c in row if c}
    except Exception:
        logger.warning("Seed gold.ref_stations_meteeau_bsn indisponible, filtre réseau ignoré")
        return None
    return _official_codes_cache


def _restrict_to_official(rows: list[tuple]) -> list[tuple]:
    """Filtre des lignes (code, ...) sur le réseau officiel ; identité si seed absent."""
    codes = _official_codes()
    if codes is None:
        return rows
    return [r for r in rows if r[0] in codes]


@router.get("/situation/sectors/timeline")
def get_sector_timeline(
    type: Literal["piezo", "hydro"] = Query("piezo"),
    network: Literal["all", "meteeau"] = Query("all"),
):
    from api.services.sector_mapping import get_mapping
    from dashboard.utils.territory_situation import MIN_ELIGIBLE

    def fetch():
        official = _official_codes() if network == "meteeau" else None
        min_eligible = 2 if network == "meteeau" else MIN_ELIGIBLE
        code_to_sector, meta = get_mapping(type)
        sql = text("""
            SELECT code, TO_CHAR(month, 'YYYY-MM') AS period, z, index_class, flag
            FROM gold.fct_monthly_index
            WHERE type = :t AND month >= '2000-01-01'
            ORDER BY code, month
        """)
        engine = get_brgm_sync_engine()
        per: dict[tuple, dict] = {}
        prev_z: dict[str, dict] = {}
        periods_set: set[str] = set()
        with engine.connect() as conn:
            for r in conn.execute(sql, {"t": type}).mappings():
                if official is not None and r["code"] not in official:
                    continue
                sid = code_to_sector.get(r["code"])
                if sid is None:
                    continue
                p = r["period"]
                periods_set.add(p)
                slot = per.setdefault((sid, p), {"z": [], "dz": []})
                if r["flag"] in ("normale", "adaptee") and r["z"] is not None and r["index_class"] != "UNKNOWN":
                    slot["z"].append(float(r["z"]))
                    hist = prev_z.setdefault(r["code"], {})
                    y, m = int(p[:4]), int(p[5:7])
                    pm = m - 3
                    py = y
                    if pm <= 0:
                        pm += 12
                        py -= 1
                    prior = hist.get(f"{py:04d}-{pm:02d}")
                    if prior is not None:
                        slot["dz"].append(float(r["z"]) - prior)
                    hist[p] = float(r["z"])

        periods = sorted(periods_set)
        sectors: dict[str, list[int]] = {}
        trends: dict[str, list[int]] = {}
        sids = {sid for (sid, _p) in per.keys()}
        for sid in sids:
            cls_arr, trd_arr = [], []
            for p in periods:
                slot = per.get((sid, p))
                if not slot or len(slot["z"]) < min_eligible:
                    cls_arr.append(7)
                    trd_arr.append(0)
                    continue
                sit = aggregate_situation(slot["z"], min_eligible=min_eligible)
                cls = sit["situation_class"]
                cls_arr.append(_CLASS_TO_IDX.get(cls, 7) if cls else 7)
                trd_arr.append(_TREND_CODE.get(aggregate_trend(slot["dz"]) or "stable", 0))
            sectors[str(sid)] = cls_arr
            trends[str(sid)] = trd_arr
        return {"periods": periods, "sectors": sectors, "trends": trends}

    return get_cached("obs_sectors_timeline", {"type": type, "network": network}, 86400, fetch)


@router.get("/situation/sectors")
def get_sector_situation(
    type: Literal["piezo", "hydro"] = Query("piezo"),
    month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
    network: Literal["all", "meteeau"] = Query("all"),
):
    from api.services.sector_mapping import get_mapping

    def fetch():
        code_to_sector, meta = get_mapping(type)
        rows = (_fetch_month_rows_with_code(type, month) if month
                else _fetch_station_rows_with_code(type))
        if network == "meteeau":
            rows = _restrict_to_official(rows)
        keyed = _key_rows_by_sector(rows, code_to_sector, meta)
        out = _eligible_rows_to_territories(
            keyed, "sector", type,
            min_eligible=2 if network == "meteeau" else None)
        for t in out:
            sid = t["code"]
            t["code"] = str(sid)
            t["tendancy_coord"] = meta.get(sid, {}).get("tendancy_coord")
        out.sort(key=lambda t: t["name"])
        return out

    return get_cached("obs_situation_sectors", {"type": type, "month": month, "network": network}, SITUATION_TTL, fetch)


@router.get("/situation/territories", response_model=list[TerritorySituation])
def get_territory_situation(
    level: Literal["region", "department"] = Query("region"),
    type: Literal["piezo", "hydro"] = Query("piezo"),
):
    def fetch():
        station_rows = _fetch_station_rows(type)
        keyed = []
        for dept, z, dz, flag in station_rows:
            if level == "region":
                code = region_of(dept)
                if code is None:
                    continue
                name = REGION_NAMES[code]
            else:
                code = dept
                name = DEPARTMENT_NAMES.get(dept, dept)
            keyed.append((code, name, z, dz, flag))
        out = _eligible_rows_to_territories(keyed, level, type)
        out.sort(key=lambda t: t["name"])
        return out

    return get_cached("obs_situation_territories", {"level": level, "type": type}, SITUATION_TTL, fetch)


@router.get("/situation/national", response_model=NationalSituation)
def get_national_situation(type: Literal["piezo", "hydro"] = Query("piezo")):
    def fetch():
        station_rows = _fetch_station_rows(type)
        keyed = [("FR", "France", z, dz, flag) for dept, z, dz, flag in station_rows]
        aggs = _eligible_rows_to_territories(keyed, "region", type)
        if not aggs:
            return {
                "type": type, "situation_class": None, "trend": None,
                "pct_below_normal": None, "n_eligible": 0, "n_provisoire": 0,
                "distribution": {}, "insufficient": True, "outlook": None,
            }
        agg = aggs[0]
        return {
            "type": type,
            "situation_class": agg["situation_class"], "trend": agg["trend"],
            "pct_below_normal": agg["pct_below_normal"], "n_eligible": agg["n_eligible"],
            "n_provisoire": agg["n_provisoire"], "distribution": agg["distribution"],
            "insufficient": agg["insufficient"], "outlook": None,
        }

    return get_cached("obs_situation_national", {"type": type}, SITUATION_TTL, fetch)
