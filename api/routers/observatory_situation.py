"""Observatory situation router — territory + national 'météo des nappes' verdicts.

Aggregates per-station fixed-reference indices (IPS/SSFI) into a situation class
+ trend per territory, reusing pure helpers in dashboard.utils.territory_situation.
"""
from __future__ import annotations

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
         AND rs.month = EXTRACT(MONTH FROM sci.index_ref_month)::int
        LEFT JOIN gold.fct_monthly_chroniques lag
          ON lag.code_bss = s.code_bss
         AND lag.mois = (date_trunc('month', sci.index_ref_month) - INTERVAL '3 months')::date
        LEFT JOIN gold.station_reference_stats glag
          ON glag.type = 'piezo' AND glag.code = s.code_bss
         AND glag.month = EXTRACT(MONTH FROM (sci.index_ref_month - INTERVAL '3 months'))::int
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
         AND rs.month = EXTRACT(MONTH FROM sci.index_ref_month)::int
        LEFT JOIN gold.fct_monthly_hydro lag
          ON lag.code_station = s.code_station
         AND lag.mois = (date_trunc('month', sci.index_ref_month) - INTERVAL '3 months')::date
        LEFT JOIN gold.station_reference_stats glag
          ON glag.type = 'hydro' AND glag.code = s.code_station
         AND glag.month = EXTRACT(MONTH FROM (sci.index_ref_month - INTERVAL '3 months'))::int
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


def _eligible_rows_to_territories(rows, level, type_) -> list[dict]:
    """rows: iterable of (territory_code, territory_name, z, delta_z, flag).

    Groups by territory_code, excludes provisoire/UNKNOWN from the verdict but
    counts them, and produces a TerritorySituation-shaped dict per territory.
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

    territories = []
    for code, g in groups.items():
        sit = aggregate_situation(g["z"])
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
