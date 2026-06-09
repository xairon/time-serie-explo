"""Load secteurs-bsh.geojson and map station coords -> sector_id (cached process-wide)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from api.database import get_brgm_sync_engine
from dashboard.utils.geo_sectors import point_in_geometry

# Backend copy of the sectors geojson. The frontend serves its own copy from
# frontend/public/geo/; the backend Docker image ships api/data/ but NOT frontend/,
# so the loader must read the api/data/ copy (both written by build_secteurs_bsh_geojson).
GEOJSON = Path(__file__).resolve().parents[1] / "data" / "secteurs-bsh.geojson"
STATION_SECTORS = Path(__file__).resolve().parents[1] / "data" / "station_sectors.json"


def build_mapping(geojson: dict, stations: list[tuple]):
    """stations: list of (code, lon, lat). Returns (code->sector_id, {sector_id: meta})."""
    feats = geojson["features"]
    meta = {f["properties"]["sector_id"]: {
        "nom": f["properties"].get("nom"),
        "tendancy_coord": f["properties"].get("tendancy_coord"),
    } for f in feats}
    code_to_sector: dict[str, int] = {}
    for code, lon, lat in stations:
        if lon is None or lat is None:
            continue
        for f in feats:
            if point_in_geometry(float(lon), float(lat), f["geometry"]):
                code_to_sector[code] = f["properties"]["sector_id"]
                break
    return code_to_sector, meta


def _load_geojson() -> dict:
    return json.loads(GEOJSON.read_text())


def _load_stations(type_: str) -> list[tuple]:
    if type_ == "piezo":
        sql = "SELECT code_bss AS code, longitude AS lon, latitude AS lat FROM gold.dim_piezo_stations WHERE longitude IS NOT NULL"
    else:
        sql = "SELECT code_station AS code, longitude_station AS lon, latitude_station AS lat FROM gold.dim_hydro_stations WHERE longitude_station IS NOT NULL"
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        return [(r["code"], r["lon"], r["lat"]) for r in conn.execute(text(sql)).mappings()]


def _meta_from_geojson(geojson: dict) -> dict:
    return {f["properties"]["sector_id"]: {
        "nom": f["properties"].get("nom"),
        "tendancy_coord": f["properties"].get("tendancy_coord"),
    } for f in geojson["features"]}


@lru_cache(maxsize=2)
def get_mapping(type_: str):
    """Cached (code->sector_id, {sector_id: meta}) for a station type.

    Prefers the precomputed station->sector map (api/data/station_sectors.json,
    written by build_secteurs_bsh_geojson) so the API never runs point-in-polygon
    at request time. Falls back to a one-off runtime computation if it's absent.
    """
    geojson = _load_geojson()
    if STATION_SECTORS.exists():
        precomputed = json.loads(STATION_SECTORS.read_text()).get(type_, {})
        return precomputed, _meta_from_geojson(geojson)
    return build_mapping(geojson, _load_stations(type_))
