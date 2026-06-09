"""Load secteurs-bsh.geojson and map station coords -> sector_id (cached process-wide)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from api.database import get_brgm_sync_engine
from dashboard.utils.geo_sectors import point_in_geometry

GEOJSON = Path(__file__).resolve().parents[2] / "frontend" / "public" / "geo" / "secteurs-bsh.geojson"


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


@lru_cache(maxsize=2)
def get_mapping(type_: str):
    """Cached (code->sector_id, {sector_id: meta}) for a station type."""
    return build_mapping(_load_geojson(), _load_stations(type_))
