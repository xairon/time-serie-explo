"""One-shot: fetch BRGM BSH parent sectors (geometry only) and bake a name per sector
from the dominant entité hydrogéologique (libelle_eh) of the piezo stations inside it.

Run (from the host, warehouse published on localhost:49502):
    DEBUG=true BRGM_DB_HOST=localhost BRGM_DB_PORT=49502 \
        [SECTORS_WFS_FILE=/path/to/wfs_parents.json] \
        .venv/bin/python -m scripts.build_secteurs_bsh_geojson

The BRGM WFS is slow/flaky; if it times out, download a GetFeature GeoJSON response
once (filtered to is_parent=true for one period) and pass it via SECTORS_WFS_FILE.
Re-run only if BRGM changes the sectorization.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from sqlalchemy import text

from api.database import get_brgm_sync_engine
from dashboard.utils.geo_sectors import point_in_geometry, dominant_label

# Two copies: the frontend serves the public/ one; the backend (sector_mapping.py)
# reads the api/data/ one, because the backend image ships api/ but not frontend/.
OUT = Path("frontend/public/geo/secteurs-bsh.geojson")
OUT_BACKEND = Path("api/data/secteurs-bsh.geojson")
# Precomputed station->sector map so the API never does point-in-polygon at runtime.
OUT_MAP = Path("api/data/station_sectors.json")
WFS = "https://app.meteeaunappes.brgm.fr/wfs/indicateur_bsn/ows"
# Restrict to a SINGLE published snapshot (one 15-day window) so the WFS returns one
# polygon per sector, not the same sector repeated for every period since 2025.
# Any date covered by a communicated window works; geometry is identical across periods.
SNAPSHOT_DATE = "2026-05-09"
FILTER = (
    '<Filter xmlns:gml="http://www.opengis.net/gml"><And>'
    f"<PropertyIsLessThanOrEqualTo><PropertyName>start_period</PropertyName><Literal>{SNAPSHOT_DATE}</Literal></PropertyIsLessThanOrEqualTo>"
    f"<PropertyIsGreaterThanOrEqualTo><PropertyName>end_period</PropertyName><Literal>{SNAPSHOT_DATE}</Literal></PropertyIsGreaterThanOrEqualTo>"
    "<PropertyIsEqualTo><PropertyName>communicate</PropertyName><Literal>true</Literal></PropertyIsEqualTo>"
    "<PropertyIsEqualTo><PropertyName>is_parent</PropertyName><Literal>true</Literal></PropertyIsEqualTo>"
    "<PropertyIsEqualTo><PropertyName>visualizer</PropertyName><Literal>true</Literal></PropertyIsEqualTo>"
    "</And></Filter>"
)


def fetch_sectors() -> list[dict]:
    """Return parent-sector features. The BRGM WFS is slow/flaky, so a pre-downloaded
    snapshot may be supplied via the SECTORS_WFS_FILE env var (a GetFeature GeoJSON
    response filtered to is_parent=true for one period); otherwise fetch live.
    """
    local = os.environ.get("SECTORS_WFS_FILE")
    if local:
        feats = json.loads(Path(local).read_text())["features"]
        return [f for f in feats if f["properties"].get("is_parent") in (True, "true")]
    url = (
        f"{WFS}?service=WFS&version=1.0.0&request=GetFeature"
        f"&outputFormat=application%2Fjson"
        f"&typeName=indicateur_bsn:view_global_indicator_details&filter={quote(FILTER)}"
    )
    with urlopen(url, timeout=300) as resp:
        return json.load(resp)["features"]


def piezo_stations() -> list[tuple]:
    """(code, lon, lat, libelle_eh) for piezo stations (EH label used for naming)."""
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.code_bss AS code, s.longitude AS lon, s.latitude AS lat, m.libelle_eh
            FROM gold.dim_piezo_stations s
            LEFT JOIN gold.int_station_era5_mapping m ON m.code_bss = s.code_bss
            WHERE s.longitude IS NOT NULL AND s.latitude IS NOT NULL
        """)).mappings().all()
    return [(r["code"], float(r["lon"]), float(r["lat"]), r["libelle_eh"]) for r in rows]


def hydro_stations() -> list[tuple]:
    """(code, lon, lat) for hydro stations."""
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.code_station AS code, s.longitude_station AS lon, s.latitude_station AS lat
            FROM gold.dim_hydro_stations s
            WHERE s.longitude_station IS NOT NULL AND s.latitude_station IS NOT NULL
        """)).mappings().all()
    return [(r["code"], float(r["lon"]), float(r["lat"]) ) for r in rows]


def main() -> int:
    feats = fetch_sectors()
    # Dedup to one polygon per sector.
    sectors: list[tuple] = []  # (sid, tendancy_coord, geom)
    seen: set = set()
    for f in feats:
        sid = f["properties"]["sector_id"]
        if sid in seen:
            continue
        seen.add(sid)
        sectors.append((sid, f["properties"].get("tendancy_coord"), f["geometry"]))

    # Piezo: map each station to its sector + collect EH labels per sector for naming.
    piezo_map: dict[str, int] = {}
    labels_by_sid: dict[int, list] = {}
    for code, lon, lat, eh in piezo_stations():
        for sid, _coord, geom in sectors:
            if point_in_geometry(lon, lat, geom):
                piezo_map[code] = sid
                if eh:
                    labels_by_sid.setdefault(sid, []).append(eh)
                break

    # Hydro: map each station to its sector.
    hydro_map: dict[str, int] = {}
    for code, lon, lat in hydro_stations():
        for sid, _coord, geom in sectors:
            if point_in_geometry(lon, lat, geom):
                hydro_map[code] = sid
                break

    out_features = [{
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "sector_id": sid,
            "tendancy_coord": coord,
            "nom": dominant_label(labels_by_sid.get(sid, [])) or f"Secteur {sid}",
        },
    } for sid, coord, geom in sectors]

    payload = json.dumps({"type": "FeatureCollection", "features": out_features})
    for dest in (OUT, OUT_BACKEND):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload)
    OUT_MAP.write_text(json.dumps({"piezo": piezo_map, "hydro": hydro_map}))

    named = sum(1 for f in out_features if not f["properties"]["nom"].startswith("Secteur "))
    print(f"wrote {len(out_features)} sectors to {OUT} ({named} with an EH name)")
    print(f"mapped piezo: {len(piezo_map)}, hydro: {len(hydro_map)} stations -> {OUT_MAP}")
    assert len(out_features) >= 50, "expected ~66 parent sectors"
    return 0


if __name__ == "__main__":
    sys.exit(main())
