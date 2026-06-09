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

OUT = Path("frontend/public/geo/secteurs-bsh.geojson")
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


def station_points() -> list[tuple]:
    """(lon, lat, libelle_eh) for piezo stations that have an EH label."""
    engine = get_brgm_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.longitude AS lon, s.latitude AS lat, m.libelle_eh
            FROM gold.dim_piezo_stations s
            JOIN gold.int_station_era5_mapping m ON m.code_bss = s.code_bss
            WHERE s.longitude IS NOT NULL AND s.latitude IS NOT NULL
        """)).mappings().all()
    return [(float(r["lon"]), float(r["lat"]), r["libelle_eh"]) for r in rows]


def main() -> int:
    feats = fetch_sectors()
    pts = station_points()
    out_features = []
    seen: set = set()
    for f in feats:
        sid = f["properties"]["sector_id"]
        if sid in seen:  # one polygon per sector (guard against multi-period leakage)
            continue
        seen.add(sid)
        coord = f["properties"].get("tendancy_coord")  # "lat lon"
        geom = f["geometry"]
        labels = [eh for (lon, lat, eh) in pts if point_in_geometry(lon, lat, geom)]
        nom = dominant_label(labels) or f"Secteur {sid}"
        out_features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"sector_id": sid, "tendancy_coord": coord, "nom": nom},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": out_features}))
    named = sum(1 for f in out_features if not f["properties"]["nom"].startswith("Secteur "))
    print(f"wrote {len(out_features)} sectors to {OUT} ({named} with an EH name)")
    assert len(out_features) >= 50, "expected ~66 parent sectors"
    return 0


if __name__ == "__main__":
    sys.exit(main())
