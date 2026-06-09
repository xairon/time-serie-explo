"""Pure geometry helpers for sector mapping. No DB, no Streamlit, no shapely."""
from __future__ import annotations

from collections import Counter


def point_in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _in_polygon(x: float, y: float, polygon: list) -> bool:
    outer, *holes = polygon
    if not point_in_ring(x, y, outer):
        return False
    return not any(point_in_ring(x, y, h) for h in holes)


def point_in_geometry(lon: float, lat: float, geometry: dict) -> bool:
    coords = geometry.get("coordinates")
    gtype = geometry.get("type")
    if gtype == "Polygon":
        return _in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        return any(_in_polygon(lon, lat, poly) for poly in coords)
    return False


def dominant_label(labels) -> str | None:
    clean = [str(s).strip() for s in labels if s is not None and str(s).strip()]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]
