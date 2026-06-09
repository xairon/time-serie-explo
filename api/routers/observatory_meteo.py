"""Observatory BRGM MétéEau-des-nappes proxy — fetches per-sector class/color/trend."""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import httpx
from fastapi import APIRouter

from dashboard.utils.cache import get_cached

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/observatory/meteo", tags=["observatory-meteo"])

_BRGM_WFS_URL = "https://app.meteeaunappes.brgm.fr/wfs/indicateur_bsn/ows"
_TTL = 86400  # 24 h

_TENDENCY_MAP = {-1: "baisse", 0: "stable", 1: "hausse"}


def parse_brgm_sectors(features: list[dict]) -> list[dict]:
    """Map BRGM WFS features to a deduplicated list of per-sector attribute dicts.

    Keeps only features where ``is_parent`` is truthy; deduplicates by
    ``sector_id`` (first occurrence wins).

    Args:
        features: Raw GeoJSON feature dicts from the BRGM WFS response.

    Returns:
        List of dicts with keys: sector_id, color, brgm_class, trend, ips,
        status, tendancy_coord.
    """
    seen: set[int] = set()
    result: list[dict] = []

    for feat in features:
        props = feat.get("properties") or {}
        if not props.get("is_parent"):
            continue

        sector_id = props.get("sector_id")
        if sector_id in seen:
            continue
        seen.add(sector_id)

        tendency_raw = props.get("tendency")
        trend: Optional[str] = _TENDENCY_MAP.get(tendency_raw)  # None if not in map

        result.append(
            {
                "sector_id": sector_id,
                "color": props.get("color"),
                "brgm_class": props.get("class"),
                "trend": trend,
                "ips": props.get("ips"),
                "status": props.get("status"),
                "tendancy_coord": props.get("tendancy_coord"),
            }
        )

    return result


def _fetch_brgm_sectors() -> list[dict]:
    """Fetch BRGM WFS and parse into sector attribute dicts (sync, ~60-120s)."""
    today = date.today().isoformat()
    xml_filter = (
        '<Filter xmlns:gml="http://www.opengis.net/gml"><And>'
        "<PropertyIsLessThanOrEqualTo><PropertyName>start_period</PropertyName>"
        f"<Literal>{today}</Literal></PropertyIsLessThanOrEqualTo>"
        "<PropertyIsGreaterThanOrEqualTo><PropertyName>end_period</PropertyName>"
        f"<Literal>{today}</Literal></PropertyIsGreaterThanOrEqualTo>"
        "<PropertyIsEqualTo><PropertyName>communicate</PropertyName>"
        "<Literal>true</Literal></PropertyIsEqualTo>"
        "<PropertyIsEqualTo><PropertyName>is_parent</PropertyName>"
        "<Literal>true</Literal></PropertyIsEqualTo>"
        "<PropertyIsEqualTo><PropertyName>visualizer</PropertyName>"
        "<Literal>true</Literal></PropertyIsEqualTo>"
        "</And></Filter>"
    )
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "outputFormat": "application/json",
        "typeName": "indicateur_bsn:view_global_indicator_details",
        "filter": xml_filter,
    }
    resp = httpx.get(_BRGM_WFS_URL, params=params, timeout=120)
    resp.raise_for_status()
    features = resp.json()["features"]
    return parse_brgm_sectors(features)


@router.get("/brgm-sectors")
def get_brgm_sectors() -> list:
    """Return BRGM per-sector class/color/trend for today, cached 24 h.

    On any network or parse error, returns an empty list so the frontend
    can fall back to our own IPS data without showing a hard error.
    """
    today = date.today().isoformat()

    def fetch():
        return _fetch_brgm_sectors()

    try:
        return get_cached("meteo_brgm_sectors", {"date": today}, _TTL, fetch)
    except Exception as exc:
        logger.warning("BRGM WFS fetch/parse failed, returning []: %s", exc)
        return []
