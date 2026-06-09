"""Observatory BRGM MétéEau-des-nappes proxy — fetches per-sector class/color/trend."""
from __future__ import annotations

import logging
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
    # The WFS returns every published 15-day window; keep only the most recent one
    # (max end_period). Today's literal date is often past the last published window,
    # so we never filter by "today" — we always take the latest available situation.
    parents = [f for f in features if (f.get("properties") or {}).get("is_parent")]
    end_periods = [(f["properties"].get("end_period")) for f in parents if f["properties"].get("end_period")]
    if end_periods:
        latest = max(end_periods)
        parents = [f for f in parents if f["properties"].get("end_period") == latest]

    seen: set[int] = set()
    result: list[dict] = []

    for feat in parents:
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


def _fetch_brgm_features() -> list[dict]:
    """Fetch BRGM WFS (all communicated parent windows, attributes only) and return
    raw GeoJSON feature dicts. No date filter (today is often past the last window).
    `propertyName` drops the heavy geometry (~788 KB vs ~100 MB)."""
    xml_filter = (
        '<Filter xmlns:gml="http://www.opengis.net/gml"><And>'
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
        "propertyName": "sector_id,class,color,tendency,ips,status,tendancy_coord,end_period,is_parent",
    }
    resp = httpx.get(_BRGM_WFS_URL, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()["features"]


def _fetch_brgm_sectors() -> list[dict]:
    """Fetch BRGM WFS features and parse the latest published window."""
    return parse_brgm_sectors(_fetch_brgm_features())


def parse_brgm_timeline(features: list[dict]) -> dict:
    """Group BRGM parent-sector features into MONTHLY periods (one window per month:
    the latest end_period within that calendar month).

    Returns {"periods": ["YYYY-MM", ...] sorted asc,
             "windows": { "YYYY-MM": { "<sector_id>": {"color","brgm_class","trend","tendancy_coord"} } }}
    """
    # best[month][sector_id] = (end_period_str, props)
    best: dict[str, dict[str, tuple[str, dict]]] = {}

    for feat in features:
        props = (feat.get("properties") or {})
        if not props.get("is_parent"):
            continue
        end_period = props.get("end_period")
        if not end_period:
            continue
        sector_id = str(props.get("sector_id"))
        month = end_period[:7]  # "YYYY-MM"

        month_map = best.setdefault(month, {})
        existing = month_map.get(sector_id)
        if existing is None or end_period > existing[0]:
            month_map[sector_id] = (end_period, props)

    windows: dict[str, dict[str, dict]] = {}
    for month, sector_map in best.items():
        windows[month] = {}
        for sector_id, (_ep, props) in sector_map.items():
            tendency_raw = props.get("tendency")
            trend: Optional[str] = _TENDENCY_MAP.get(tendency_raw)
            windows[month][sector_id] = {
                "color": props.get("color"),
                "brgm_class": props.get("class"),
                "trend": trend,
                "tendancy_coord": props.get("tendancy_coord"),
            }

    periods = sorted(windows.keys())
    return {"periods": periods, "windows": windows}


@router.get("/brgm-sectors")
def get_brgm_sectors() -> list:
    """Return BRGM per-sector class/color/trend for today, cached 24 h.

    On any network or parse error, returns an empty list so the frontend
    can fall back to our own IPS data without showing a hard error.
    """
    def fetch():
        return _fetch_brgm_sectors()

    try:
        # Cache key v2 (latest-window strategy); busts the stale empty v1 cache.
        return get_cached("meteo_brgm_sectors", {"v": 2}, _TTL, fetch)
    except Exception as exc:
        logger.warning("BRGM WFS fetch/parse failed, returning []: %s", exc)
        return []


@router.get("/brgm-timeline")
def get_brgm_timeline() -> dict:
    """Return all BRGM windows collapsed to monthly periods, cached 24 h.

    Returns {"periods": [...], "windows": {...}} suitable for a time slider.
    On any network or parse error, returns empty periods/windows.
    """
    def fetch():
        return parse_brgm_timeline(_fetch_brgm_features())

    try:
        return get_cached("meteo_brgm_timeline", {"v": 1}, _TTL, fetch)
    except Exception as exc:
        logger.warning("BRGM timeline fetch/parse failed, returning empty: %s", exc)
        return {"periods": [], "windows": {}}
