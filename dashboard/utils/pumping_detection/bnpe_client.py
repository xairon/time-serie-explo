"""Hub'Eau Prélèvements API client with caching."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

HUBEAU_PRELEVEMENTS_URL = "https://hubeau.eaufrance.fr/api/v1/prelevements"

_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 24 * 3600


class BNPEClient:
    """Fetch nearby BNPE declared pumping facilities."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch_nearby(self, lat: float, lon: float, radius_km: float = 5) -> dict[str, Any]:
        cache_key = f"{round(lat, 2)}_{round(lon, 2)}_{radius_km}"

        if cache_key in _cache:
            ts, data = _cache[cache_key]
            if time.time() - ts < _CACHE_TTL:
                return data

        try:
            delta = radius_km / 111.0
            params = {
                "latitude": f"[{lat - delta},{lat + delta}]",
                "longitude": f"[{lon - delta},{lon + delta}]",
                "fields": "code_ouvrage,nom_ouvrage,latitude,longitude,code_commune_insee,nom_commune",
                "format": "json",
                "size": 100,
            }
            resp = requests.get(
                f"{HUBEAU_PRELEVEMENTS_URL}/ouvrages",
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            result = {
                "bnpe_available": True,
                "ouvrages": data.get("data", []),
                "count": data.get("count", 0),
            }
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"BNPE API unavailable: {e}")
            result = {"bnpe_available": False, "ouvrages": [], "count": 0}
        except Exception as e:
            logger.error(f"BNPE API error: {e}")
            result = {"bnpe_available": False, "ouvrages": [], "count": 0}

        _cache[cache_key] = (time.time(), result)
        return result
