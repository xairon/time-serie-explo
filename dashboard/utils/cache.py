"""Simple Redis cache wrapper for sync code paths.

Uses a connection pool to the Redis service defined in docker-compose.
"""
from __future__ import annotations

import hashlib
import json
import logging

import redis

logger = logging.getLogger(__name__)

_pool = redis.ConnectionPool(host="redis", port=6379, db=0, decode_responses=True)


def _normalize_value(v):
    if isinstance(v, list):
        return sorted(str(x) for x in v)
    return v


def get_cached(prefix: str, params: dict, ttl: int, fetch_fn):
    """Try Redis cache, on miss call fetch_fn() and store result.

    Args:
        prefix: Cache key namespace (e.g. "piezo_list").
        params: Dict of query params used as cache key discriminator.
        ttl: Time-to-live in seconds.
        fetch_fn: Callable that returns JSON-serializable data on cache miss.

    Returns:
        The cached or freshly-fetched result (Python object, not bytes).
    """
    normalized = {k: _normalize_value(v) for k, v in params.items()}
    raw = json.dumps(normalized, sort_keys=True, default=str)
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    key = f"junon:{prefix}:{h}"

    r = redis.Redis(connection_pool=_pool)
    try:
        cached = r.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        logger.debug("Redis GET miss/error for %s", key)

    result = fetch_fn()

    try:
        r.setex(key, ttl, json.dumps(result, default=str))
    except Exception:
        logger.debug("Redis SETEX error for %s", key)

    return result
