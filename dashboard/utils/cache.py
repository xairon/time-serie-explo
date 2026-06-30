"""Simple Redis cache wrapper for sync code paths.

Uses a connection pool to the Redis service defined in docker-compose.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import redis

logger = logging.getLogger(__name__)

# Honor REDIS_URL so the host is explicit. Default to the unambiguous container
# name junon-redis: the backend is attached to both the compose network and the
# BRGM network, and the bare service alias "redis" collides with the BRGM stack's
# own redis (it resolved there, serving stale cache). See compose REDIS_URL.
_REDIS_URL = os.environ.get("REDIS_URL", "redis://junon-redis:6379/0")
_pool = redis.ConnectionPool.from_url(_REDIS_URL, decode_responses=True)


def _normalize_value(v):
    if isinstance(v, list):
        return sorted(str(x) for x in v)
    return v


def _make_key(prefix: str, params: dict) -> str:
    """Build the canonical Redis key for a given prefix + params dict."""
    normalized = {k: _normalize_value(v) for k, v in params.items()}
    raw = json.dumps(normalized, sort_keys=True, default=str)
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"junon:{prefix}:{h}"


def read_cached(prefix: str, params: dict):
    """Return the deserialized cached value if present, else None.

    Never triggers a fetch. Swallows Redis errors (returns None on any failure).
    Key is built identically to get_cached via _make_key.
    """
    key = _make_key(prefix, params)
    r = redis.Redis(connection_pool=_pool)
    try:
        cached = r.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        logger.debug("Redis GET miss/error for %s", key)
    return None


def delete_cached(prefix: str, params: dict) -> None:
    """Delete the cache entry for the given prefix + params. Swallows errors."""
    key = _make_key(prefix, params)
    r = redis.Redis(connection_pool=_pool)
    try:
        r.delete(key)
    except Exception:
        logger.debug("Redis DELETE error for %s", key)


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
    key = _make_key(prefix, params)

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
