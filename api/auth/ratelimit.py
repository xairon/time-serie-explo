import logging

from fastapi import HTTPException, Request

from api.cache import get_redis
from api.config import settings

logger = logging.getLogger(__name__)


def _norm(email: str) -> str:
    return (email or "").strip().lower()


def _fail_key(email: str) -> str:
    return f"junon:loginfail:{_norm(email)}"


def _lock_key(email: str) -> str:
    return f"junon:loginlock:{_norm(email)}"


def _rate_key(ip: str) -> str:
    return f"junon:loginrate:{ip}"


async def is_locked(email: str) -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        return (await r.get(_lock_key(email))) is not None
    except Exception as e:
        logger.debug("lockout check failed: %s", e)
        return False


async def register_failure(email: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        n = await r.incr(_fail_key(email))
        if n == 1:
            await r.expire(_fail_key(email), settings.login_fail_window_minutes * 60)
        if n >= settings.login_max_failures:
            await r.setex(_lock_key(email), settings.login_lockout_minutes * 60, b"1")
    except Exception as e:
        logger.debug("register_failure failed: %s", e)


async def clear_failures(email: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        await r.delete(_fail_key(email), _lock_key(email))
    except Exception as e:
        logger.debug("clear_failures failed: %s", e)


async def enforce_ip_rate_limit(request: Request) -> None:
    """Sliding 60s counter per client IP. Raises 429 over the limit.

    Fails CLOSED when Redis is configured but unreachable: an attacker must not be
    able to disable login throttling by knocking Redis over. A 503 is retryable and
    does not permanently lock anyone out. When Redis is intentionally not configured
    (client is None, e.g. dev/test), this is a no-op by design.
    """
    r = get_redis()
    if r is None:
        return
    ip = request.client.host if request.client else "unknown"
    key = _rate_key(ip)
    try:
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, 60)
        if n > settings.login_rate_limit_per_minute:
            raise HTTPException(
                status_code=429, detail="Trop de tentatives, réessayez plus tard"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("rate limit backend unavailable, failing closed: %s", e)
        raise HTTPException(
            status_code=503, detail="Service d'authentification temporairement indisponible"
        )
