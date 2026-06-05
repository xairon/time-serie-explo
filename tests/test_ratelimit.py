import pytest

from api.auth import ratelimit


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def incr(self, k):
        self.store[k] = self.store.get(k, 0) + 1
        return self.store[k]

    async def expire(self, k, ttl):
        return True

    async def get(self, k):
        v = self.store.get(k)
        return None if v is None else str(v).encode()

    async def setex(self, k, ttl, v):
        self.store[k] = v
        return True

    async def delete(self, *ks):
        for k in ks:
            self.store.pop(k, None)
        return True


@pytest.mark.asyncio
async def test_lockout_after_threshold(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)
    email = "victim@example.com"
    assert await ratelimit.is_locked(email) is False
    for _ in range(5):
        await ratelimit.register_failure(email)
    assert await ratelimit.is_locked(email) is True


@pytest.mark.asyncio
async def test_success_clears_failures(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)
    email = "ok@example.com"
    await ratelimit.register_failure(email)
    await ratelimit.clear_failures(email)
    for _ in range(4):
        await ratelimit.register_failure(email)
    assert await ratelimit.is_locked(email) is False


@pytest.mark.asyncio
async def test_no_redis_is_noop(monkeypatch):
    monkeypatch.setattr(ratelimit, "get_redis", lambda: None)
    assert await ratelimit.is_locked("x@y.z") is False
    await ratelimit.register_failure("x@y.z")  # must not raise
    await ratelimit.clear_failures("x@y.z")
