from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from api.models_db.auth_event import AuthEvent
from scripts.purge_expired import purge_auth_events


@pytest.mark.asyncio
async def test_purge_removes_old_events(db_session):
    old = AuthEvent(event_type="login_success", email="a@b.c")
    db_session.add(old)
    await db_session.flush()
    old.created_at = datetime.now(timezone.utc) - timedelta(days=400)
    await db_session.commit()

    deleted = await purge_auth_events(db_session, retention_days=365)
    await db_session.commit()
    assert deleted == 1
    assert (await db_session.execute(select(AuthEvent))).scalars().all() == []


@pytest.mark.asyncio
async def test_purge_keeps_recent_events(db_session):
    recent = AuthEvent(event_type="login_success", email="b@b.c")
    db_session.add(recent)
    await db_session.commit()

    deleted = await purge_auth_events(db_session, retention_days=365)
    await db_session.commit()
    assert deleted == 0
    assert len((await db_session.execute(select(AuthEvent))).scalars().all()) == 1
