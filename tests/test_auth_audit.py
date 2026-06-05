import pytest
from sqlalchemy import select

from api.auth.audit import record_event
from api.models_db.auth_event import AuthEvent


@pytest.mark.asyncio
async def test_record_event_persists(db_session):
    await record_event(
        db_session, event_type="login_success", email="a@b.c",
        user_id=None, ip="1.2.3.4", user_agent="pytest",
    )
    await db_session.commit()
    rows = (await db_session.execute(select(AuthEvent))).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_type == "login_success"
    assert rows[0].ip == "1.2.3.4"
    assert rows[0].email == "a@b.c"


@pytest.mark.asyncio
async def test_record_event_truncates_user_agent(db_session):
    await record_event(db_session, event_type="login_success", user_agent="x" * 1000)
    await db_session.commit()
    row = (await db_session.execute(select(AuthEvent))).scalars().one()
    assert len(row.user_agent) == 400
