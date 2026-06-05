import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from api.models_db.auth_event import AuthEvent


async def record_event(
    db: AsyncSession,
    *,
    event_type: str,
    email: str | None = None,
    user_id: uuid.UUID | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Append an auth audit event. Caller is responsible for the commit."""
    db.add(
        AuthEvent(
            user_id=user_id,
            email=email,
            event_type=event_type,
            ip=ip,
            user_agent=(user_agent or "")[:400] or None,
        )
    )
