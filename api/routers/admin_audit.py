from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models_db import User
from api.models_db.auth_event import AuthEvent
from api.auth.deps import require_admin

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/auth-events")
async def list_auth_events(
    limit: int = 100,
    offset: int = 0,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Paginated auth audit events, newest first."""
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    rows = (
        await db.execute(
            select(AuthEvent)
            .order_by(AuthEvent.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()
    return [
        {
            "id": str(e.id),
            "user_id": str(e.user_id) if e.user_id else None,
            "email": e.email,
            "event_type": e.event_type,
            "ip": e.ip,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in rows
    ]
