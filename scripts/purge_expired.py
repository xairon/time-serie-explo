"""RGPD retention purge.

Deletes auth audit events older than the configured retention window.
Safe to run repeatedly. Intended to run from cron, e.g. daily:

    python -m scripts.purge_expired
"""
import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import async_session
from api.models_db.auth_event import AuthEvent


async def purge_auth_events(db: AsyncSession, retention_days: int) -> int:
    """Delete auth_events older than retention_days. Returns rows deleted."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    result = await db.execute(delete(AuthEvent).where(AuthEvent.created_at < cutoff))
    return result.rowcount or 0


async def main() -> None:
    async with async_session() as db:
        n = await purge_auth_events(db, settings.auth_event_retention_days)
        await db.commit()
        print(
            f"Purged {n} auth_events older than "
            f"{settings.auth_event_retention_days} days"
        )


if __name__ == "__main__":
    asyncio.run(main())
