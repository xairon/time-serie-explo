"""Create (or promote) an admin user. Usage:
    python scripts/create_admin.py --email a@b.fr --name "Nicolas" [--password ...]
Password resolution order: --password, then $ADMIN_PASSWORD (for automation),
then a secure interactive prompt (no echo). Avoid --password on shared hosts:
it leaks via `ps` and shell history.
"""
import argparse
import asyncio
import getpass
import os

from sqlalchemy import select

from api.database import async_session
from api.models_db import User, UserRole
from api.auth.passwords import hash_password


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--password")
    args = p.parse_args()
    password = args.password or os.environ.get("ADMIN_PASSWORD") or getpass.getpass("Password: ")

    async with async_session() as db:
        existing = (await db.execute(select(User).where(User.email == args.email))).scalar_one_or_none()
        if existing:
            existing.role = UserRole.admin
            existing.password_hash = hash_password(password)
            existing.is_active = True
            print(f"Promoted existing user {args.email} to admin.")
        else:
            db.add(User(
                email=args.email, display_name=args.name,
                password_hash=hash_password(password), role=UserRole.admin, is_active=True,
            ))
            print(f"Created admin {args.email}.")
        await db.commit()


if __name__ == "__main__":
    asyncio.run(main())
