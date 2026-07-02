import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models_db import User, UserRole
from api.auth.audit import record_event
from api.auth.deps import require_admin
from api.auth.erasure import erase_user_artifacts
from api.auth.passwords import hash_password
from api.auth.schemas import UserOut

router = APIRouter(prefix="/api/v1/admin/users", tags=["admin"])


async def _other_active_admins(db: AsyncSession, exclude_user_id: uuid.UUID) -> int:
    """Count active admins other than the given user (the "at least one admin" invariant)."""
    return (
        await db.execute(
            select(func.count())
            .select_from(User)
            .where(
                User.role == UserRole.admin,
                User.is_active.is_(True),
                User.id != exclude_user_id,
            )
        )
    ).scalar_one()


class CreateUserRequest(BaseModel):
    email: EmailStr
    display_name: str
    role: UserRole = UserRole.user
    initial_password: str = Field(min_length=8)


class UpdateUserRequest(BaseModel):
    is_active: bool | None = None
    role: UserRole | None = None
    new_password: str | None = Field(default=None, min_length=8)


@router.get("", response_model=list[UserOut])
async def list_users(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    return list((await db.execute(select(User).order_by(User.created_at))).scalars())


@router.post("", response_model=UserOut, status_code=201)
async def create_user(req: CreateUserRequest, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    exists = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    user = User(
        email=req.email, display_name=req.display_name,
        password_hash=hash_password(req.initial_password), role=req.role, is_active=True,
    )
    db.add(user)
    await db.flush()
    await record_event(db, event_type="admin_user_created", email=user.email, user_id=user.id)
    await db.commit()
    await db.refresh(user)
    return user


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(user_id: uuid.UUID, req: UpdateUserRequest, current_admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    # Guard the "at least one active admin" invariant (delete_user enforces it too):
    # demoting or deactivating the last active admin would lock everyone out permanently.
    demotes_admin = user.role == UserRole.admin and req.role is not None and req.role != UserRole.admin
    deactivates_admin = user.role == UserRole.admin and req.is_active is False
    if demotes_admin or deactivates_admin:
        if user.id == current_admin.id:
            raise HTTPException(
                status_code=409,
                detail="Un administrateur ne peut pas se rétrograder ni se désactiver lui-même",
            )
        if await _other_active_admins(db, exclude_user_id=user.id) == 0:
            raise HTTPException(
                status_code=409, detail="Impossible de rétrograder/désactiver le dernier administrateur"
            )

    if req.is_active is not None:
        user.is_active = req.is_active
        user.token_version += 1
    if req.role is not None:
        user.role = req.role
        user.token_version += 1
    if req.new_password is not None:
        user.password_hash = hash_password(req.new_password)
        user.token_version += 1
    await record_event(db, event_type="admin_user_updated", email=user.email, user_id=user.id)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/{user_id}/reset-password")
async def reset_password(
    user_id: uuid.UUID,
    request: Request,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    temp = secrets.token_urlsafe(12)
    user.password_hash = hash_password(temp)
    user.token_version += 1
    user.must_change_password = True
    await record_event(
        db, event_type="password_reset", email=user.email, user_id=user.id,
        ip=(request.client.host if request.client else None),
        user_agent=request.headers.get("user-agent"),
    )
    await db.commit()
    return {"temporary_password": temp}


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: uuid.UUID,
    request: Request,
    response: Response,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if user.role == UserRole.admin and await _other_active_admins(db, exclude_user_id=user.id) == 0:
        raise HTTPException(
            status_code=409, detail="Impossible de supprimer le dernier administrateur"
        )
    uid = user.id
    await db.delete(user)
    await record_event(
        db, event_type="admin_user_deleted", email=None, user_id=uid,
        ip=(request.client.host if request.client else None),
        user_agent=request.headers.get("user-agent"),
    )
    await db.commit()
    # Irreversible artifact erasure happens only AFTER the DB deletion is committed,
    # so a commit failure can never destroy data belonging to a still-existing user.
    erase_user_artifacts(uid)
