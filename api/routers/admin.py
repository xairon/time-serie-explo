import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models_db import User, UserRole
from api.auth.deps import require_admin
from api.auth.passwords import hash_password
from api.auth.schemas import UserOut

router = APIRouter(prefix="/api/v1/admin/users", tags=["admin"])


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
    await db.commit()
    await db.refresh(user)
    return user


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(user_id: uuid.UUID, req: UpdateUserRequest, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if req.is_active is not None:
        user.is_active = req.is_active
        user.token_version += 1
    if req.role is not None:
        user.role = req.role
    if req.new_password is not None:
        user.password_hash = hash_password(req.new_password)
        user.token_version += 1
    await db.commit()
    await db.refresh(user)
    return user
