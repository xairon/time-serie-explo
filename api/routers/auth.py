from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from api.config import settings
from api.database import get_db
from api.models_db import User
from api.auth.deps import get_current_user
from api.auth.passwords import hash_password, verify_password
from api.auth.schemas import ChangePasswordRequest, LoginRequest, UserOut
from api.auth.tokens import create_session_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _set_session_cookie(response: Response, user: User) -> None:
    token = create_session_token(user.id, user.token_version)
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.session_ttl_hours * 3600,
        path="/",
    )


@router.post("/login", response_model=UserOut)
async def login(req: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    user.last_login_at = func.now()
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)
    return user


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(settings.cookie_name, path="/")


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user


@router.post("/change-password", status_code=204)
async def change_password(
    req: ChangePasswordRequest,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Ancien mot de passe incorrect")
    user.password_hash = hash_password(req.new_password)
    user.token_version += 1
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)
