from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from api.config import settings
from api.database import get_db
from api.models_db import User
from api.auth.audit import record_event
from api.auth.deps import get_current_user
from api.auth.passwords import hash_password, verify_password
from api.auth.ratelimit import (
    clear_failures,
    enforce_ip_rate_limit,
    is_locked,
    register_failure,
)
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
async def login(
    req: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    await enforce_ip_rate_limit(request)
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    if await is_locked(req.email):
        raise HTTPException(
            status_code=429, detail="Compte temporairement verrouillé, réessayez plus tard"
        )
    user = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(req.password, user.password_hash):
        await register_failure(req.email)
        await record_event(
            db, event_type="login_failure", email=req.email,
            user_id=(user.id if user else None), ip=ip, user_agent=ua,
        )
        await db.commit()
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    await clear_failures(req.email)
    user.last_login_at = func.now()
    await record_event(
        db, event_type="login_success", email=user.email, user_id=user.id, ip=ip, user_agent=ua
    )
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
    request: Request,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Ancien mot de passe incorrect")
    user.password_hash = hash_password(req.new_password)
    user.token_version += 1
    user.must_change_password = False
    await record_event(
        db, event_type="password_change", email=user.email, user_id=user.id,
        ip=(request.client.host if request.client else None),
        user_agent=request.headers.get("user-agent"),
    )
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)


@router.delete("/me", status_code=204)
async def delete_me(
    request: Request,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from api.auth.erasure import erase_user_artifacts

    uid = user.id
    erase_user_artifacts(uid)
    await db.delete(user)
    await record_event(
        db, event_type="account_deleted", email=None, user_id=uid,
        ip=(request.client.host if request.client else None),
        user_agent=request.headers.get("user-agent"),
    )
    await db.commit()
    response.delete_cookie(settings.cookie_name, path="/")
