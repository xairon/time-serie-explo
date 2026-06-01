from fastapi import HTTPException

from api.models_db import User, UserRole


def is_owner_or_admin(user: User, owner_id: str | None) -> bool:
    if user.role == UserRole.admin:
        return True
    if owner_id is None:
        return False
    return str(user.id) == str(owner_id)


def assert_owner_or_admin(user: User, owner_id: str | None) -> None:
    """Raise 404 (not 403) so we do not disclose existence of others' resources."""
    if not is_owner_or_admin(user, owner_id):
        raise HTTPException(status_code=404, detail="Introuvable")


def owner_filter_clause(user: User) -> str | None:
    """MLflow search filter for the current user, or None for admin (no filter)."""
    if user.role == UserRole.admin:
        return None
    return f"tags.owner_id = '{user.id}'"
