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


def _model_registry():
    from pathlib import Path
    from api.config import settings
    from dashboard.utils.model_registry import ModelRegistry
    return ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))


def _dataset_registry():
    from pathlib import Path
    from api.config import settings
    from dashboard.utils.dataset_registry import DatasetRegistry
    return DatasetRegistry(Path(settings.data_dir) / "prepared")


def assert_owns_model(user, model_id: str) -> None:
    assert_owner_or_admin(user, _model_registry().get_model_owner(model_id))


def assert_owns_dataset(user, dataset_id: str) -> None:
    assert_owner_or_admin(user, _dataset_registry().get_owner(dataset_id))
