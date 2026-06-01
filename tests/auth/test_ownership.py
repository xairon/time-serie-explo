import uuid

import pytest

from api.auth.ownership import is_owner_or_admin
from api.models_db import User, UserRole


def _user(role=UserRole.user):
    u = User(id=uuid.uuid4(), email="x@y.fr", display_name="x", password_hash="h", role=role, is_active=True, token_version=0)
    return u


def test_owner_matches():
    u = _user()
    assert is_owner_or_admin(u, str(u.id)) is True


def test_other_owner_denied():
    u = _user()
    assert is_owner_or_admin(u, str(uuid.uuid4())) is False


def test_admin_always_allowed():
    u = _user(role=UserRole.admin)
    assert is_owner_or_admin(u, str(uuid.uuid4())) is True


def test_missing_owner_denied_for_user_allowed_for_admin():
    assert is_owner_or_admin(_user(), None) is False
    assert is_owner_or_admin(_user(UserRole.admin), None) is True
