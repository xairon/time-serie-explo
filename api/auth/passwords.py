from pwdlib import PasswordHash

_hasher = PasswordHash.recommended()

# Precomputed hash used to run a verify even when the account does not exist, so
# login timing does not reveal whether an email is registered (user enumeration).
_DUMMY_HASH = _hasher.hash("constant-time-dummy-password")


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _hasher.verify(plain, hashed)


def dummy_verify() -> None:
    """Run a throwaway verification to equalize timing when no user/hash exists."""
    try:
        _hasher.verify("x", _DUMMY_HASH)
    except Exception:
        pass
