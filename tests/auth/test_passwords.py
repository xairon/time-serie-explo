from api.auth.passwords import hash_password, verify_password


def test_hash_and_verify_roundtrip():
    h = hash_password("s3cret-pass")
    assert h != "s3cret-pass"
    assert verify_password("s3cret-pass", h) is True


def test_verify_rejects_wrong_password():
    h = hash_password("s3cret-pass")
    assert verify_password("wrong", h) is False
