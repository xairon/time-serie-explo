import uuid

import pytest

from api.auth.tokens import create_session_token, decode_session_token


def test_roundtrip_encodes_subject_and_token_version():
    uid = uuid.uuid4()
    token = create_session_token(uid, token_version=3)
    claims = decode_session_token(token)
    assert claims["sub"] == str(uid)
    assert claims["tv"] == 3


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        decode_session_token("not-a-jwt")
