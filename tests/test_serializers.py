import numpy as np
from api.serializers import clean_nans, serialize_tensor


def test_clean_nans_basic():
    d = {"a": 1.0, "b": float("nan"), "c": float("inf")}
    result = clean_nans(d)
    assert result["a"] == 1.0
    assert result["b"] is None
    assert result["c"] is None


def test_clean_nans_nested():
    d = {"outer": {"inner": float("nan")}}
    result = clean_nans(d)
    assert result["outer"]["inner"] is None


def test_clean_nans_list():
    d = {"values": [1.0, float("nan"), 3.0]}
    result = clean_nans(d)
    assert result["values"] == [1.0, None, 3.0]


def test_serialize_tensor_numpy():
    arr = np.array([1.0, 2.0, 3.0])
    result = serialize_tensor(arr)
    assert result == [1.0, 2.0, 3.0]


def test_serialize_tensor_passthrough():
    result = serialize_tensor([1, 2, 3])
    assert result == [1, 2, 3]
