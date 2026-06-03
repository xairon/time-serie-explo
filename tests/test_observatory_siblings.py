"""Unit tests for BDLISA grouping helpers in observatory_piezo router."""
from api.routers.observatory_piezo import _bdlisa_primary, _bdlisa_system_prefix


def test_bdlisa_primary_single_code():
    assert _bdlisa_primary("101AC01") == "101AC01"


def test_bdlisa_primary_takes_first_of_list():
    assert _bdlisa_primary("101AC01,123AK03") == "101AC01"
    assert _bdlisa_primary("101AC01, 123AK03") == "101AC01"


def test_bdlisa_primary_empty_is_none():
    assert _bdlisa_primary(None) is None
    assert _bdlisa_primary("") is None
    assert _bdlisa_primary("   ") is None


def test_bdlisa_system_prefix_strips_entity_suffix():
    assert _bdlisa_system_prefix("101AC01") == "101AC"
    assert _bdlisa_system_prefix("121BD01") == "121BD"
    assert _bdlisa_system_prefix("139AM15") == "139AM"


def test_bdlisa_system_prefix_keeps_bare_system():
    assert _bdlisa_system_prefix("101AC") == "101AC"


def test_bdlisa_system_prefix_none_and_fallback():
    assert _bdlisa_system_prefix(None) is None
    assert _bdlisa_system_prefix("") is None
    # No regex match → return the primary code unchanged
    assert _bdlisa_system_prefix("WEIRD") == "WEIRD"
