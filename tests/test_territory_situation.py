import math
from dashboard.utils.territory_situation import (
    zscore_to_class, aggregate_situation, aggregate_trend,
    MIN_ELIGIBLE, TREND_STABLE_BAND, CLASS_ORDER,
)


def test_zscore_to_class_spans_all_classes():
    assert zscore_to_class(-3.0) == "EXTREMEMENT_BAS"
    assert zscore_to_class(0.0) == "NORMAL"
    assert zscore_to_class(3.0) == "EXTREMEMENT_HAUT"
    idx = [CLASS_ORDER.index(zscore_to_class(z)) for z in [-3, -2, -1, 0, 1, 2, 3]]
    assert idx == sorted(idx)


def test_zscore_to_class_none():
    assert zscore_to_class(None) == "UNKNOWN"


def test_aggregate_situation_uses_median_and_distribution():
    res = aggregate_situation([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert res["situation_class"] == "NORMAL"
    assert res["n_eligible"] == 5
    assert res["pct_below_normal"] == 40.0
    assert sum(res["distribution"].values()) == 5


def test_aggregate_situation_insufficient_coverage():
    res = aggregate_situation([0.0] * (MIN_ELIGIBLE - 1))
    assert res["situation_class"] is None
    assert res["insufficient"] is True


def test_aggregate_trend_directions():
    assert aggregate_trend([TREND_STABLE_BAND + 0.2] * 3) == "hausse"
    assert aggregate_trend([-(TREND_STABLE_BAND + 0.2)] * 3) == "baisse"
    assert aggregate_trend([0.0, 0.0, 0.0]) == "stable"


def test_aggregate_trend_empty():
    assert aggregate_trend([]) is None
