from api.routers.observatory_situation import (
    router, _eligible_rows_to_territories, _coerce_grid,
)
from dashboard.utils.reference import value_to_zscore


def test_coerce_grid_parses_json_string_and_passes_through_list():
    # driver may return the quantile_grid as a JSON string; must become a list
    assert _coerce_grid("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]
    assert _coerce_grid([1.0, 2.0]) == [1.0, 2.0]
    assert _coerce_grid(None) is None


def test_coerce_grid_output_feeds_value_to_zscore_without_error():
    # regression: list("[1.0,...]") used to explode np.interp with a ValueError.
    # The grid holds one value per percentile 1..99 (see reference.PCTL_GRID).
    import json
    grid_json = json.dumps([float(p) for p in range(1, 100)])  # value == percentile
    z = value_to_zscore(50.0, _coerce_grid(grid_json))
    assert z is not None and abs(z) < 0.1  # 50th percentile -> ~0 sigma


def test_router_mounts_situation_paths():
    paths = {r.path for r in router.routes}
    assert "/api/v1/observatory/situation/national" in paths
    assert "/api/v1/observatory/situation/territories" in paths


def test_eligible_rows_to_territories_groups_and_aggregates():
    rows = [
        ("24", "Centre-Val de Loire", -2.0, -0.9, "normale"),
        ("24", "Centre-Val de Loire", -1.0, -0.7, "normale"),
        ("24", "Centre-Val de Loire", 0.0, -0.6, "adaptee"),
        ("24", "Centre-Val de Loire", None, None, "provisoire"),
        ("11", "Île-de-France", 1.5, 0.0, "normale"),
    ]
    out = _eligible_rows_to_territories(rows, level="region", type_="piezo")
    by_code = {t["code"]: t for t in out}
    assert by_code["24"]["n_eligible"] == 3
    assert by_code["24"]["n_provisoire"] == 1
    assert by_code["24"]["trend"] == "baisse"
    # median of eligible z [-2.0, -1.0, 0.0] is -1.0 -> BAS under the fixed
    # BRGM cutoffs (see dashboard.utils.territory_situation.zscore_to_class).
    assert by_code["24"]["situation_class"] == "BAS"
    assert by_code["11"]["insufficient"] is True
    assert by_code["11"]["outlook"] is None
