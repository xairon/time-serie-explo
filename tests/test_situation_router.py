from api.routers.observatory_situation import router, _eligible_rows_to_territories


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
