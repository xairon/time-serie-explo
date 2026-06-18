import inspect

from api.routers import observatory_piezo
from api.routers import observatory_hydro


def test_piezo_export_endpoint_exists_and_is_resilient():
    src = inspect.getsource(observatory_piezo.export_csv)
    # joins the index from the materialized table
    assert "gold.fct_monthly_index" in src
    # 404 on unknown station, reusing the dim table existence check
    assert "introuvable" in src
    assert "gold.dim_piezo_stations" in src
    # tolerates a missing index table (pre-materialization)
    assert "ProgrammingError" in src
    # delegates formatting to the pure builder
    assert "build_station_csv" in src


def test_hydro_export_endpoint_converts_flow_and_is_resilient():
    src = inspect.getsource(observatory_hydro.export_csv)
    assert "gold.fct_monthly_index" in src
    assert "gold.dim_hydro_stations" in src
    assert "ProgrammingError" in src
    assert "build_station_csv" in src
    # L/s -> m³/s conversion reused for non-height rows
    assert "_convert_qmnj_row" in src
    assert "_FLOW_COLS_DAILY" in src


def test_piezo_export_route_registered_before_greedy_detail_route():
    from api.routers.observatory_piezo import router
    paths = [r.path for r in router.routes]
    export_idx = next(i for i, p in enumerate(paths) if p.endswith("/export.csv"))
    detail_idx = next(i for i, p in enumerate(paths) if p.endswith("/stations/{code_bss:path}"))
    assert export_idx < detail_idx, (
        "piezo /export.csv must be registered before the greedy {code_bss:path} detail route"
    )
