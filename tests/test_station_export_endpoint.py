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


def test_export_endpoints_emit_utf8_bom_for_excel():
    # utf-8-sig prepends a BOM so Excel stops decoding the file as Windows-1252
    # (which turned "Département" into "DÃ©partement").
    for mod in (observatory_piezo, observatory_hydro):
        src = inspect.getsource(mod.export_csv)
        assert "utf-8-sig" in src, f"{mod.__name__}.export_csv should encode body as utf-8-sig"


def test_export_endpoints_accept_and_apply_date_range():
    import inspect as _inspect

    for mod in (observatory_piezo, observatory_hydro):
        sig = _inspect.signature(mod.export_csv)
        assert "start_date" in sig.parameters, f"{mod.__name__}.export_csv missing start_date"
        assert "end_date" in sig.parameters, f"{mod.__name__}.export_csv missing end_date"
        src = _inspect.getsource(mod.export_csv)
        # the daily chronique query is bounded by the optional range
        assert "date >= :start_date" in src
        assert "date <= :end_date" in src


def test_piezo_export_route_registered_before_greedy_detail_route():
    from api.routers.observatory_piezo import router
    paths = [r.path for r in router.routes]
    export_idx = next(i for i, p in enumerate(paths) if p.endswith("/export.csv"))
    detail_idx = next(i for i, p in enumerate(paths) if p.endswith("/stations/{code_bss:path}"))
    assert export_idx < detail_idx, (
        "piezo /export.csv must be registered before the greedy {code_bss:path} detail route"
    )
