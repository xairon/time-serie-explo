from api.routers.observatory_meteo import parse_brgm_timeline


def _feat(sid, end, cls, color, tend):
    return {"properties": {"is_parent": True, "sector_id": sid, "end_period": end,
                            "class": cls, "color": color, "tendency": tend, "tendancy_coord": "50 3"}}


def test_groups_by_month_latest_window_wins():
    feats = [
        _feat(1, "2026-04-14T22:00:00Z", 3, "#f8930f", -1),   # April window A
        _feat(1, "2026-04-29T22:00:00Z", 4, "#ffde1a", 0),    # April window B (later) -> wins for 2026-04
        _feat(1, "2026-05-14T22:00:00Z", 5, "#60a3d6", 1),    # May
        _feat(2, "2026-05-14T22:00:00Z", 0, "#d9d9d9", None),
    ]
    out = parse_brgm_timeline(feats)
    assert out["periods"] == ["2026-04", "2026-05"]
    assert out["windows"]["2026-04"]["1"]["brgm_class"] == 4   # later April window won
    assert out["windows"]["2026-04"]["1"]["trend"] == "stable"
    assert out["windows"]["2026-05"]["1"]["color"] == "#60a3d6"
    assert out["windows"]["2026-05"]["2"]["trend"] is None
