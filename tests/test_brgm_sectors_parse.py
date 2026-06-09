import json
from api.routers.observatory_meteo import parse_brgm_sectors


def test_parse_brgm_sectors_from_real_snapshot():
    feats = json.load(open("/tmp/bdrv/wfs_2026-05-01.json"))["features"]
    out = parse_brgm_sectors(feats)
    assert len(out) == 66
    ids = {o["sector_id"] for o in out}
    assert len(ids) == 66
    sample = out[0]
    assert set(sample) >= {"sector_id", "color", "brgm_class", "trend", "ips", "status", "tendancy_coord"}
    assert all(o["trend"] in ("hausse", "stable", "baisse", None) for o in out)
    # tendency -1 must map to baisse
    assert any(o["trend"] == "baisse" for o in out)
