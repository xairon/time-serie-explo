import json
from pathlib import Path

import pytest

from api.routers.observatory_meteo import parse_brgm_sectors

# ponytail: l'instantane WFS n'est pas versionne, le test se saute donc partout
# sauf sur une machine qui l'a deja telecharge. Le commiter en fixture le rendrait
# executable en CI, au prix d'un fichier de donnees dans le depot.
INSTANTANE = Path("/tmp/bdrv/wfs_2026-05-01.json")


@pytest.mark.skipif(not INSTANTANE.exists(), reason=f"instantane absent : {INSTANTANE}")
def test_parse_brgm_sectors_from_real_snapshot():
    feats = json.loads(INSTANTANE.read_text())["features"]
    out = parse_brgm_sectors(feats)
    assert len(out) == 66
    ids = {o["sector_id"] for o in out}
    assert len(ids) == 66
    sample = out[0]
    assert set(sample) >= {"sector_id", "color", "brgm_class", "trend", "ips", "status", "tendancy_coord"}
    assert all(o["trend"] in ("hausse", "stable", "baisse", None) for o in out)
    # tendency -1 must map to baisse
    assert any(o["trend"] == "baisse" for o in out)
