from api.routers.observatory_situation import _key_rows_by_sector


def test_key_rows_by_sector_uses_mapping_and_meta():
    rows = [("S1", 1.0, 0.2, "normale"), ("S2", -2.0, None, "normale"), ("SX", 0.0, 0.0, "normale")]
    code_to_sector = {"S1": 7, "S2": 7}                      # SX unmapped -> dropped
    meta = {7: {"nom": "Craie", "tendancy_coord": "50 3"}}
    keyed = _key_rows_by_sector(rows, code_to_sector, meta)
    assert {k[0] for k in keyed} == {7}                       # only sector 7
    assert keyed[0][1] == "Craie"                             # name from meta
    assert len(keyed) == 2                                    # SX dropped
