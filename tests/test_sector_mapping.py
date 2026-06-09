from api.services.sector_mapping import build_mapping

SECTORS = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"sector_id": 1, "nom": "A", "tendancy_coord": "0.5 0.5"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}},
    {"type": "Feature", "properties": {"sector_id": 2, "nom": "B", "tendancy_coord": "5.5 5.5"},
     "geometry": {"type": "Polygon", "coordinates": [[[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]]}},
]}

def test_build_mapping_assigns_station_to_containing_sector():
    stations = [("S1", 0.5, 0.5), ("S2", 5.5, 5.5), ("S3", 9.0, 9.0)]
    code_to_sector, meta = build_mapping(SECTORS, stations)
    assert code_to_sector["S1"] == 1
    assert code_to_sector["S2"] == 2
    assert "S3" not in code_to_sector
    assert meta[1]["nom"] == "A" and meta[1]["tendancy_coord"] == "0.5 0.5"
