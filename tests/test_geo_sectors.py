from dashboard.utils.geo_sectors import point_in_ring, point_in_geometry, dominant_label

SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]}
MULTI = {"type": "MultiPolygon", "coordinates": [[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                                                  [[[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]]]}

def test_point_in_ring_inside_and_outside():
    ring = [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]
    assert point_in_ring(1, 1, ring) is True
    assert point_in_ring(3, 3, ring) is False

def test_point_in_geometry_polygon_and_multipolygon():
    assert point_in_geometry(1.0, 1.0, SQUARE) is True
    assert point_in_geometry(9.0, 9.0, SQUARE) is False
    assert point_in_geometry(5.5, 5.5, MULTI) is True
    assert point_in_geometry(3.0, 3.0, MULTI) is False

def test_dominant_label_picks_most_frequent_nonempty():
    assert dominant_label(["Craie", "Craie", "Alluvions", None, ""]) == "Craie"

def test_dominant_label_returns_none_when_all_empty():
    assert dominant_label([None, "", "  "]) is None
