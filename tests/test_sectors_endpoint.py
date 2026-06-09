from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_sectors_route_returns_list_with_sector_shape():
    r = client.get("/api/v1/observatory/situation/sectors?type=piezo")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        e = data[0]
        for k in ("level", "code", "name", "situation_class", "trend", "tendancy_coord"):
            assert k in e
        assert e["level"] == "sector"
