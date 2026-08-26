from fastapi.testclient import TestClient
from api.main import app
import pytest

# exige gold.dim_piezo_stations, donc un entrepot peuple
pytestmark = pytest.mark.integration

client = TestClient(app)

def test_sectors_timeline_shape():
    r = client.get("/api/v1/observatory/situation/sectors/timeline?type=piezo")
    assert r.status_code == 200
    d = r.json()
    assert set(d.keys()) >= {"periods", "sectors", "trends"}
    assert isinstance(d["periods"], list)
    assert isinstance(d["sectors"], dict)
