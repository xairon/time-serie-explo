import pytest
from unittest.mock import patch, MagicMock


class TestBNPEClient:
    def test_fetch_nearby_returns_list(self):
        from dashboard.utils.pumping_detection.bnpe_client import BNPEClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"code_ouvrage": "OPR001", "nom_ouvrage": "Forage A", "latitude": 48.0, "longitude": 2.0}],
            "count": 1,
        }

        client = BNPEClient()
        with patch("dashboard.utils.pumping_detection.bnpe_client.requests.get", return_value=mock_response):
            result = client.fetch_nearby(lat=48.0, lon=2.0, radius_km=5)

        assert result["bnpe_available"] is True
        assert len(result["ouvrages"]) == 1
        assert result["ouvrages"][0]["code_ouvrage"] == "OPR001"

    def test_fetch_nearby_timeout(self):
        import dashboard.utils.pumping_detection.bnpe_client as bnpe_module
        from dashboard.utils.pumping_detection.bnpe_client import BNPEClient
        import requests

        # Clear module-level cache to avoid collision with previous test
        bnpe_module._cache.clear()

        client = BNPEClient(timeout=1)
        with patch("dashboard.utils.pumping_detection.bnpe_client.requests.get", side_effect=requests.Timeout):
            result = client.fetch_nearby(lat=48.0, lon=2.0, radius_km=5)

        assert result["bnpe_available"] is False
        assert result["ouvrages"] == []
