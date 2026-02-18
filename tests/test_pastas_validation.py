"""Tests for Pastas dual validation module."""

import pytest
import numpy as np
import pandas as pd

from dashboard.utils.counterfactual.pastas_validation import (
    compute_rmse,
    validate_with_pastas,
    cf_stresses_to_pastas_series,
    build_pastas_series_from_data,
    run_dual_validation_for_results,
    PASTAS_AVAILABLE,
)


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

class TestComputeRMSE:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(a, a) == 0.0

    def test_known_value(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert abs(compute_rmse(a, b) - 1.0) < 1e-10

    def test_single_element(self):
        assert compute_rmse(np.array([5.0]), np.array([8.0])) == 3.0


# ---------------------------------------------------------------------------
# validate_with_pastas (mock-based)
# ---------------------------------------------------------------------------

class TestValidateWithPastas:
    def test_accepted_when_agreement(self):
        """CF accepted when TFT-Pastas disagreement is small."""
        y_cf_tft = np.array([100.0, 101.0, 102.0])
        y_factual_tft = np.array([100.0, 100.5, 101.0])
        y_factual_pastas = np.array([100.1, 100.6, 101.1])

        class MockPastas:
            def simulate_with_stresses(self, **kwargs):
                return np.array([100.05, 101.05, 102.05])

        result = validate_with_pastas(
            s_cf_phys={"precip": None, "evap": None},
            pastas_model=MockPastas(),
            y_cf_tft=y_cf_tft,
            y_factual_tft=y_factual_tft,
            y_factual_pastas=y_factual_pastas,
            gamma=1.5,
        )
        assert result["accepted"] is True
        assert result["rmse_cf"] < result["epsilon"]
        assert result["y_cf_pastas"] is not None

    def test_rejected_when_disagreement(self):
        """CF rejected when TFT-Pastas disagreement is large."""
        y_cf_tft = np.array([100.0, 101.0, 102.0])
        y_factual_tft = np.array([100.0, 100.5, 101.0])
        y_factual_pastas = np.array([100.1, 100.6, 101.1])

        class MockPastas:
            def simulate_with_stresses(self, **kwargs):
                return np.array([90.0, 91.0, 92.0])

        result = validate_with_pastas(
            s_cf_phys={"precip": None, "evap": None},
            pastas_model=MockPastas(),
            y_cf_tft=y_cf_tft,
            y_factual_tft=y_factual_tft,
            y_factual_pastas=y_factual_pastas,
            gamma=1.5,
        )
        assert result["accepted"] is False
        assert result["rmse_cf"] > result["epsilon"]

    def test_pastas_error_returns_rejection(self):
        """Pastas error should result in rejection, not crash."""
        class FailPastas:
            def simulate_with_stresses(self, **kwargs):
                raise RuntimeError("Pastas error")

        result = validate_with_pastas(
            s_cf_phys={"precip": None, "evap": None},
            pastas_model=FailPastas(),
            y_cf_tft=np.array([1.0]),
            y_factual_tft=np.array([1.0]),
            y_factual_pastas=np.array([1.0]),
        )
        assert result["accepted"] is False
        assert result["rmse_cf"] == float("inf")
        assert result["y_cf_pastas"] is None

    def test_gamma_controls_tolerance(self):
        """Higher gamma = more tolerant."""
        y_cf_tft = np.array([100.0, 101.0, 102.0])
        y_factual_tft = np.array([100.0, 100.5, 101.0])
        y_factual_pastas = np.array([100.1, 100.6, 101.1])

        class MockPastas:
            def simulate_with_stresses(self, **kwargs):
                return np.array([101.0, 102.0, 103.0])  # moderate disagreement

        # Strict gamma
        r1 = validate_with_pastas(
            s_cf_phys={"precip": None, "evap": None},
            pastas_model=MockPastas(),
            y_cf_tft=y_cf_tft,
            y_factual_tft=y_factual_tft,
            y_factual_pastas=y_factual_pastas,
            gamma=1.0,
        )
        # Lenient gamma
        r2 = validate_with_pastas(
            s_cf_phys={"precip": None, "evap": None},
            pastas_model=MockPastas(),
            y_cf_tft=y_cf_tft,
            y_factual_tft=y_factual_tft,
            y_factual_pastas=y_factual_pastas,
            gamma=20.0,
        )
        assert r2["epsilon"] > r1["epsilon"]
        # With gamma=20, should be accepted
        assert r2["accepted"] is True


# ---------------------------------------------------------------------------
# cf_stresses_to_pastas_series
# ---------------------------------------------------------------------------

class TestCfStressesToPastasSeries:
    def test_returns_precip_evap_series(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        arr = np.random.rand(10, 3)
        result = cf_stresses_to_pastas_series(arr, dates)
        assert "precip" in result
        assert "evap" in result
        assert isinstance(result["precip"], pd.Series)
        assert isinstance(result["evap"], pd.Series)
        assert len(result["precip"]) == 10
        np.testing.assert_array_equal(result["precip"].values, np.clip(arr[:, 0], 0, None))
        np.testing.assert_array_equal(result["evap"].values, np.clip(arr[:, 2], 0, None))

    def test_handles_torch_tensor(self):
        import torch
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        t = torch.rand(5, 3)
        result = cf_stresses_to_pastas_series(t, dates)
        assert len(result["precip"]) == 5
        assert len(result["evap"]) == 5

    def test_clips_negative_values(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        arr = np.array([[-1.0, 5.0, -2.0], [3.0, 4.0, 1.0], [0.5, 3.0, 0.0]])
        result = cf_stresses_to_pastas_series(arr, dates)
        assert (result["precip"].values >= 0).all()
        assert (result["evap"].values >= 0).all()


# ---------------------------------------------------------------------------
# build_pastas_series_from_data
# ---------------------------------------------------------------------------

class TestBuildPastasSeriesFromData:
    @pytest.fixture
    def sample_data(self):
        train_dates = pd.date_range("2020-01-01", periods=70, freq="D")
        test_dates = pd.date_range("2020-03-12", periods=30, freq="D")
        data_dict = {
            "train": pd.DataFrame({"gwl": np.random.randn(70)}, index=train_dates),
            "test": pd.DataFrame({"gwl": np.random.randn(30)}, index=test_dates),
            "train_cov": pd.DataFrame({
                "precipitation": np.random.randn(70),
                "evapotranspiration": np.random.randn(70),
            }, index=train_dates),
            "test_cov": pd.DataFrame({
                "precipitation": np.random.randn(30),
                "evapotranspiration": np.random.randn(30),
            }, index=test_dates),
        }
        physcf_scaler = {
            "precipitation": {"mean": 5.0, "std": 3.0},
            "evapotranspiration": {"mean": 3.0, "std": 1.0},
            "precip": {"mean": 5.0, "std": 3.0},
            "evap": {"mean": 3.0, "std": 1.0},
        }
        return data_dict, physcf_scaler

    def test_extracts_correct_series(self, sample_data):
        data_dict, physcf_scaler = sample_data
        gwl_s, precip_s, evap_s, train_end = build_pastas_series_from_data(
            data_dict, "gwl",
            ["precipitation", "evapotranspiration"],
            mu_target=50.0, sigma_target=2.0,
            physcf_scaler=physcf_scaler,
        )
        assert isinstance(gwl_s, pd.Series)
        assert isinstance(precip_s, pd.Series)
        assert isinstance(evap_s, pd.Series)
        assert gwl_s.name == "gwl"
        assert precip_s.name == "precip"
        assert evap_s.name == "evap"
        # GWL should be denormalized
        assert gwl_s.mean() != 0  # unlikely to be exactly 0 after denorm

    def test_train_end_date(self, sample_data):
        data_dict, physcf_scaler = sample_data
        _, _, _, train_end = build_pastas_series_from_data(
            data_dict, "gwl",
            ["precipitation", "evapotranspiration"],
            mu_target=50.0, sigma_target=2.0,
            physcf_scaler=physcf_scaler,
        )
        expected = str(data_dict["train"].index.max().date())
        assert train_end == expected

    def test_raises_on_missing_precip(self, sample_data):
        data_dict, physcf_scaler = sample_data
        with pytest.raises(ValueError, match="precipitation"):
            build_pastas_series_from_data(
                data_dict, "gwl",
                ["temperature", "wind_speed"],
                mu_target=50.0, sigma_target=2.0,
                physcf_scaler=physcf_scaler,
            )

    def test_raises_on_missing_evap(self, sample_data):
        data_dict, physcf_scaler = sample_data
        with pytest.raises(ValueError, match="evaporation"):
            build_pastas_series_from_data(
                data_dict, "gwl",
                ["precipitation", "temperature"],
                mu_target=50.0, sigma_target=2.0,
                physcf_scaler=physcf_scaler,
            )

    def test_non_negative_outputs(self, sample_data):
        data_dict, physcf_scaler = sample_data
        _, precip_s, evap_s, _ = build_pastas_series_from_data(
            data_dict, "gwl",
            ["precipitation", "evapotranspiration"],
            mu_target=50.0, sigma_target=2.0,
            physcf_scaler=physcf_scaler,
        )
        assert (precip_s >= 0).all(), "Precipitation must be non-negative"
        assert (evap_s >= 0).all(), "Evaporation must be non-negative"


# ---------------------------------------------------------------------------
# run_dual_validation_for_results (mock-based)
# ---------------------------------------------------------------------------

class TestRunDualValidation:
    def test_validates_all_methods(self):
        class MockPastas:
            def predict(self, tmin=None, tmax=None):
                return np.array([50.0, 50.5, 51.0])
            def simulate_with_stresses(self, **kwargs):
                return np.array([50.1, 50.6, 51.1])

        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        results = {
            "PhysCF (gradient)": {
                "s_cf_phys": np.random.rand(30, 3),
                "y_cf": np.array([0.1, 0.2, 0.3]),
                "method_key": "physcf_gradient",
            },
            "COMET-Hydro": {
                "s_cf_phys": np.random.rand(30, 3),
                "y_cf": np.array([0.15, 0.25, 0.35]),
                "method_key": "comet_hydro",
            },
        }

        validation = run_dual_validation_for_results(
            results_dict=results,
            pastas_model=MockPastas(),
            lookback_dates=dates,
            y_factual_tft_raw=np.array([50.0, 50.5, 51.0]),
            horizon_start="2020-01-31",
            horizon_end="2020-03-01",
            mu_target=50.0,
            sigma_target=2.0,
            gamma=1.5,
        )
        assert "PhysCF (gradient)" in validation
        assert "COMET-Hydro" in validation
        assert "accepted" in validation["PhysCF (gradient)"]
        assert "rmse_cf" in validation["PhysCF (gradient)"]

    def test_skips_methods_without_s_cf_phys(self):
        class MockPastas:
            def predict(self, tmin=None, tmax=None):
                return np.array([50.0])

        results = {
            "incomplete": {"y_cf": np.array([0.1])},
        }
        validation = run_dual_validation_for_results(
            results_dict=results,
            pastas_model=MockPastas(),
            lookback_dates=pd.date_range("2020-01-01", periods=5, freq="D"),
            y_factual_tft_raw=np.array([50.0]),
            horizon_start="2020-01-06",
            horizon_end="2020-01-10",
            mu_target=50.0,
            sigma_target=2.0,
        )
        assert "incomplete" not in validation


# ---------------------------------------------------------------------------
# Integration tests (require actual pastas installation)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PASTAS_AVAILABLE, reason="pastas not installed")
class TestPastasWrapperIntegration:
    def test_fit_and_predict(self):
        from dashboard.utils.counterfactual.pastas_validation import PastasWrapper

        dates = pd.date_range("2000-01-01", periods=3650, freq="D")
        gwl = pd.Series(
            np.sin(np.arange(3650) * 2 * np.pi / 365) * 2 + 50,
            index=dates, name="gwl",
        )
        precip = pd.Series(np.random.rand(3650) * 10, index=dates, name="precip")
        evap = pd.Series(np.random.rand(3650) * 5, index=dates, name="evap")

        w = PastasWrapper()
        w.fit(gwl, precip, evap)
        assert w._fitted
        pred = w.predict()
        assert len(pred) > 0

        params = w.get_response_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_simulate_with_stresses(self):
        from dashboard.utils.counterfactual.pastas_validation import PastasWrapper

        dates = pd.date_range("2000-01-01", periods=3650, freq="D")
        gwl = pd.Series(
            np.sin(np.arange(3650) * 2 * np.pi / 365) * 2 + 50,
            index=dates, name="gwl",
        )
        precip = pd.Series(np.random.rand(3650) * 10, index=dates, name="precip")
        evap = pd.Series(np.random.rand(3650) * 5, index=dates, name="evap")

        w = PastasWrapper()
        w.fit(gwl, precip, evap)

        cf_precip = precip * 0.7  # -30% precipitation
        sim = w.simulate_with_stresses(cf_precip, evap)
        assert len(sim) > 0

    def test_not_fitted_raises(self):
        from dashboard.utils.counterfactual.pastas_validation import PastasWrapper
        w = PastasWrapper()
        with pytest.raises(RuntimeError, match="not fitted"):
            w.predict()
        with pytest.raises(RuntimeError, match="not fitted"):
            w.simulate_with_stresses(pd.Series(), pd.Series())
        with pytest.raises(RuntimeError, match="not fitted"):
            w.get_response_params()
