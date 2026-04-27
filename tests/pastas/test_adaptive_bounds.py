"""Tests for adaptive pumping bounds computation."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from dashboard.utils.pastas.scenario_presets import (
    HARD_DRAWDOWN_M,
    SOFT_DRAWDOWN_M,
    _compute_t95,
    compute_adaptive_bounds,
)


class TestComputeT95:
    def test_empty_returns_zero(self):
        assert _compute_t95(np.array([]), np.array([])) == 0.0

    def test_zero_final_returns_zero(self):
        assert _compute_t95(np.array([0.0, 0.0, 0.0]), np.array([0, 1, 2])) == 0.0

    def test_immediate_response(self):
        values = np.array([0.0, 1.0, 1.0, 1.0])
        index = np.array([0, 1, 2, 3])
        assert _compute_t95(values, index) == 1.0

    def test_gradual_response(self):
        values = np.array([0.0, 0.3, 0.6, 0.9, 0.96, 1.0])
        index = np.array([0, 100, 200, 300, 400, 500])
        assert _compute_t95(values, index) == 400.0

    def test_negative_gain(self):
        values = np.array([0.0, -0.3, -0.6, -0.96, -1.0])
        index = np.array([0, 100, 200, 300, 400])
        assert _compute_t95(values, index) == 300.0

    def test_never_reaches_threshold(self):
        values = np.array([0.0, 0.1, 0.2, 0.5])
        index = np.array([0, 100, 200, 300])
        assert _compute_t95(values, index) == 300.0


def _make_step_response(values, index):
    sr = MagicMock()
    sr.values = np.array(values)
    sr.index = np.array(index)
    sr.__len__ = lambda self: len(values)
    return sr


def _mock_io(model):
    """Inject a mock io module so compute_adaptive_bounds can import load_model."""
    from unittest.mock import patch
    mock_io = ModuleType("dashboard.utils.pastas.io")
    mock_io.load_model = MagicMock(return_value=model)
    return patch.dict(sys.modules, {"dashboard.utils.pastas.io": mock_io})


def _make_sm(type_name, up=True):
    """Create a mock stressmodel with a given type name."""
    sm = MagicMock()
    sm.__class__ = type(type_name, (), {})
    sm.up = up
    return sm


class TestComputeAdaptiveBounds:
    def test_returns_none_when_no_stressmodels(self):
        model = MagicMock()
        model.stressmodels = {}
        with _mock_io(model):
            assert compute_adaptive_bounds("run123") is None

    def test_returns_none_when_gain_is_zero(self):
        model = MagicMock()
        model.stressmodels = {"recharge": _make_sm("RechargeModel")}
        model.get_step_response.return_value = _make_step_response([0.0, 0.0], [0, 100])
        with _mock_io(model):
            assert compute_adaptive_bounds("run123") is None

    def test_recharge_model_no_q_bounds(self):
        model = MagicMock()
        model.stressmodels = {"recharge": _make_sm("RechargeModel")}
        model.get_step_response.return_value = _make_step_response(
            [0.0, 0.05, 0.08, 0.095, 0.1], [0, 100, 200, 300, 400]
        )
        with _mock_io(model):
            result = compute_adaptive_bounds("run123")
        assert result is not None
        assert result.source == "recharge_model"
        assert result.Q_soft is None
        assert result.Q_hard is None
        assert result.gain_A == pytest.approx(0.1)
        assert result.t95_days == 300.0

    def test_well_model_computes_q_bounds(self):
        model = MagicMock()
        model.stressmodels = {
            "recharge": _make_sm("RechargeModel"),
            "well_1": _make_sm("StressModel", up=False),
        }

        def get_step(name):
            if name == "well_1":
                return _make_step_response(
                    [0.0, -0.001, -0.0019, -0.002], [0, 200, 400, 600]
                )
            return _make_step_response([0.0, 0.05, 0.1], [0, 100, 200])

        model.get_step_response = get_step
        with _mock_io(model):
            result = compute_adaptive_bounds("run123")
        assert result is not None
        assert result.source == "calibrated_well"
        assert result.gain_A == pytest.approx(-0.002)
        assert result.Q_soft == pytest.approx(SOFT_DRAWDOWN_M / 0.002)
        assert result.Q_hard == pytest.approx(HARD_DRAWDOWN_M / 0.002)

    def test_t_final_days_shorter_than_equilibrium(self):
        model = MagicMock()
        model.stressmodels = {"well_1": _make_sm("StressModel", up=False)}
        model.get_step_response.return_value = _make_step_response(
            [0.0, -0.001, -0.0015, -0.0019, -0.002], [0, 200, 400, 600, 800]
        )
        with _mock_io(model):
            result = compute_adaptive_bounds("run123", t_final_days=300)
        assert result is not None
        assert result.t_final_days == 300
        assert abs(result.step_response_at_t) < abs(result.gain_A)
        assert result.Q_soft is not None
        assert result.Q_soft > SOFT_DRAWDOWN_M / 0.002

    def test_step_response_exception_returns_none(self):
        model = MagicMock()
        model.stressmodels = {"recharge": _make_sm("RechargeModel")}
        model.get_step_response.side_effect = Exception("model error")
        with _mock_io(model):
            assert compute_adaptive_bounds("run123") is None

    def test_well_model_with_recharge_first(self):
        """Well detection works even if recharge comes first in dict."""
        model = MagicMock()
        model.stressmodels = {
            "recharge": _make_sm("RechargeModel"),
            "pumping": _make_sm("WellModel", up=False),
        }
        model.get_step_response.return_value = _make_step_response(
            [0.0, -0.005, -0.01], [0, 100, 200]
        )
        with _mock_io(model):
            result = compute_adaptive_bounds("run123")
        assert result is not None
        assert result.source == "calibrated_well"
