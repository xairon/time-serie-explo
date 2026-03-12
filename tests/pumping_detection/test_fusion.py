# tests/pumping_detection/test_fusion.py
import pandas as pd
import pytest


class TestFusionEngine:
    def test_two_layer_concordance(self):
        from dashboard.utils.pumping_detection.fusion import FusionEngine
        months = pd.date_range("2015-01-01", periods=24, freq="MS")
        layer1_flags = pd.Series([False]*6 + [True]*6 + [False]*12, index=months)
        layer2_flags = pd.Series([False]*6 + [True]*4 + [False]*14, index=months)
        engine = FusionEngine()
        result = engine.fuse({"pastas": layer1_flags, "xai": layer2_flags})
        assert "suspect_windows" in result
        assert "global_score" in result
        assert 0 <= result["global_score"] <= 1
        high_windows = [w for w in result["suspect_windows"] if w["confidence"] == "high"]
        assert len(high_windows) >= 1

    def test_all_clean_returns_zero_score(self):
        from dashboard.utils.pumping_detection.fusion import FusionEngine
        months = pd.date_range("2015-01-01", periods=12, freq="MS")
        layer1_flags = pd.Series([False]*12, index=months)
        layer2_flags = pd.Series([False]*12, index=months)
        engine = FusionEngine()
        result = engine.fuse({"pastas": layer1_flags, "xai": layer2_flags})
        assert result["global_score"] == 0.0
        assert len(result["suspect_windows"]) == 0

    def test_adapts_to_single_layer(self):
        from dashboard.utils.pumping_detection.fusion import FusionEngine
        months = pd.date_range("2015-01-01", periods=12, freq="MS")
        layer1_flags = pd.Series([False]*3 + [True]*6 + [False]*3, index=months)
        engine = FusionEngine()
        result = engine.fuse({"pastas": layer1_flags})
        assert result["global_score"] > 0
        assert len(result["suspect_windows"]) >= 1
