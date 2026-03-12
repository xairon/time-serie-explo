import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from dashboard.utils.pumping_detection.ml_layer import MLAnalyzer


class TestFilterToClean:
    def test_longest_clean_segment(self):
        analyzer = MLAnalyzer()
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        mask = pd.Series(False, index=dates)
        mask.iloc[10:30] = True
        mask.iloc[40:90] = True

        ts = MagicMock()
        result_ts = MagicMock()
        ts.slice.return_value = result_ts

        result = analyzer._filter_to_clean(ts, mask)
        assert result is result_ts
        call_args = ts.slice.call_args[0]
        assert call_args[0] == pd.Timestamp("2020-02-10")
        assert call_args[1] == pd.Timestamp("2020-03-30")

    def test_no_clean_segment_returns_none(self):
        analyzer = MLAnalyzer()
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        mask = pd.Series(False, index=dates)
        result = analyzer._filter_to_clean(None, mask)
        assert result is None


class TestMLAnalyzerIntegration:
    @pytest.mark.slow
    def test_train_transient_model_returns_model(self):
        pytest.skip("Integration test — run manually with GPU")
