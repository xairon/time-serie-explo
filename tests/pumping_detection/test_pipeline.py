"""Integration test for the pumping detection pipeline.

Uses synthetic data with known pumping to validate end-to-end behavior.
"""
import numpy as np
import pandas as pd
import pytest


def _make_full_synthetic_dataset(n_years=5, seed=42):
    """Create a multi-year synthetic dataset with pumping in summers of years 3-4."""
    rng = np.random.default_rng(seed)
    n_days = n_years * 365
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    # Natural piezo: seasonal + noise
    piezo = 50.0 + 3.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 0.3, n_days)

    # Inject pumping: summers of year 3 and 4
    for start in [730 + 150, 1095 + 150]:
        end = min(start + 120, n_days)
        piezo[start:end] -= 1.5

    piezo = pd.Series(piezo, index=dates, name="piezo")
    precip = pd.Series(
        3.0 + 2.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 1, n_days),
        index=dates, name="precip"
    ).clip(lower=0)
    etp = pd.Series(
        2.0 + 1.5 * np.sin(2 * np.pi * (t - 90) / 365.25) + rng.normal(0, 0.3, n_days),
        index=dates, name="etp"
    ).clip(lower=0)

    return piezo, precip, etp


class TestPipelineE2E:
    @pytest.mark.slow
    def test_pipeline_runs_end_to_end(self):
        from dashboard.utils.pumping_detection.pipeline import PumpingDetectionPipeline

        piezo, precip, etp = _make_full_synthetic_dataset()
        events = []

        def capture_emit(event_type, data):
            events.append((event_type, data))

        config = {
            "pastas": {"response_function": "Gamma"},
            "changepoint": {"method": "pelt", "min_segment_length": 60},
            "ml": {"model_type": "TFTModel", "max_epochs": 5, "input_chunk_length": 180, "output_chunk_length": 14},
            "xai": {"methods": ["integrated_gradients"], "window_size": 90, "stride": 30},
            "fusion": {"js_divergence_threshold": 0.3, "merge_gap_days": 30},
        }

        pipeline = PumpingDetectionPipeline(config=config, emit=capture_emit)
        result = pipeline.run(piezo, precip, etp)

        # Basic structure checks
        assert "pastas" in result
        assert "fusion" in result
        assert "global_score" in result["fusion"]
        assert 0 <= result["fusion"]["global_score"] <= 1

        # Should have detected some suspect windows
        suspect = result["fusion"]["suspect_windows"]
        assert len(suspect) >= 1

        # Events should have been emitted
        assert len(events) > 0
        stages = [e[1].get("stage") for e in events if e[0] == "progress"]
        assert "pastas" in stages
