# tests/pumping_detection/test_xai_layer.py
import numpy as np
import pytest


class TestDriftMetrics:
    def test_js_divergence_identical_distributions(self):
        from dashboard.utils.pumping_detection.xai_layer import js_divergence
        p = np.array([0.5, 0.3, 0.2])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_js_divergence_different_distributions(self):
        from dashboard.utils.pumping_detection.xai_layer import js_divergence
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.1, 0.8])
        jsd = js_divergence(p, q)
        assert 0 < jsd <= np.log(2)

    def test_feature_agreement_perfect_overlap(self):
        from dashboard.utils.pumping_detection.xai_layer import feature_agreement
        ranking_a = [0, 1, 2, 3, 4]
        ranking_b = [0, 1, 2, 3, 4]
        assert feature_agreement(ranking_a, ranking_b, k=3) == 1.0

    def test_feature_agreement_no_overlap(self):
        from dashboard.utils.pumping_detection.xai_layer import feature_agreement
        ranking_a = [0, 1, 2, 3, 4]
        ranking_b = [4, 3, 2, 1, 0]
        fa = feature_agreement(ranking_a, ranking_b, k=2)
        assert fa == 0.0

    def test_compute_window_drift(self):
        from dashboard.utils.pumping_detection.xai_layer import compute_window_drift
        ref_attrs = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2]])
        test_attrs = np.array([[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.2, 0.1, 0.7]])
        drift = compute_window_drift(ref_attrs, test_attrs)
        assert "js_divergence" in drift
        assert "spearman_corr" in drift
        assert "feature_agreement" in drift
        assert drift["js_divergence"] > 0.1
