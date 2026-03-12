"""Pumping detection pipeline orchestrator.

Runs Layer 1 → (Layer 2 + Layer 3 in parallel) → Fusion.
Emits SSE-compatible progress events via a callback.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import pandas as pd

from dashboard.utils.pumping_detection.pastas_layer import PastasAnalyzer
from dashboard.utils.pumping_detection.changepoint import ChangepointDetector
from dashboard.utils.pumping_detection.clean_period import CleanPeriodSelector
from dashboard.utils.pumping_detection.ml_layer import MLAnalyzer
from dashboard.utils.pumping_detection.xai_layer import XAIDriftAnalyzer
from dashboard.utils.pumping_detection.embedding_layer import EmbeddingAnalyzer
from dashboard.utils.pumping_detection.fusion import FusionEngine

logger = logging.getLogger(__name__)


class PumpingDetectionPipeline:
    """Orchestrate the 3-layer pumping detection pipeline."""

    def __init__(self, config: dict[str, Any], emit: Callable | None = None):
        self.config = config
        self.emit = emit or (lambda *a, **kw: None)

    def run(
        self,
        piezo: pd.Series,
        precip: pd.Series,
        etp: pd.Series,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline. Returns complete results dict."""
        results: dict[str, Any] = {}
        stop = stop_event or threading.Event()

        # --- Layer 1: Physics ---
        self.emit("progress", {"stage": "pastas", "pct": 0.10, "message": "Calibrating Pastas..."})
        if stop.is_set():
            return {"cancelled": True, "partial": results}

        pastas_cfg = self.config.get("pastas", {})
        analyzer = PastasAnalyzer(**pastas_cfg)
        try:
            pastas_result = analyzer.analyze(piezo, precip, etp)
            results["pastas"] = pastas_result
            self.emit("metrics", {"stage": "pastas", "partial_result": {
                "evp": pastas_result["pastas_fit_quality"]["evp"],
                "rmse": pastas_result["pastas_fit_quality"]["rmse"],
            }})
        except Exception as e:
            logger.error(f"Layer 1 (Pastas) failed: {e}")
            results["pastas"] = {"error": str(e)}

        # Change points
        self.emit("progress", {"stage": "changepoint", "pct": 0.20, "message": "Detecting change points..."})
        if stop.is_set():
            return {"cancelled": True, "partial": results}

        cp_cfg = self.config.get("changepoint", {})
        try:
            detector = ChangepointDetector(**cp_cfg)
            residuals = results.get("pastas", {}).get("residuals", pd.Series(dtype=float))
            cp_result = detector.detect(residuals)
            results["changepoints"] = cp_result
            self.emit("metrics", {"stage": "changepoint", "partial_result": {
                "n_changepoints": len(cp_result.get("changepoints", [])),
            }})
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            results["changepoints"] = {"error": str(e)}

        # Clean period selection
        self.emit("progress", {"stage": "clean", "pct": 0.30, "message": "Selecting clean periods..."})
        clean_cfg = self.config.get("ml", {})
        selector = CleanPeriodSelector(
            n_sigma=clean_cfg.get("clean_residual_threshold", 2.0)
            if isinstance(clean_cfg.get("clean_residual_threshold"), (int, float))
            else 2.0
        )
        residuals = results.get("pastas", {}).get("residuals", pd.Series(dtype=float))
        clean_result = selector.select(residuals) if len(residuals) > 0 else {
            "mask": pd.Series(dtype=bool), "n_clean_days": 0, "pct_clean": 0
        }
        results["clean_periods"] = {k: v for k, v in clean_result.items() if k != "mask"}
        clean_mask = clean_result.get("mask", pd.Series(dtype=bool))
        self.emit("metrics", {"stage": "clean", "partial_result": {
            "n_clean_days": clean_result.get("n_clean_days", 0),
            "pct_clean": clean_result.get("pct_clean", 0),
        }})

        # --- Layer 2 + 3 in parallel ---
        layer2_result: dict[str, Any] = {}
        layer3_result: dict[str, Any] = {}

        def run_layer2():
            nonlocal layer2_result
            if stop.is_set():
                return
            self.emit("progress", {"stage": "ml_train", "pct": 0.45, "message": "Training TFT on clean data..."})
            try:
                ml_cfg = self.config.get("ml", {})
                ml_analyzer = MLAnalyzer(
                    model_type=ml_cfg.get("model_type", "TFTModel"),
                    input_chunk_length=ml_cfg.get("input_chunk_length", 365),
                    output_chunk_length=ml_cfg.get("output_chunk_length", 30),
                    max_epochs=ml_cfg.get("max_epochs", 100),
                )
                # Convert to Darts TimeSeries for ML
                from darts import TimeSeries
                target_ts = TimeSeries.from_series(piezo)
                cov_df = pd.DataFrame({"precip": precip, "temp": etp}, index=piezo.index)
                cov_ts = TimeSeries.from_dataframe(cov_df)

                ml_result = ml_analyzer.train_and_predict(target_ts, cov_ts, clean_mask, stop)
                layer2_result = ml_result

                if "error" not in ml_result:
                    self.emit("progress", {"stage": "xai", "pct": 0.65, "message": "Computing XAI attributions..."})
                    xai_cfg = self.config.get("xai", {})
                    xai_analyzer = XAIDriftAnalyzer(**xai_cfg)
                    xai_result = xai_analyzer.analyze(
                        model=ml_result.get("model"),
                        series=target_ts,
                        covariates=cov_ts,
                        clean_mask=clean_mask,
                        feature_names=list(cov_df.columns),
                    )
                    layer2_result["xai"] = xai_result
            except Exception as e:
                logger.error(f"Layer 2 (ML+XAI) failed: {e}")
                layer2_result = {"error": str(e)}

        def run_layer3():
            nonlocal layer3_result
            if stop.is_set():
                return
            self.emit("progress", {"stage": "embedding", "pct": 0.80, "message": "Analyzing embeddings..."})
            emb_cfg = self.config.get("embeddings", {})
            emb_analyzer = EmbeddingAnalyzer(**emb_cfg)
            layer3_result = emb_analyzer.analyze(piezo)

        t2 = threading.Thread(target=run_layer2)
        t3 = threading.Thread(target=run_layer3)
        t2.start()
        t3.start()
        t2.join()
        t3.join()

        results["ml_xai"] = layer2_result
        results["embeddings"] = layer3_result

        # --- Fusion ---
        self.emit("progress", {"stage": "fusion", "pct": 0.90, "message": "Computing fusion scores..."})
        if stop.is_set():
            return {"cancelled": True, "partial": results}

        layer_flags = self._build_monthly_flags(results, piezo.index)
        fusion_cfg = self.config.get("fusion", {})
        engine = FusionEngine(merge_gap_days=fusion_cfg.get("merge_gap_days", 30))
        fusion_result = engine.fuse(layer_flags)
        results["fusion"] = fusion_result

        return results

    def _build_monthly_flags(self, results: dict, date_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
        """Convert layer results to per-month boolean flags for fusion."""
        months = pd.date_range(date_index.min(), date_index.max(), freq="MS")
        flags: dict[str, pd.Series] = {}

        # Layer 1: Pastas — flag months with significant ACF
        pastas = results.get("pastas", {})
        if "acf_stats" in pastas:
            lb_pval = pastas["acf_stats"].get("ljung_box_pvalue", 1.0)
            acf_sig = self.config.get("fusion", {}).get("acf_significance", 0.05)
            if lb_pval < acf_sig:
                pastas_flags = pd.Series(False, index=months)
                for cp in results.get("changepoints", {}).get("changepoints", []):
                    cp_date = pd.Timestamp(cp["date"])
                    for m in months:
                        if abs((m - cp_date).days) < 90:
                            pastas_flags.loc[m] = True
                flags["pastas"] = pastas_flags

        # Layer 2: XAI — flag months with high JS divergence
        xai = results.get("ml_xai", {}).get("xai", {})
        if xai.get("drift_metrics"):
            js_thresh = self.config.get("fusion", {}).get("js_divergence_threshold", 0.3)
            xai_flags = pd.Series(False, index=months)
            for dm in xai["drift_metrics"]:
                if dm.get("js_divergence", 0) > js_thresh:
                    date = pd.Timestamp(dm["window_date"])
                    nearest_month = months[months.get_indexer([date], method="nearest")[0]]
                    xai_flags.loc[nearest_month] = True
            flags["xai"] = xai_flags

        # Layer 3: Embeddings — skip if not available
        emb = results.get("embeddings", {})
        if emb.get("available") and emb.get("drift_scores") is not None:
            emb_flags = pd.Series(False, index=months)
            flags["embeddings"] = emb_flags

        return flags
