"""Pumping detection pipeline — unsupervised 3-layer hybrid detection."""

__all__ = [
    "PastasAnalyzer",
    "ChangepointDetector",
    "CleanPeriodSelector",
    "MLAnalyzer",
    "XAIDriftAnalyzer",
    "EmbeddingAnalyzer",
    "FusionEngine",
    "PumpingDetectionPipeline",
    "BNPEClient",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading heavy dependencies at package import time."""
    _imports = {
        "PastasAnalyzer": "dashboard.utils.pumping_detection.pastas_layer",
        "ChangepointDetector": "dashboard.utils.pumping_detection.changepoint",
        "CleanPeriodSelector": "dashboard.utils.pumping_detection.clean_period",
        "MLAnalyzer": "dashboard.utils.pumping_detection.ml_layer",
        "XAIDriftAnalyzer": "dashboard.utils.pumping_detection.xai_layer",
        "EmbeddingAnalyzer": "dashboard.utils.pumping_detection.embedding_layer",
        "FusionEngine": "dashboard.utils.pumping_detection.fusion",
        "PumpingDetectionPipeline": "dashboard.utils.pumping_detection.pipeline",
        "BNPEClient": "dashboard.utils.pumping_detection.bnpe_client",
    }
    if name in _imports:
        import importlib
        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
