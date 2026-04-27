"""Shared utility functions for the API layer."""
from __future__ import annotations


def force_cpu_if_needed(model):
    """Override trainer_params to use CPU if CUDA is not available."""
    import torch
    if not torch.cuda.is_available() and hasattr(model, "trainer_params"):
        model.trainer_params["accelerator"] = "cpu"
        model.trainer_params.pop("devices", None)
