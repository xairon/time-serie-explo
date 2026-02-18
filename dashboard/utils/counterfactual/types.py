"""Shared type definitions for counterfactual analysis results."""

from __future__ import annotations

from typing import Optional, TypedDict

import torch


class CounterfactualResult(TypedDict, total=False):
    """Standardized result dict returned by all CF methods.

    Required keys (present in all methods):
        method, y_cf, s_cf_norm, loss_history, converged, wall_clock_s, n_params

    Optional keys (method-specific):
        s_cf_phys: Physical stresses (PhysCF only, None for COMET)
        theta_star: Interpretable params (PhysCF only, None for COMET)
        target_history: Per-iteration target loss
        prox_history: Per-iteration proximity loss (gradient only)
        n_iter: Number of iterations (gradient/COMET)
        n_trials: Number of Optuna trials
        best_loss: Best loss found (Optuna)
    """

    method: str
    y_cf: torch.Tensor
    s_cf_phys: Optional[torch.Tensor]
    s_cf_norm: torch.Tensor
    theta_star: Optional[dict]
    loss_history: list
    target_history: Optional[list]
    prox_history: Optional[list]
    converged: bool
    wall_clock_s: float
    n_params: int
    n_iter: Optional[int]
    n_trials: Optional[int]
    best_loss: Optional[float]
