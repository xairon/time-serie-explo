"""COMET-Hydro baseline: raw stress space counterfactual optimization.

Optimizes delta = s_cf - s_obs in R^{L x 3} directly with soft constraints.
No Clausius-Clapeyron coupling -- T and ETP perturbed independently.

This serves as a non-physics-constrained baseline for comparison with PhysCF.
COMET has L x 3 free parameters (vs PhysCF's 7 constrained parameters),
making the comparison of explanation quality meaningful: more parameters
does not necessarily mean better explanations.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .perturbation import PerturbationLayer


def generate_counterfactual_comet(
    h_obs: torch.Tensor,
    s_obs_norm: torch.Tensor,
    model: nn.Module,
    target_bounds: tuple[torch.Tensor, torch.Tensor],
    scaler: dict,
    k_sigma: float = 4.0,
    lambda_smooth: float = 0.1,
    n_iter: int = 1000,
    lr: float = 0.01,
    device: str = "cpu",
) -> dict:
    """Generate counterfactual using COMET-Hydro approach.

    Optimizes raw stress perturbation delta in R^{L x 3}.
    No physics constraints (no CC coupling, no seasonal structure).

    Args:
        h_obs: Normalized gwl lookback tensor (L,).
        s_obs_norm: Normalized stresses tensor (L, 3).
        model: Forecasting model with forward(h_obs, s_obs) -> y_hat.
        target_bounds: Tuple of (lower, upper) target IPS bounds (normalized).
        scaler: Dict of covariate scaler params {col: {mean, std}}.
        k_sigma: Maximum perturbation magnitude in sigma units (default 4.0).
        lambda_smooth: Smoothness regularization weight (default 0.1).
        n_iter: Number of gradient descent iterations (default 1000).
        lr: Learning rate for Adam optimizer (default 0.01).
        device: Torch device string (default "cpu").

    Returns:
        CounterfactualResult dict with standardized keys.
    """
    stress_cols = PerturbationLayer.STRESS_COLUMNS

    h_obs = h_obs.to(device)
    s_obs_norm = s_obs_norm.to(device)
    lower, upper = target_bounds[0].to(device), target_bounds[1].to(device)

    # Save model state to restore later
    model = model.to(device)
    _original_training = model.training
    _original_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}

    # Freeze model weights but keep in train mode for cuDNN RNN backward
    if hasattr(model, "to_train_mode"):
        model.to_train_mode()
    else:
        model.train()
        for p in model.parameters():
            p.requires_grad_(False)

    L = s_obs_norm.shape[0]

    if n_iter <= 0:
        with torch.no_grad():
            y_obs = model(h_obs.unsqueeze(0), s_obs_norm.unsqueeze(0)).squeeze(0)
        # Restore model state
        model.train(_original_training)
        for n, p in model.named_parameters():
            p.requires_grad_(_original_requires_grad.get(n, True))
        return {
            "method": "comet_hydro",
            "y_cf": y_obs.detach().cpu(),
            "s_cf_phys": None,
            "s_cf_norm": s_obs_norm.detach().cpu(),
            "theta_star": None,
            "loss_history": [],
            "target_history": [],
            "prox_history": [],
            "smooth_history": [],
            "converged": False,
            "wall_clock_s": 0.0,
            "n_params": L * len(stress_cols),
            "n_iter": 0,
            "n_trials": None,
            "best_loss": None,
        }

    delta = nn.Parameter(torch.zeros(L, len(stress_cols), device=device))
    optimizer = torch.optim.Adam([delta], lr=lr)

    max_delta = k_sigma

    loss_history = []
    target_history = []
    prox_history = []
    smooth_history = []
    start_time = time.time()

    for i in range(n_iter):
        optimizer.zero_grad()

        delta_clamped = delta.clamp(-max_delta, max_delta)
        s_cf_clamped = s_obs_norm + delta_clamped

        y_cf = model(h_obs.unsqueeze(0), s_cf_clamped.unsqueeze(0)).squeeze(0)

        l_target = torch.mean(torch.relu(lower - y_cf) + torch.relu(y_cf - upper))
        l_prox = (delta ** 2).mean()
        l_smooth = (torch.abs(delta[1:] - delta[:-1])).mean()

        loss = l_target + l_prox + lambda_smooth * l_smooth
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-max_delta, max_delta)

        loss_history.append(loss.item())
        target_history.append(l_target.item())
        prox_history.append(l_prox.item())
        smooth_history.append(l_smooth.item())

    elapsed = time.time() - start_time

    with torch.no_grad():
        delta_clamped = delta.clamp(-max_delta, max_delta)
        s_cf_final = s_obs_norm + delta_clamped
        y_cf_final = model(h_obs.unsqueeze(0), s_cf_final.unsqueeze(0)).squeeze(0)

    # Denormalize s_cf to physical units if scaler available
    s_cf_phys = None
    try:
        s_cf_phys_t = s_cf_final.clone()
        for j, col in enumerate(stress_cols):
            if col in scaler:
                mu = scaler[col]["mean"]
                sigma = scaler[col]["std"]
                if sigma > 0:
                    s_cf_phys_t[..., j] = s_cf_phys_t[..., j] * sigma + mu
        s_cf_phys = s_cf_phys_t.detach().cpu()
    except Exception:
        pass  # s_cf_phys remains None

    # Restore model state
    model.train(_original_training)
    for n, p in model.named_parameters():
        p.requires_grad_(_original_requires_grad.get(n, True))

    converged = target_history[-1] < PerturbationLayer.CONVERGENCE_THRESHOLD

    return {
        "method": "comet_hydro",
        "y_cf": y_cf_final.detach().cpu(),
        "s_cf_phys": s_cf_phys,
        "s_cf_norm": s_cf_final.detach().cpu(),
        "theta_star": None,
        "loss_history": loss_history,
        "target_history": target_history,
        "prox_history": prox_history,
        "smooth_history": smooth_history,
        "converged": converged,
        "wall_clock_s": elapsed,
        "n_params": L * len(stress_cols),
        "n_iter": n_iter,
        "n_trials": None,
        "best_loss": None,
    }
