"""COMET-Hydro baseline: raw stress space counterfactual optimization.

Optimizes delta = s_cf - s_obs in R^{L x 3} directly with soft constraints.
No Clausius-Clapeyron coupling -- T and ETP perturbed independently.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


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
    """
    h_obs = h_obs.to(device)
    s_obs_norm = s_obs_norm.to(device)
    lower, upper = target_bounds[0].to(device), target_bounds[1].to(device)

    # Freeze model weights but keep in train mode for cuDNN RNN backward
    model = model.to(device)
    if hasattr(model, "to_train_mode"):
        model.to_train_mode()
    else:
        model.train()
        for p in model.parameters():
            p.requires_grad_(False)

    L = s_obs_norm.shape[0]

    delta = nn.Parameter(torch.zeros(L, 3, device=device))
    optimizer = torch.optim.Adam([delta], lr=lr)

    max_delta = k_sigma

    loss_history = []
    target_history = []
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

    elapsed = time.time() - start_time

    with torch.no_grad():
        s_cf_final = s_obs_norm + delta
        y_cf_final = model(h_obs.unsqueeze(0), s_cf_final.unsqueeze(0)).squeeze(0)

    converged = target_history[-1] < 1e-4

    return {
        "s_cf_norm": s_cf_final.detach().cpu(),
        "delta": delta.detach().cpu(),
        "y_cf": y_cf_final.detach().cpu(),
        "loss_history": loss_history,
        "target_history": target_history,
        "converged": converged,
        "n_iter": n_iter,
        "n_params": L * 3,
        "wall_clock_s": elapsed,
        "method": "comet_hydro",
    }
