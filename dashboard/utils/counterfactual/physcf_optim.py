"""PhysCF gradient-based counterfactual optimization.

Works with any model that implements the adapter interface:
    model(h_obs, s_obs) -> y_hat  where y_hat is (batch, H)

This includes both DartsModelAdapter (TFT, GRU, etc.) and
PhysCF's standalone GRUForecaster.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .perturbation import PerturbationLayer


def proximity_loss(perturbation: PerturbationLayer) -> torch.Tensor:
    """Compute proximity loss (Eq. 7 from paper).

    L_prox = sum(s_P[k] - 1)^2 + (delta_T/5)^2 + (delta/0.03)^2 + (delta_s/30)^2
    """
    s_p = perturbation.s_P
    l_sp = ((s_p - 1.0) ** 2).sum()
    l_dt = (perturbation.delta_T / 5.0) ** 2
    l_de = (perturbation.delta_etp / 0.03) ** 2
    l_ds = (perturbation.delta_s / 30.0) ** 2
    return l_sp + l_dt.squeeze() + l_de.squeeze() + l_ds.squeeze()


def target_loss(
    y_cf: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Target loss: penalize predictions outside IPS bounds.

    L_target = mean(relu(lower - y_cf) + relu(y_cf - upper))
    """
    return torch.mean(torch.relu(lower - y_cf) + torch.relu(y_cf - upper))


def generate_counterfactual(
    h_obs: torch.Tensor,
    s_obs_phys: torch.Tensor,
    model: nn.Module,
    target_bounds: tuple[torch.Tensor, torch.Tensor],
    scaler: dict,
    months: torch.Tensor,
    lambda_prox: float = 0.1,
    n_iter: int = 500,
    lr: float = 0.02,
    cc_rate: float = 0.07,
    device: str = "cpu",
) -> dict:
    """Generate a physics-informed counterfactual explanation.

    Pipeline per iteration:
    1. s_cf_phys = perturbation_layer(s_obs_phys, months)
    2. s_cf_norm = normalize(s_cf_phys, scaler)
    3. y_cf = model(h_obs, s_cf_norm) -- h_obs unchanged
    4. L_target = penalty for y_cf outside bounds
    5. L_prox = Eq. 7 (closeness to identity)
    6. L = L_target + lambda * L_prox

    Args:
        h_obs: (L,) normalized gwl lookback
        s_obs_phys: (L, 3) stresses in physical units
        model: Model with forward(h_obs, s_obs) -> y_hat interface.
            Can be DartsModelAdapter, StandaloneGRUAdapter, or any nn.Module.
        target_bounds: (lower, upper) tensors (H,) in normalized units
        scaler: Scaler dict for stress normalization
        months: (L,) month indices
        lambda_prox: Proximity weight (default 0.1)
        n_iter: Number of optimization iterations
        lr: Learning rate
        cc_rate: Clausius-Clapeyron rate
        device: Computation device

    Returns:
        Dict with s_cf_phys, theta_star, loss_history, y_cf, converged, etc.
    """
    h_obs = h_obs.to(device)
    s_obs_phys = s_obs_phys.to(device)
    months = months.to(device)
    lower, upper = target_bounds[0].to(device), target_bounds[1].to(device)

    # Freeze model weights but keep in train mode for cuDNN RNN backward
    model = model.to(device)
    if hasattr(model, "to_train_mode"):
        model.to_train_mode()
    else:
        model.train()
        for p in model.parameters():
            p.requires_grad_(False)

    # Initialize perturbation layer
    perturbation = PerturbationLayer(cc_rate=cc_rate).to(device)
    perturbation.identity_init()

    optimizer = torch.optim.Adam(perturbation.parameters(), lr=lr)

    loss_history = []
    target_history = []
    prox_history = []
    start_time = time.time()

    for i in range(n_iter):
        optimizer.zero_grad()

        # 1. Perturb stresses in physical space
        s_cf_phys = perturbation(s_obs_phys, months)

        # 2. Normalize for model input
        stress_cols = ["precip", "temp", "evap"]
        s_cf_norm = s_cf_phys.clone()
        for j, col in enumerate(stress_cols):
            if col in scaler:
                mu = scaler[col]["mean"]
                sigma = scaler[col]["std"]
                if sigma > 0:
                    s_cf_norm[..., j] = (s_cf_norm[..., j] - mu) / sigma

        # 3. Forward through model
        y_cf = model(h_obs.unsqueeze(0), s_cf_norm.unsqueeze(0)).squeeze(0)

        # 4 & 5. Compute losses
        l_target = target_loss(y_cf, lower, upper)
        l_prox = proximity_loss(perturbation)
        loss = l_target + lambda_prox * l_prox

        # 6. Backward
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        target_history.append(l_target.item())
        prox_history.append(l_prox.item())

    elapsed = time.time() - start_time

    # Final CF
    with torch.no_grad():
        s_cf_final = perturbation(s_obs_phys, months)
        s_cf_norm_final = s_cf_final.clone()
        for j, col in enumerate(stress_cols):
            if col in scaler:
                mu = scaler[col]["mean"]
                sigma = scaler[col]["std"]
                if sigma > 0:
                    s_cf_norm_final[..., j] = (s_cf_norm_final[..., j] - mu) / sigma
        y_cf_final = model(h_obs.unsqueeze(0), s_cf_norm_final.unsqueeze(0)).squeeze(0)

    converged = target_history[-1] < 1e-4

    return {
        "s_cf_phys": s_cf_final.detach().cpu(),
        "s_cf_norm": s_cf_norm_final.detach().cpu(),
        "y_cf": y_cf_final.detach().cpu(),
        "theta_star": perturbation.to_interpretable(),
        "loss_history": loss_history,
        "target_history": target_history,
        "prox_history": prox_history,
        "converged": converged,
        "n_iter": n_iter,
        "wall_clock_s": elapsed,
        "method": "physcf_gradient",
    }
