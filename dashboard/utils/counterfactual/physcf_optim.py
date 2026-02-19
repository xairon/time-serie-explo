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

    Uses PARAM_RANGES as single source of truth for normalization scales.
    L_prox = sum((param - identity) / scale)^2 for all parameters.

    Args:
        perturbation: PerturbationLayer whose parameters are evaluated.

    Returns:
        Scalar proximity loss tensor.
    """
    _ranges = PerturbationLayer.PARAM_RANGES
    s_p = perturbation.s_P
    # Seasonal precip: scale = max - identity = 2.0 - 1.0 = 1.0
    p_scale = _ranges["s_P_DJF"]["max"] - _ranges["s_P_DJF"]["identity"]
    l_sp = ((s_p - _ranges["s_P_DJF"]["identity"]) / p_scale) ** 2
    l_sp = l_sp.sum()

    dt_scale = _ranges["delta_T"]["max"]  # 5.0
    l_dt = (perturbation.delta_T / dt_scale) ** 2

    de_scale = _ranges["delta_etp"]["max"]  # 0.03
    l_de = (perturbation.delta_etp / de_scale) ** 2

    ds_scale = _ranges["delta_s"]["max"]  # 30.0
    l_ds = (perturbation.delta_s / ds_scale) ** 2

    return l_sp + l_dt.squeeze() + l_de.squeeze() + l_ds.squeeze()


def target_loss(
    y_cf: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Target loss: penalize predictions outside IPS bounds.

    L_target = mean(relu(lower - y_cf) + relu(y_cf - upper))

    Args:
        y_cf: Counterfactual predictions tensor.
        lower: Lower bound of the target IPS band.
        upper: Upper bound of the target IPS band.

    Returns:
        Scalar target loss tensor.
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
        h_obs: Normalized gwl lookback tensor (L,).
        s_obs_phys: Physical stresses tensor (L, 3).
        model: Forecasting model with forward(h_obs, s_obs) -> y_hat.
        target_bounds: Tuple of (lower, upper) target IPS bounds (normalized).
        scaler: Dict of covariate scaler params {col: {mean, std}}.
        months: Month indices tensor (L,), values 1-12.
        lambda_prox: Weight for proximity loss (default 0.1).
        n_iter: Number of gradient descent iterations (default 500).
        lr: Learning rate for Adam optimizer (default 0.02).
        cc_rate: Clausius-Clapeyron rate (default 0.07 per degC).
        device: Torch device string (default "cpu").

    Returns:
        CounterfactualResult dict with standardized keys.
    """
    h_obs = h_obs.to(device)
    s_obs_phys = s_obs_phys.to(device)
    months = months.to(device)
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

    stress_cols = PerturbationLayer.STRESS_COLUMNS

    if n_iter <= 0:
        with torch.no_grad():
            # Normalize s_obs_phys for model input
            s_norm = s_obs_phys.clone()
            for j, col in enumerate(stress_cols):
                if col in scaler:
                    mu_val = scaler[col]["mean"]
                    sigma_val = scaler[col]["std"]
                    if sigma_val > 0:
                        s_norm[..., j] = (s_norm[..., j] - mu_val) / sigma_val
            y_obs = model(h_obs.unsqueeze(0), s_norm.unsqueeze(0)).squeeze(0)
        # Restore model state
        model.train(_original_training)
        for n, p in model.named_parameters():
            p.requires_grad_(_original_requires_grad.get(n, True))
        return {
            "method": "physcf_gradient",
            "y_cf": y_obs.detach().cpu(),
            "s_cf_phys": s_obs_phys.detach().cpu(),
            "s_cf_norm": s_norm.detach().cpu(),
            "theta_star": PerturbationLayer(cc_rate=cc_rate).to_interpretable(),
            "loss_history": [],
            "target_history": [],
            "prox_history": [],
            "converged": False,
            "wall_clock_s": 0.0,
            "n_params": len(PerturbationLayer.PARAM_RANGES),
            "n_iter": 0,
            "n_trials": None,
            "best_loss": None,
        }

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

    # Restore model state
    model.train(_original_training)
    for n, p in model.named_parameters():
        p.requires_grad_(_original_requires_grad.get(n, True))

    converged = target_history[-1] < PerturbationLayer.CONVERGENCE_THRESHOLD

    return {
        "method": "physcf_gradient",
        "y_cf": y_cf_final.detach().cpu(),
        "s_cf_phys": s_cf_final.detach().cpu(),
        "s_cf_norm": s_cf_norm_final.detach().cpu(),
        "theta_star": perturbation.to_interpretable(),
        "loss_history": loss_history,
        "target_history": target_history,
        "prox_history": prox_history,
        "converged": converged,
        "wall_clock_s": elapsed,
        "n_params": len(PerturbationLayer.PARAM_RANGES),
        "n_iter": n_iter,
        "n_trials": None,
        "best_loss": None,
    }
