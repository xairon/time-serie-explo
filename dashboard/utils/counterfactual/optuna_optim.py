"""PhysCF black-box counterfactual optimization via Optuna."""

from __future__ import annotations

import time

import torch

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .perturbation import PerturbationLayer


def generate_counterfactual_optuna(
    h_obs: torch.Tensor,
    s_obs_phys: torch.Tensor,
    model: torch.nn.Module,
    target_bounds: tuple[torch.Tensor, torch.Tensor],
    scaler: dict,
    months: torch.Tensor,
    lambda_prox: float = 0.1,
    n_trials: int = 200,
    cc_rate: float = 0.07,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Generate counterfactual using Optuna (TPE sampler).

    Same parameter space as gradient-based PhysCF but optimized via black-box.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna required. Install with: pip install optuna")

    h_obs = h_obs.to(device)
    s_obs_phys = s_obs_phys.to(device)
    months = months.to(device)
    lower, upper = target_bounds[0].to(device), target_bounds[1].to(device)

    model = model.to(device)
    model.eval()

    loss_history = []
    start_time = time.time()

    def objective(trial):
        params = {
            "s_P_DJF": trial.suggest_float("s_P_DJF", 0.3, 2.0),
            "s_P_MAM": trial.suggest_float("s_P_MAM", 0.3, 2.0),
            "s_P_JJA": trial.suggest_float("s_P_JJA", 0.3, 2.0),
            "s_P_SON": trial.suggest_float("s_P_SON", 0.3, 2.0),
            "delta_T": trial.suggest_float("delta_T", -5.0, 5.0),
            "delta_etp": trial.suggest_float("delta_etp", -0.03, 0.03),
            "delta_s": trial.suggest_float("delta_s", -30.0, 30.0),
        }

        perturbation = PerturbationLayer(cc_rate=cc_rate).to(device)
        perturbation.from_interpretable(params)

        with torch.no_grad():
            s_cf = perturbation(s_obs_phys, months)
            s_cf_norm = s_cf.clone()
            stress_cols = ["precip", "temp", "evap"]
            for j, col in enumerate(stress_cols):
                if col in scaler:
                    mu = scaler[col]["mean"]
                    sigma = scaler[col]["std"]
                    if sigma > 0:
                        s_cf_norm[..., j] = (s_cf_norm[..., j] - mu) / sigma

            y_cf = model(h_obs.unsqueeze(0), s_cf_norm.unsqueeze(0)).squeeze(0)
            l_target = torch.mean(
                torch.relu(lower - y_cf) + torch.relu(y_cf - upper)
            ).item()

            l_prox = (
                sum((params[f"s_P_{s}"] - 1.0) ** 2 for s in ["DJF", "MAM", "JJA", "SON"])
                + (params["delta_T"] / 5.0) ** 2
                + (params["delta_etp"] / 0.03) ** 2
                + (params["delta_s"] / 30.0) ** 2
            )

            total = l_target + lambda_prox * l_prox
            loss_history.append(total)
            return total

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    elapsed = time.time() - start_time

    best = study.best_params
    perturbation = PerturbationLayer(cc_rate=cc_rate).to(device)
    perturbation.from_interpretable(best)

    with torch.no_grad():
        s_cf_final = perturbation(s_obs_phys, months)
        s_cf_norm_final = s_cf_final.clone()
        for j, col in enumerate(["precip", "temp", "evap"]):
            if col in scaler:
                mu = scaler[col]["mean"]
                sigma = scaler[col]["std"]
                if sigma > 0:
                    s_cf_norm_final[..., j] = (s_cf_norm_final[..., j] - mu) / sigma
        y_cf_final = model(h_obs.unsqueeze(0), s_cf_norm_final.unsqueeze(0)).squeeze(0)

    l_target_final = torch.mean(
        torch.relu(lower - y_cf_final) + torch.relu(y_cf_final - upper)
    ).item()
    converged = l_target_final < 1e-4

    return {
        "s_cf_phys": s_cf_final.detach().cpu(),
        "s_cf_norm": s_cf_norm_final.detach().cpu(),
        "y_cf": y_cf_final.detach().cpu(),
        "theta_star": perturbation.to_interpretable(),
        "loss_history": loss_history,
        "converged": converged,
        "n_trials": n_trials,
        "best_loss": study.best_value,
        "wall_clock_s": elapsed,
        "method": "physcf_optuna",
    }
