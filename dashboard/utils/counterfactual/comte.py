"""CoMTE — Counterfactual Explanations for Multivariate Time Series.

Faithful adaptation of Ates et al. (2021) for regression/IPS-band targeting.
Original paper: "Counterfactual Explanations for Multivariate Time Series" (AIME 2021)
Reference implementation: https://github.com/peaclab/CoMTE

Key adaptation for regression:
- Original CoMTE targets classification (maximize predict_proba[target_class])
- We target IPS bands: maximize the fraction of output timesteps within target bounds
- Distractors are training windows classified by dominant IPS class of GT values
- With 3 stress features (precip, temp, evap), exhaustive search over 2^3=8 combos

Algorithm:
1. Build distractor pool: group training windows by IPS class, index with KD-Tree
2. For each target IPS class, find k nearest distractors
3. Exhaustive search over 2^3 feature masks: for each mask, swap selected features
   from distractor into observed stresses, run model, compute in-band fraction
4. Select mask with highest in-band fraction (ties broken by fewer swapped features)
5. Greedy pruning: try removing each swapped feature, keep only those that help
"""

from __future__ import annotations

import logging
import time
from itertools import product
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .perturbation import PerturbationLayer

logger = logging.getLogger(__name__)


def _classify_window_ips(
    values: np.ndarray,
    dates: np.ndarray,
    ref_stats: dict[int, tuple[float, float]],
) -> str:
    """Classify a prediction window by its dominant IPS class.

    Computes monthly z-scores, classifies each timestep, returns the most
    frequent class (mode). Used to assign training windows to IPS classes
    for distractor pool construction.
    """
    from .ips import IPS_CLASSES, IPS_ORDER

    if len(values) == 0:
        return "normal"

    # Compute z-score per timestep using monthly ref_stats
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1 if hasattr(dates, "astype") else np.full(len(values), 6)
    try:
        import pandas as pd
        if isinstance(dates, pd.DatetimeIndex):
            months = dates.month.values
        elif hasattr(dates, "month"):
            months = np.array([d.month for d in dates])
    except Exception:
        pass

    class_counts: dict[str, int] = {c: 0 for c in IPS_ORDER}
    for i, (val, m) in enumerate(zip(values, months)):
        if np.isnan(val):
            continue
        mu_m, sigma_m = ref_stats.get(int(m), (0.0, 1.0))
        if sigma_m <= 0:
            sigma_m = 1.0
        z = (val - mu_m) / sigma_m
        for cls_name in IPS_ORDER:
            z_min, z_max = IPS_CLASSES[cls_name]
            if z_min <= z < z_max:
                class_counts[cls_name] += 1
                break

    # Return most frequent class
    if sum(class_counts.values()) == 0:
        return "normal"
    return max(class_counts, key=lambda c: class_counts[c])


def _build_distractor_pools(
    df_full: "pd.DataFrame",
    target_col: str,
    covariate_cols: list[str],
    L: int,
    H: int,
    mu_target: float,
    sigma_target: float,
    ref_stats: dict[int, tuple[float, float]],
) -> dict[str, list[dict]]:
    """Build distractor pools from training/validation data.

    Groups sliding windows by their dominant IPS class. Each distractor stores:
    - s_norm: normalized stresses (L, C) for feature swapping
    - s_phys: physical stresses (L, C) for interpretability
    - h_norm: normalized GWL lookback (L,)
    - gt_values: raw GWL values in the horizon window
    - ips_class: dominant IPS class label
    """
    import pandas as pd

    pools: dict[str, list[dict]] = {}
    n = len(df_full)
    stride = max(1, H // 2)  # 50% overlap for more distractors

    for start in range(0, n - L - H + 1, stride):
        lookback_slice = df_full.iloc[start : start + L]
        horizon_slice = df_full.iloc[start + L : start + L + H]

        if len(lookback_slice) < L or len(horizon_slice) < H:
            continue

        # Extract normalized stresses
        s_norm = lookback_slice[covariate_cols].values.astype(np.float32)
        # Extract normalized GWL
        h_norm = lookback_slice[target_col].values.astype(np.float32)

        # Raw GWL values in horizon (for IPS classification)
        gt_norm = horizon_slice[target_col].values.astype(np.float32)
        gt_raw = gt_norm * sigma_target + mu_target

        # Classify this window
        ips_class = _classify_window_ips(gt_raw, horizon_slice.index, ref_stats)

        entry = {
            "s_norm": s_norm,
            "h_norm": h_norm,
            "ips_class": ips_class,
            "s_flat": s_norm.flatten(),  # I3 fix: flattened (L*C,) for KD-Tree, preserves temporal structure
        }

        if ips_class not in pools:
            pools[ips_class] = []
        pools[ips_class].append(entry)

    return pools


def _find_nearest_distractors(
    s_obs_flat: np.ndarray,
    pool: list[dict],
    k: int,
) -> list[dict]:
    """Find k nearest distractors by flattened stress distance (KD-Tree or brute force).

    I3 fix: Following the original CoMTE, uses flattened (L*C,) stress vectors
    for KD-Tree distance. This preserves temporal structure unlike mean-based distance.
    Falls back to brute force for small pools.
    """
    if len(pool) == 0:
        return []
    if len(pool) <= k:
        return pool

    try:
        from scipy.spatial import KDTree

        flats = np.array([d["s_flat"] for d in pool])
        tree = KDTree(flats)
        _, indices = tree.query(s_obs_flat, k=k)
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        return [pool[i] for i in indices]
    except ImportError:
        # Brute force fallback
        dists = [np.linalg.norm(d["s_flat"] - s_obs_flat) for d in pool]
        indices = np.argsort(dists)[:k]
        return [pool[i] for i in indices]


def generate_counterfactual_comte(
    h_obs: torch.Tensor,
    s_obs_norm: torch.Tensor,
    model: nn.Module,
    target_bounds: tuple[torch.Tensor, torch.Tensor],
    df_full: "pd.DataFrame",
    target_col: str,
    covariate_cols: list[str],
    mu_target: float,
    sigma_target: float,
    ref_stats: dict[int, tuple[float, float]],
    L: int,
    H: int,
    scaler: dict,
    target_ips_class: str = "normal",
    num_distractors: int = 5,
    tau: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Generate counterfactual using CoMTE (Ates et al. 2021).

    Adapted for regression: swaps stress features from distractor windows
    belonging to the target IPS class. With 3 features, exhaustive search
    over 2^3=8 feature masks replaces the original hill climbing.

    Args:
        h_obs: Normalized GWL lookback tensor (L,).
        s_obs_norm: Normalized stresses tensor (L, C).
        model: Forecasting model with forward(h_obs, s_obs) -> y_hat.
        target_bounds: Tuple of (lower, upper) target IPS bounds (normalized).
        df_full: Full DataFrame with all data (for distractor pool).
        target_col: Name of the target column.
        covariate_cols: Names of covariate columns.
        mu_target: Target mean (for denormalization).
        sigma_target: Target std (for denormalization).
        ref_stats: IPS monthly reference statistics {month: (mu, sigma)}.
        L: Input chunk length.
        H: Output chunk length.
        scaler: Dict of covariate scaler params {col: {mean, std}}.
        target_ips_class: Target IPS class for distractor selection.
        num_distractors: Number of nearest distractors to consider (k).
        tau: In-band fraction threshold for success.
        device: Torch device string.

    Returns:
        CounterfactualResult dict with CoMTE-specific keys.
    """
    stress_cols = PerturbationLayer.STRESS_COLUMNS
    C = s_obs_norm.shape[1]  # number of stress features

    h_obs = h_obs.to(device)
    s_obs_norm_t = s_obs_norm.to(device)
    lower, upper = target_bounds[0].to(device), target_bounds[1].to(device)

    model = model.to(device)
    # C1 fix: Save state of the underlying PyTorch module, not the adapter wrapper
    _inner = model._pytorch_module if hasattr(model, "_pytorch_module") else model
    _original_training = _inner.training
    _original_requires_grad = {n: p.requires_grad for n, p in _inner.named_parameters()}

    # Freeze model weights — use to_train_mode() which sets train + freezes weights
    if hasattr(model, "to_train_mode"):
        model.to_train_mode()
    else:
        _inner.train()
        for p in _inner.parameters():
            p.requires_grad_(False)

    start_time = time.time()

    # Step 1: Build distractor pools from training data
    pools = _build_distractor_pools(
        df_full, target_col, covariate_cols,
        L, H, mu_target, sigma_target, ref_stats,
    )

    # Step 2: Find distractors from target class
    s_obs_np = s_obs_norm.numpy() if isinstance(s_obs_norm, torch.Tensor) else s_obs_norm
    s_obs_flat = s_obs_np.flatten()  # I3 fix: flattened (L*C,) for KD-Tree

    actual_distractor_class = target_ips_class
    target_pool = pools.get(target_ips_class, [])
    if len(target_pool) == 0:
        # G4 fix: Fallback to adjacent classes, track actual class used
        from .ips import IPS_ORDER
        target_idx = IPS_ORDER.index(target_ips_class) if target_ips_class in IPS_ORDER else 3
        for offset in [1, -1, 2, -2, 3, -3]:
            adj_idx = target_idx + offset
            if 0 <= adj_idx < len(IPS_ORDER):
                adj_class = IPS_ORDER[adj_idx]
                if adj_class in pools and len(pools[adj_class]) > 0:
                    target_pool = pools[adj_class]
                    actual_distractor_class = adj_class
                    break

    if len(target_pool) == 0:
        # No distractors available at all — return factual as-is
        with torch.no_grad():
            y_obs = model(h_obs.unsqueeze(0), s_obs_norm_t.unsqueeze(0)).squeeze(0)
        # C1 fix: Restore underlying module state
        _inner.train(_original_training)
        for n, p in _inner.named_parameters():
            p.requires_grad_(_original_requires_grad.get(n, True))
        return {
            "method": "comte",
            "y_cf": y_obs.detach().cpu(),
            "s_cf_phys": None,
            "s_cf_norm": s_obs_norm.detach().cpu() if isinstance(s_obs_norm, torch.Tensor) else torch.tensor(s_obs_norm),
            "theta_star": None,
            "loss_history": [],
            "target_history": [],
            "prox_history": [],
            "smooth_history": [],
            "converged": False,
            "wall_clock_s": time.time() - start_time,
            "n_params": 0,
            "n_iter": 0,
            "n_trials": None,
            "best_loss": None,
            "comte_info": {
                "swapped_features": [],
                "distractor_class": actual_distractor_class,
                "n_distractors_available": 0,
                "in_band_fraction": 0.0,
                "explanation": "Aucun distracteur disponible pour cette classe IPS",
            },
        }

    distractors = _find_nearest_distractors(s_obs_flat, target_pool, num_distractors)

    # Step 3: Exhaustive search over feature masks × distractors
    # With C features, there are 2^C - 1 non-empty masks (exclude all-zeros)
    feature_indices = list(range(C))
    all_masks = list(product([0, 1], repeat=C))
    # Remove the all-zeros mask (no swap = factual)
    all_masks = [m for m in all_masks if sum(m) > 0]

    best_score = -1.0
    best_mask: tuple = (0,) * C
    best_distractor_idx = 0
    best_y_cf = None
    best_s_cf = None
    all_scores: list[dict] = []

    with torch.no_grad():
        for d_idx, distractor in enumerate(distractors):
            d_s_norm = torch.tensor(distractor["s_norm"], dtype=torch.float32, device=device)

            for mask in all_masks:
                # Build counterfactual stresses: swap selected features from distractor
                s_cf = s_obs_norm_t.clone()
                for f_idx in range(C):
                    if mask[f_idx] == 1:
                        s_cf[:, f_idx] = d_s_norm[:, f_idx]

                # Run model
                y_cf = model(h_obs.unsqueeze(0), s_cf.unsqueeze(0)).squeeze(0)

                # Compute in-band fraction (adapted from CoMTE's predict_proba)
                in_band = ((y_cf >= lower) & (y_cf <= upper)).float()
                score = in_band.mean().item()

                n_swapped = sum(mask)
                all_scores.append({
                    "mask": mask,
                    "distractor_idx": d_idx,
                    "in_band_fraction": score,
                    "n_swapped": n_swapped,
                })

                # Select best: highest in-band fraction, ties broken by fewer features
                if score > best_score or (score == best_score and n_swapped < sum(best_mask)):
                    best_score = score
                    best_mask = mask
                    best_distractor_idx = d_idx
                    best_y_cf = y_cf.clone()
                    best_s_cf = s_cf.clone()

    # Step 4: Greedy pruning (CoMTE paper section 3.3)
    # Try removing each swapped feature one at a time
    if best_s_cf is not None and sum(best_mask) > 1:
        pruned_mask = list(best_mask)
        improved = True
        while improved:
            improved = False
            for f_idx in range(C):
                if pruned_mask[f_idx] == 0:
                    continue
                # Try removing this feature
                test_mask = pruned_mask.copy()
                test_mask[f_idx] = 0
                if sum(test_mask) == 0:
                    continue  # don't remove all features

                s_test = s_obs_norm_t.clone()
                d_s = torch.tensor(distractors[best_distractor_idx]["s_norm"],
                                   dtype=torch.float32, device=device)
                for fi in range(C):
                    if test_mask[fi] == 1:
                        s_test[:, fi] = d_s[:, fi]

                with torch.no_grad():
                    y_test = model(h_obs.unsqueeze(0), s_test.unsqueeze(0)).squeeze(0)
                    in_band = ((y_test >= lower) & (y_test <= upper)).float()
                    test_score = in_band.mean().item()

                # Keep removal if score doesn't degrade
                if test_score >= best_score - 1e-6:
                    pruned_mask = test_mask
                    best_score = test_score
                    best_y_cf = y_test.clone()
                    best_s_cf = s_test.clone()
                    improved = True
                    break  # restart pruning loop

        best_mask = tuple(pruned_mask)

    elapsed = time.time() - start_time

    # C1 fix: Restore underlying module state
    _inner.train(_original_training)
    for n, p in _inner.named_parameters():
        p.requires_grad_(_original_requires_grad.get(n, True))

    # Build feature names from mask
    feature_names = covariate_cols[:C] if len(covariate_cols) >= C else stress_cols[:C]
    swapped_features = [feature_names[i] for i in range(C) if best_mask[i] == 1]

    # Denormalize s_cf to physical units
    # G9 fix: Try both canonical names and actual covariate column names, log failures
    s_cf_phys = None
    if best_s_cf is not None:
        try:
            s_cf_phys_t = best_s_cf.clone().cpu()
            for j in range(C):
                # Try canonical name first, then actual covariate name
                canonical = stress_cols[j] if j < len(stress_cols) else None
                actual = covariate_cols[j] if j < len(covariate_cols) else None
                scaler_entry = None
                if canonical and canonical in scaler:
                    scaler_entry = scaler[canonical]
                elif actual and actual in scaler:
                    scaler_entry = scaler[actual]

                if scaler_entry:
                    mu_c = scaler_entry["mean"]
                    sigma_c = scaler_entry["std"]
                    if sigma_c > 0:
                        s_cf_phys_t[:, j] = s_cf_phys_t[:, j] * sigma_c + mu_c
                else:
                    logger.warning("CoMTE: no scaler found for feature %d (canonical=%s, actual=%s)", j, canonical, actual)
            s_cf_phys = s_cf_phys_t
        except Exception as exc:
            logger.warning("CoMTE: denormalization failed: %s", exc)

    # Build theta dict: for CoMTE, theta represents which features were swapped
    theta_dict: dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        theta_dict[fname] = float(best_mask[i])

    converged = best_score >= tau

    return {
        "method": "comte",
        "y_cf": best_y_cf.detach().cpu() if best_y_cf is not None else torch.zeros(H),
        "s_cf_phys": s_cf_phys,
        "s_cf_norm": best_s_cf.detach().cpu() if best_s_cf is not None else s_obs_norm.detach().cpu(),
        "theta_star": theta_dict,
        "loss_history": [1.0 - s["in_band_fraction"] for s in all_scores],
        "target_history": [1.0 - s["in_band_fraction"] for s in all_scores],
        "prox_history": [],
        "smooth_history": [],
        "converged": converged,
        "wall_clock_s": elapsed,
        "n_params": sum(best_mask),
        "n_iter": len(all_scores),
        "n_trials": len(distractors),
        "best_loss": 1.0 - best_score,
        "comte_info": {
            "swapped_features": swapped_features,
            "distractor_class": actual_distractor_class,
            "n_distractors_available": len(target_pool),
            "n_distractors_used": len(distractors),
            "in_band_fraction": best_score,
            "best_mask": list(best_mask),
            "n_candidates_evaluated": len(all_scores),
            "tau": tau,
            "explanation": _build_explanation(swapped_features, best_score, tau),
        },
    }


def _build_explanation(swapped_features: list[str], score: float, tau: float) -> str:
    """Build a human-readable French explanation of the CoMTE result."""
    if len(swapped_features) == 0:
        return "Aucune combinaison de features ne permet d'atteindre la classe cible."

    # M3 fix: Map feature names to French using substring matching
    name_map = {
        "precip": "precipitations",
        "rain": "precipitations",
        "temp": "temperature",
        "evap": "evapotranspiration",
        "etp": "evapotranspiration",
    }

    def _to_french(name: str) -> str:
        lower = name.lower()
        for key, fr in name_map.items():
            if key in lower:
                return fr
        return name

    fr_names = [_to_french(f) for f in swapped_features]

    if len(fr_names) == 1:
        feat_str = fr_names[0]
    elif len(fr_names) == 2:
        feat_str = f"{fr_names[0]} et {fr_names[1]}"
    else:
        feat_str = ", ".join(fr_names[:-1]) + f" et {fr_names[-1]}"

    pct = round(score * 100)
    if score >= tau:
        return (
            f"En modifiant {feat_str} selon un scenario de reference, "
            f"le modele predit la classe cible sur {pct}% des pas de temps."
        )
    else:
        return (
            f"La meilleure combinaison ({feat_str}) n'atteint que {pct}% "
            f"de conformite (seuil: {round(tau * 100)}%). "
            f"La classe cible est difficile a atteindre par simple substitution de features."
        )
