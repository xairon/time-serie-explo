"""PerturbationLayer: physics-informed differentiable perturbation of climate stresses.

Central component of PhysCF. Transforms raw learnable parameters θ₁-θ₄ into
physically constrained perturbations of precipitation, temperature, and ETP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Season mapping: month → season index {DJF=0, MAM=1, JJA=2, SON=3}
MONTH_TO_SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def month_to_season_tensor(months: torch.Tensor) -> torch.Tensor:
    """Convert month tensor (1-12) to season index tensor (0-3)."""
    mapping = torch.zeros(13, dtype=torch.long, device=months.device)
    for m, s in MONTH_TO_SEASON.items():
        mapping[m] = s
    return mapping[months]


class PerturbationLayer(nn.Module):
    """Differentiable perturbation layer for climate stresses.

    Learnable parameters (raw space, transformed via sigmoid/tanh):
        theta1_raw: (4,) - one per season {DJF, MAM, JJA, SON} → precipitation scaling
        theta2_raw: (1,) - global temperature offset
        theta3_raw: (1,) - ETP residual
        theta4_raw: (1,) - temporal shift

    Input:  s_obs (L, 3) or (B, L, 3) - [precip, temp, evap] in physical units
    Output: s_cf  (L, 3) or (B, L, 3) - perturbed stresses

    Transformations:
        θ₁ → s_P[k] = 0.3 + 1.7 × σ(θ₁_raw[k])       ∈ [0.3, 2.0]
        θ₂ → ΔT     = 5.0 × tanh(θ₂_raw)               ∈ [-5, +5] °C
        θ₃ → δ      = 0.03 × tanh(θ₃_raw)              ∈ [-0.03, +0.03]
        θ₄ → Δs     = 30.0 × tanh(θ₄_raw)              ∈ [-30, +30] days
    """

    # ---- Single source of truth for parameter ranges ----
    PARAM_RANGES = {
        "s_P_DJF": {"min": 0.3, "max": 2.0, "identity": 1.0, "unit": "multiplicateur"},
        "s_P_MAM": {"min": 0.3, "max": 2.0, "identity": 1.0, "unit": "multiplicateur"},
        "s_P_JJA": {"min": 0.3, "max": 2.0, "identity": 1.0, "unit": "multiplicateur"},
        "s_P_SON": {"min": 0.3, "max": 2.0, "identity": 1.0, "unit": "multiplicateur"},
        "delta_T":  {"min": -5.0, "max": 5.0, "identity": 0.0, "unit": "degC"},
        "delta_etp": {"min": -0.03, "max": 0.03, "identity": 0.0, "unit": "fractionnaire"},
        "delta_s":  {"min": -30.0, "max": 30.0, "identity": 0.0, "unit": "jours"},
    }
    STRESS_COLUMNS = ["precip", "temp", "evap"]
    CONVERGENCE_THRESHOLD = 1e-4

    def __init__(self, cc_rate: float = 0.07) -> None:
        """Initialize the perturbation layer.

        Args:
            cc_rate: Clausius-Clapeyron coupling rate (default 0.07 per degC).
        """
        super().__init__()
        self.cc_rate = cc_rate

        # Raw learnable parameters (initialized to 0 → identity perturbation)
        self.theta1_raw = nn.Parameter(torch.zeros(4))   # seasonal precip scaling
        self.theta2_raw = nn.Parameter(torch.zeros(1))   # temperature offset
        self.theta3_raw = nn.Parameter(torch.zeros(1))   # ETP residual
        self.theta4_raw = nn.Parameter(torch.zeros(1))   # temporal shift

    def identity_init(self) -> None:
        """Reset parameters to identity (no perturbation).

        Sets theta1_raw such that s_P = 1.0 exactly:
        1.0 = 0.3 + 1.7 * sigmoid(x) → sigmoid(x) = 0.7/1.7 → x = logit(0.7/1.7)
        """
        with torch.no_grad():
            # s_P = 1.0: sigmoid(x) = (1.0 - 0.3) / 1.7
            p = torch.tensor((1.0 - 0.3) / 1.7)
            self.theta1_raw.fill_(torch.log(p / (1.0 - p)).item())
            self.theta2_raw.zero_()
            self.theta3_raw.zero_()
            self.theta4_raw.zero_()

    @property
    def s_P(self) -> torch.Tensor:
        """Seasonal precipitation scaling factors ∈ [0.3, 2.0]."""
        return 0.3 + 1.7 * torch.sigmoid(self.theta1_raw)

    @property
    def delta_T(self) -> torch.Tensor:
        """Temperature offset ∈ [-5, +5] °C."""
        return 5.0 * torch.tanh(self.theta2_raw)

    @property
    def delta_etp(self) -> torch.Tensor:
        """ETP residual ∈ [-0.03, +0.03]."""
        return 0.03 * torch.tanh(self.theta3_raw)

    @property
    def delta_s(self) -> torch.Tensor:
        """Temporal shift ∈ [-30, +30] days."""
        return 30.0 * torch.tanh(self.theta4_raw)

    def _differentiable_shift(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply differentiable temporal shift via linear interpolation.

        Args:
            x: (L,) or (B, L) time series
            shift: scalar shift in days (fractional)

        Returns:
            Shifted series with same shape.
        """
        L = x.shape[-1]
        t = torch.arange(L, dtype=x.dtype, device=x.device)
        t_shifted = t - shift  # shift > 0 means delayed

        # Clamp to valid range
        t_shifted = t_shifted.clamp(0, L - 1)

        # Linear interpolation indices
        t_floor = t_shifted.floor().long().clamp(0, L - 1)
        t_ceil = (t_floor + 1).clamp(0, L - 1)
        w = t_shifted - t_floor.float()

        if x.dim() == 1:
            return x[t_floor] * (1 - w) + x[t_ceil] * w
        else:
            # Batched: x is (B, L)
            return x[:, t_floor] * (1 - w).unsqueeze(0) + x[:, t_ceil] * w.unsqueeze(0)

    def forward(self, s_obs: torch.Tensor, months: torch.Tensor) -> torch.Tensor:
        """Apply physics-informed perturbation.

        Order of application:
        1. θ₄ — Joint temporal shift of P and T (differentiable interpolation)
        2. θ₁ — Scale shifted P by seasonal factor
        3. θ₂ — Offset shifted T
        4. θ₃ — CC coupling: ETP_cf = ETP_obs × (1 + cc_rate × ΔT) × (1 + δ)

        Args:
            s_obs: (L, 3) or (B, L, 3) - [precip, temp, evap] in physical units
            months: (L,) or (B, L) - month indices (1-12)

        Returns:
            s_cf: same shape as s_obs - perturbed stresses
        """
        batched = s_obs.dim() == 3
        if not batched:
            s_obs = s_obs.unsqueeze(0)
            months = months.unsqueeze(0)

        B, L, _ = s_obs.shape
        precip = s_obs[..., 0]  # (B, L)
        temp = s_obs[..., 1]    # (B, L)
        evap = s_obs[..., 2]    # (B, L)

        # Get transformed parameters
        shift = self.delta_s       # scalar
        s_p = self.s_P             # (4,)
        delta_t = self.delta_T     # (1,)
        delta = self.delta_etp     # (1,)

        # Step 1: Temporal shift of P and T
        p_shifted = self._differentiable_shift(precip, shift)
        t_shifted = self._differentiable_shift(temp, shift)

        # Step 2: Seasonal precipitation scaling
        seasons = month_to_season_tensor(months)  # (B, L)
        season_factors = s_p[seasons]              # (B, L)
        p_cf = p_shifted * season_factors
        p_cf = p_cf.clamp(min=0.0)  # precip >= 0

        # Step 3: Temperature offset
        t_cf = t_shifted + delta_t

        # Step 4: Clausius-Clapeyron coupling for ETP
        # ETP is NOT shifted by θ₄
        evap_cf = evap * (1 + self.cc_rate * delta_t) * (1 + delta)
        evap_cf = evap_cf.clamp(min=0.0)

        # Stack back
        s_cf = torch.stack([p_cf, t_cf, evap_cf], dim=-1)

        if not batched:
            s_cf = s_cf.squeeze(0)

        return s_cf

    def to_interpretable(self) -> dict[str, float]:
        """Convert raw parameters to physical/interpretable values.

        Returns:
            Dict mapping parameter names to their physical values.
        """
        return {
            "s_P_DJF": self.s_P[0].item(),
            "s_P_MAM": self.s_P[1].item(),
            "s_P_JJA": self.s_P[2].item(),
            "s_P_SON": self.s_P[3].item(),
            "delta_T": self.delta_T.item(),
            "delta_etp": self.delta_etp.item(),
            "delta_s": self.delta_s.item(),
        }

    def from_interpretable(self, params: dict[str, float]) -> None:
        """Set raw parameters from interpretable values (for Optuna).

        Args:
            params: Dict mapping parameter names to their physical values.
        """
        with torch.no_grad():
            # Inverse of: s_P = 0.3 + 1.7 * sigmoid(raw) → raw = logit((s_P - 0.3) / 1.7)
            for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
                key = f"s_P_{season}"
                if key in params:
                    val = (params[key] - 0.3) / 1.7
                    val = max(1e-6, min(1 - 1e-6, val))  # clamp for logit stability
                    self.theta1_raw[i] = torch.log(torch.tensor(val / (1 - val)))

            if "delta_T" in params:
                val = params["delta_T"] / 5.0
                val = max(-0.9999, min(0.9999, val))
                self.theta2_raw[0] = torch.atanh(torch.tensor(val))

            if "delta_etp" in params:
                val = params["delta_etp"] / 0.03
                val = max(-0.9999, min(0.9999, val))
                self.theta3_raw[0] = torch.atanh(torch.tensor(val))

            if "delta_s" in params:
                val = params["delta_s"] / 30.0
                val = max(-0.9999, min(0.9999, val))
                self.theta4_raw[0] = torch.atanh(torch.tensor(val))
