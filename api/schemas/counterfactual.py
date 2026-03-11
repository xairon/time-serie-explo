"""Schemas for the counterfactual API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CFGenerateRequest(BaseModel):
    """Request for counterfactual generation (PhysCF, Optuna, or CoMTE)."""

    model_id: str
    method: str = "physcf"  # physcf, optuna, or comte
    target_ips_class: str = "normal"  # IPS class target (applies to all months)
    target_ips_classes: dict[str, str] = Field(
        default_factory=dict,
        description='Per-month IPS class overrides. Key = month number "1"-"12", value = IPS class name. '
        "Takes priority over target_ips_class. Empty dict = use target_ips_class for all months.",
    )
    lambda_prox: float = 0.1
    n_iter: int = 500
    lr: float = 0.02
    cc_rate: float = 0.07
    device: str = "cpu"
    # Optuna-specific
    n_trials: int = 200
    seed: int = 42
    # CoMTE-specific (Ates et al. 2021)
    num_distractors: int = 5  # k nearest neighbors from target class pool
    tau: float = 0.5  # in-band fraction threshold for success
    # Legacy COMET params (kept for backwards compat, unused by CoMTE)
    k_sigma: float = 4.0
    lambda_smooth: float = 0.1
    # Position in test set (None = auto middle)
    start_idx: Optional[int] = None
    # Perturbation modifiers (for frontend compatibility)
    modifications: dict[str, float] = Field(default_factory=dict)


class CFResult(BaseModel):
    """Result of a counterfactual generation."""

    task_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class IPSReferenceRequest(BaseModel):
    """Request for IPS reference computation."""

    model_id: str
    window: int = 1  # IPS-N window (1, 3, 6, 12)
    aquifer_type: Optional[str] = None


class PastasValidateRequest(BaseModel):
    """Request for Pastas dual validation."""

    model_id: str
    cf_task_id: str
    gamma: float = 1.5  # Acceptance threshold multiplier
