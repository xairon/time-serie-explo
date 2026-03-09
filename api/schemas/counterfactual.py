"""Schemas for the counterfactual API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CFGenerateRequest(BaseModel):
    """Request for counterfactual generation (PhysCF, Optuna, or COMET)."""

    model_id: str
    target_ips_class: str = "normal"  # IPS class target
    lambda_prox: float = 0.1
    n_iter: int = 500
    lr: float = 0.02
    cc_rate: float = 0.07
    device: str = "cpu"
    # Optuna-specific
    n_trials: int = 200
    seed: int = 42
    # COMET-specific
    k_sigma: float = 4.0
    lambda_smooth: float = 0.1


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
