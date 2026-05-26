"""AI training presets — opinionated configurations for non-expert users.

Each preset bundles a Darts model + hyperparameters tuned for a specific
hydrogeology use case. Frontend exposes these as a dropdown so a BRGM user
can train a sane model in one click without picking input_chunk_length /
batch_size / etc. themselves.

Hyperparameters were chosen as reasonable defaults from the Darts docs +
published baselines on piezometric series; they are not station-tuned (use
auto-tune for that, future feature).
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class PresetSpec(TypedDict):
    id: str
    label: str
    description: str
    target_domain: Literal["piezo", "hydro"]
    model_name: str
    horizon_days: int
    hyperparams: dict[str, Any]
    n_epochs: int
    early_stopping_patience: int


PRESETS: list[PresetSpec] = [
    {
        "id": "piezo_short_7d",
        "label": "Piézo — court terme (7 jours)",
        "description": (
            "Prévision à 7 jours pour gestion opérationnelle. "
            "Modèle DLinear, rapide à entraîner (~1-2 min), bon sur courts horizons."
        ),
        "target_domain": "piezo",
        "model_name": "DLinear",
        "horizon_days": 7,
        "hyperparams": {
            "input_chunk_length": 90,
            "output_chunk_length": 7,
            "batch_size": 64,
            "optimizer_kwargs": {"lr": 1e-3},
        },
        "n_epochs": 30,
        "early_stopping_patience": 5,
    },
    {
        "id": "piezo_mid_30d",
        "label": "Piézo — moyen terme (30 jours)",
        "description": (
            "Prévision à 1 mois, anticipation de sécheresse ou recharge. "
            "Modèle NHiTS, multi-échelle, équilibre vitesse/qualité (~5 min)."
        ),
        "target_domain": "piezo",
        "model_name": "NHiTS",
        "horizon_days": 30,
        "hyperparams": {
            "input_chunk_length": 365,
            "output_chunk_length": 30,
            "batch_size": 32,
            "num_stacks": 3,
            "num_blocks": 1,
            "optimizer_kwargs": {"lr": 1e-3},
        },
        "n_epochs": 50,
        "early_stopping_patience": 8,
    },
    {
        "id": "piezo_long_90d",
        "label": "Piézo — long terme (90 jours)",
        "description": (
            "Prévision trimestrielle, planning ressources. "
            "Modèle TFT avec attention temporelle, plus lent (~15-30 min) mais "
            "capte la saisonnalité annuelle."
        ),
        "target_domain": "piezo",
        "model_name": "TFT",
        "horizon_days": 90,
        "hyperparams": {
            "input_chunk_length": 730,
            "output_chunk_length": 90,
            "batch_size": 16,
            "hidden_size": 32,
            "lstm_layers": 1,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "optimizer_kwargs": {"lr": 5e-4},
            "add_relative_index": True,
        },
        "n_epochs": 100,
        "early_stopping_patience": 12,
    },
    {
        "id": "hydro_flow_14d",
        "label": "Hydro — débit (14 jours)",
        "description": (
            "Prévision de débit à 2 semaines. Capte les pics de crue et l'étiage. "
            "Modèle NHiTS, horizon intermédiaire adapté à la dynamique fluviale (~5 min)."
        ),
        "target_domain": "hydro",
        "model_name": "NHiTS",
        "horizon_days": 14,
        "hyperparams": {
            "input_chunk_length": 180,
            "output_chunk_length": 14,
            "batch_size": 32,
            "num_stacks": 3,
            "num_blocks": 1,
            "optimizer_kwargs": {"lr": 1e-3},
        },
        "n_epochs": 50,
        "early_stopping_patience": 8,
    },
]


def get_preset(preset_id: str) -> PresetSpec | None:
    """Return the preset spec for an id, or None if unknown."""
    return next((p for p in PRESETS if p["id"] == preset_id), None)


def apply_preset_to_request(req_dict: dict[str, Any], preset_id: str) -> dict[str, Any]:
    """Merge preset hyperparams into a training request dict.

    User-supplied fields in the request override preset defaults (preset is
    the floor, not the ceiling). Returns a new dict (does not mutate input).
    """
    preset = get_preset(preset_id)
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_id}")

    merged = dict(req_dict)
    # Pydantic puts explicit `None` in the dict, so setdefault doesn't fill.
    # Use "missing or None" semantics: preset fills if the user didn't override.
    def fill(key: str, value: Any) -> None:
        if merged.get(key) in (None, ""):
            merged[key] = value

    fill("model_name", preset["model_name"])
    fill("n_epochs", preset["n_epochs"])
    fill("early_stopping_patience", preset["early_stopping_patience"])
    fill("early_stopping", True)

    # hyperparams: preset values are defaults; user-supplied keys override.
    user_hp = merged.get("hyperparams") or {}
    merged["hyperparams"] = {**preset["hyperparams"], **user_hp}
    return merged
