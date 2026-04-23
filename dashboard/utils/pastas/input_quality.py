"""AI-based anomaly detection on input data quality using Isolation Forest."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def detect_input_anomalies(
    code_bss: str, engine: Engine, contamination: float = 0.05
) -> dict:
    """Detect anomalous months in input data using Isolation Forest."""
    from sklearn.ensemble import IsolationForest

    query = sql_text("""
        SELECT mois, niveau_moyen, precipitation_totale, temperature_moyenne, evaporation_moyenne
        FROM gold.fct_monthly_chroniques
        WHERE code_bss = :code
        ORDER BY mois
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"code": code_bss})

    if df.empty or len(df) < 24:
        return {"months": [], "scores": [], "flagged": [], "n_flagged": 0, "n_total": 0}

    df["mois"] = pd.to_datetime(df["mois"])
    df = df.sort_values("mois").reset_index(drop=True)

    # Build feature matrix
    features = pd.DataFrame(index=df.index)

    for col in ["niveau_moyen", "precipitation_totale", "temperature_moyenne", "evaporation_moyenne"]:
        if col in df.columns:
            features[col] = df[col]

    # Add derived features
    if "niveau_moyen" in df.columns:
        features["delta_niveau"] = df["niveau_moyen"].diff()
        features["niveau_rolling_3m"] = df["niveau_moyen"].rolling(3, min_periods=1).mean()

    if "precipitation_totale" in df.columns:
        features["precip_rolling_3m"] = df["precipitation_totale"].rolling(3, min_periods=1).mean()

    # Drop rows with NaN (from diff and early rolling)
    valid_mask = features.notna().all(axis=1)
    valid_features = features[valid_mask]
    valid_df = df[valid_mask]

    if len(valid_features) < 24:
        return {"months": [], "scores": [], "flagged": [], "n_flagged": 0, "n_total": 0}

    # Standardize
    means = valid_features.mean()
    stds = valid_features.std()
    stds = stds.replace(0, 1)
    X = ((valid_features - means) / stds).values

    # Fit Isolation Forest
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    predictions = clf.fit_predict(X)
    scores = clf.decision_function(X)

    # Build flagged months with reasons
    flagged = []
    for i in range(len(valid_df)):
        if predictions[i] == -1:
            row = valid_df.iloc[i]
            features_row = valid_features.iloc[i]

            # Find which features are most anomalous (z-score > 2)
            reasons = []
            z_scores = ((features_row - means) / stds)
            for col in z_scores.index:
                z = z_scores[col]
                if abs(z) > 2:
                    label = col.replace("_", " ").replace("niveau moyen", "level").replace("precipitation totale", "precip")
                    direction = "high" if z > 0 else "low"
                    reasons.append(f"{label} unusually {direction} ({z:.1f}σ)")

            flagged.append({
                "month": row["mois"].strftime("%Y-%m-%d"),
                "score": round(float(scores[i]), 3),
                "reason": "; ".join(reasons) if reasons else "multivariate anomaly",
            })

    return {
        "months": [d.strftime("%Y-%m-%d") for d in valid_df["mois"]],
        "scores": [round(float(s), 4) for s in scores],
        "flagged": flagged,
        "n_flagged": len(flagged),
        "n_total": len(valid_df),
    }
