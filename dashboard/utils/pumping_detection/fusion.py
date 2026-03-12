# dashboard/utils/pumping_detection/fusion.py
"""Fusion engine: concordance-based scoring across detection layers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FusionEngine:
    """Combine per-month flags from multiple layers into a suspicion score."""

    def __init__(self, merge_gap_days: int = 30):
        self.merge_gap_days = merge_gap_days

    def fuse(self, layer_flags: dict[str, pd.Series]) -> dict[str, Any]:
        """Fuse per-month boolean flags from available layers."""
        if not layer_flags:
            return {"suspect_windows": [], "global_score": 0.0, "per_month_details": []}

        n_layers = len(layer_flags)
        all_months = sorted(set().union(*(s.index for s in layer_flags.values())))
        index = pd.DatetimeIndex(all_months)

        per_month = []
        for month in index:
            flagged_by = []
            for name, flags in layer_flags.items():
                if month in flags.index and flags.loc[month]:
                    flagged_by.append(name)

            n_flagged = len(flagged_by)
            if n_flagged == n_layers:
                confidence = "high"
            elif n_flagged > 0 and n_flagged >= n_layers / 2:
                confidence = "medium"
            elif n_flagged > 0:
                confidence = "low"
            else:
                confidence = "clean"

            per_month.append({
                "month": str(month.date()),
                "confidence": confidence,
                "flagged_by": flagged_by,
                "concordance": n_flagged / n_layers if n_layers > 0 else 0,
            })

        suspect_windows = self._merge_windows(per_month)
        concordances = [m["concordance"] for m in per_month]
        global_score = sum(concordances) / len(concordances) if concordances else 0.0

        return {
            "suspect_windows": suspect_windows,
            "global_score": round(float(global_score), 3),
            "per_month_details": per_month,
        }

    def _merge_windows(self, per_month: list[dict]) -> list[dict]:
        """Merge adjacent suspect months into contiguous windows."""
        windows = []
        current_start = None
        current_months = []

        for entry in per_month:
            if entry["confidence"] != "clean":
                if current_start is None:
                    current_start = entry["month"]
                current_months.append(entry)
            else:
                if current_start is not None:
                    windows.append(self._build_window(current_start, current_months))
                    current_start = None
                    current_months = []

        if current_start is not None:
            windows.append(self._build_window(current_start, current_months))

        return windows

    def _build_window(self, start: str, months: list[dict]) -> dict:
        """Build a suspect window from consecutive suspect months."""
        confidences = [m["confidence"] for m in months]
        if "high" in confidences:
            confidence = "high"
        elif "medium" in confidences:
            confidence = "medium"
        else:
            confidence = "low"

        all_layers = set()
        for m in months:
            all_layers.update(m["flagged_by"])

        return {
            "start": start,
            "end": months[-1]["month"],
            "confidence": confidence,
            "duration_months": len(months),
            "layers": sorted(all_layers),
            "max_concordance": max(m["concordance"] for m in months),
        }
