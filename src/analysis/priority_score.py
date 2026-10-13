"""Inspection priority scoring for DrainMind.

Formula:
    R = w_D * D + w_W * W + w_T * T

Where:
    D = normalised visible-obstruction score  (0–100)
    W = normalised standing-water indicator   (0–100)
    T = normalised worsening-trend score      (0–100)

Weights default to w_D=0.50, w_W=0.30, w_T=0.20.

If a component is unavailable the score is renormalised over available
components and flagged as INCOMPLETE.

The score is an INSPECTION PRIORITY INDICATOR only.
It is NOT a validated flood-risk coefficient.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

from src.utils.logging_config import get_logger

logger = get_logger("analysis.priority_score")


CATEGORY_LABELS = [
    (75, "High inspection priority"),
    (50, "Inspection recommended"),
    (25, "Monitor"),
    (0,  "Lower inspection priority"),
]


def _category(score: float, thresholds: Optional[Dict[str, int]] = None) -> str:
    if thresholds:
        t_high = thresholds.get("inspection_recommended", 75)
        t_mid  = thresholds.get("monitor", 50)
        t_low  = thresholds.get("lower_priority", 25)
    else:
        t_high, t_mid, t_low = 75, 50, 25

    if score >= t_high:
        return "High inspection priority"
    elif score >= t_mid:
        return "Inspection recommended"
    elif score >= t_low:
        return "Monitor"
    else:
        return "Lower inspection priority"


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def compute_priority_score(
    debris_coverage: Optional[float] = None,
    water_coverage: Optional[float] = None,
    trend_score: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Compute the inspection priority score.

    Parameters
    ----------
    debris_coverage : float | None
        Visible obstruction percentage [0–100], or None if unavailable.
    water_coverage : float | None
        Standing-water indicator percentage [0–100], or None if unavailable.
    trend_score : float | None
        Worsening trend normalised to [0–100], or None if unavailable.
    weights : dict | None
        Override default weights {debris, water, trend}.
    thresholds : dict | None
        Override default category thresholds.

    Returns
    -------
    dict with keys:
        score           float | None
        category        str | None
        is_complete     bool
        missing         list[str]
        components      dict
        weights_used    dict
        note            str
    """
    if weights is None:
        try:
            from src.config import get_config
            weights = get_config().scoring_weights
        except Exception:
            weights = {"debris": 0.50, "water": 0.30, "trend": 0.20}

    w_d = float(weights.get("debris", 0.50))
    w_w = float(weights.get("water", 0.30))
    w_t = float(weights.get("trend", 0.20))

    components = {}
    missing = []
    weighted_sum = 0.0
    total_weight = 0.0

    if debris_coverage is not None:
        d = _clamp(debris_coverage)
        components["debris"] = round(d, 2)
        weighted_sum += w_d * d
        total_weight += w_d
    else:
        missing.append("debris_coverage")

    if water_coverage is not None:
        w = _clamp(water_coverage)
        components["water"] = round(w, 2)
        weighted_sum += w_w * w
        total_weight += w_w
    else:
        missing.append("water_coverage")

    if trend_score is not None:
        t = _clamp(trend_score)
        components["trend"] = round(t, 2)
        weighted_sum += w_t * t
        total_weight += w_t
    else:
        missing.append("trend_score")

    is_complete = len(missing) == 0

    if total_weight == 0.0:
        return {
            "score": None,
            "category": None,
            "is_complete": False,
            "missing": missing,
            "components": components,
            "weights_used": {"debris": w_d, "water": w_w, "trend": w_t},
            "note": "No measurements available — score cannot be calculated.",
        }

    # Renormalise if components missing
    raw_score = weighted_sum / total_weight
    score = round(_clamp(raw_score), 1)
    category = _category(score, thresholds)

    note = ""
    if not is_complete:
        note = (
            f"Score renormalised over available components "
            f"({', '.join([c for c in ['debris','water','trend'] if c not in missing])}). "
            f"Missing: {', '.join(missing)}. Confidence is reduced."
        )
    else:
        note = "All components available."

    return {
        "score": score,
        "category": category,
        "is_complete": is_complete,
        "missing": missing,
        "components": components,
        "weights_used": {"debris": w_d, "water": w_w, "trend": w_t},
        "note": note,
    }


def compute_trend_score(
    current_coverage: Optional[float],
    previous_coverage: Optional[float],
) -> Optional[float]:
    """
    Convert a coverage change into a 0–100 trend score.
    Positive change (worsening) → higher score.
    Returns None if either input is None.
    """
    if current_coverage is None or previous_coverage is None:
        return None

    delta = current_coverage - previous_coverage
    # Map delta [-100, +100] → [0, 100]
    # delta=+100 → 100, delta=0 → 50, delta=-100 → 0
    trend = _clamp(50.0 + delta / 2.0)
    return round(trend, 1)
