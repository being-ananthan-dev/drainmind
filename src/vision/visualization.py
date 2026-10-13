"""Annotation visualization for DrainMind.

Draws bounding boxes, labels, confidence values, and coverage overlays
on BGR images using OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger("vision.visualization")

# ─── colour palette ──────────────────────────────────────────────────────────
# BGR colours keyed by rough semantic class groups
_CLASS_COLORS = {
    "default":      (70, 130, 60),   # forest green
    "debris":       (30, 100, 200),  # amber-orange
    "trash":        (30, 100, 200),
    "garbage":      (30, 100, 200),
    "leaves":       (50, 160, 50),
    "water":        (200, 90, 30),
    "flood":        (180, 60, 20),
    "standing":     (200, 90, 30),
    "block":        (20, 20, 200),   # red
    "obstruction":  (20, 20, 200),
    "drain":        (140, 90, 50),
    "pipe":         (140, 90, 50),
}

OVERLAY_ALPHA = 0.35


def score_color(score: Optional[float]) -> str:
    """Return hex color corresponding to risk score."""
    if score is None:
        return "#737A73"
    if score >= 70:
        return "#8B1A1A"
    elif score >= 50:
        return "#B87333"
    elif score >= 30:
        return "#4A7C8E"
    else:
        return "#245B45"


def _get_color(class_name: str) -> tuple:
    name_lower = class_name.lower()
    for key, color in _CLASS_COLORS.items():
        if key in name_lower:
            return color
    return _CLASS_COLORS["default"]


def draw_detections(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    show_confidence: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of img.
    """
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.7, w / 1280))
    font_thickness = 1

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = _get_color(det.get("class_name", ""))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        if show_confidence:
            label = f"{det['class_name']} {det['confidence']:.2f}"
        else:
            label = det["class_name"]

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y1, th + 4)
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 2, label_y + baseline), color, -1)
        cv2.putText(out, label, (x1 + 1, label_y - 1),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return out


def draw_coverage_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple = (30, 100, 200),
    alpha: float = OVERLAY_ALPHA,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Blend a binary mask as a coloured overlay onto img.
    mask must be 0/255 uint8 of the same spatial size as img.
    """
    if mask is None or mask.sum() == 0:
        return img.copy()

    out = img.copy()
    h, w = out.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(out)
    overlay[mask > 0] = color_bgr
    out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)

    if label:
        cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_roi(
    img: np.ndarray,
    roi: tuple,
    color: tuple = (70, 180, 70),
    label: str = "ROI",
) -> np.ndarray:
    """Draw a rectangle showing the region of interest."""
    out = img.copy()
    x1, y1, x2, y2 = [int(v) for v in roi]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def add_info_bar(
    img: np.ndarray,
    lines: List[str],
    position: str = "bottom",
    bg_color: tuple = (30, 30, 30),
    text_color: tuple = (220, 220, 200),
) -> np.ndarray:
    """Append an information strip at top or bottom of img."""
    h, w = img.shape[:2]
    line_h = 22
    bar_h = line_h * len(lines) + 10

    bar = np.full((bar_h, w, 3), bg_color, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (10, (i + 1) * line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    if position == "bottom":
        return np.vstack([img, bar])
    else:
        return np.vstack([bar, img])


def compose_annotated_image(
    original: np.ndarray,
    detections: List[Dict[str, Any]],
    coverage_info: Optional[Dict[str, Any]] = None,
    priority_score: Optional[float] = None,
    priority_category: Optional[str] = None,
    location_name: str = "",
    timestamp: str = "",
) -> np.ndarray:
    """
    Full annotation pipeline: boxes + info bar.
    """
    annotated = draw_detections(original, detections)

    info_lines = []
    if location_name:
        info_lines.append(f"Location: {location_name}")
    if timestamp:
        info_lines.append(f"Timestamp: {timestamp}")
    if detections:
        info_lines.append(f"Detections: {len(detections)}")
    else:
        info_lines.append("Detections: none")
    if coverage_info:
        debris = coverage_info.get("debris_coverage")
        water = coverage_info.get("water_coverage")
        if debris is not None:
            info_lines.append(f"Visible obstruction coverage: {debris:.1f}%")
        if water is not None:
            info_lines.append(f"Standing water indicator: {water:.1f}%")
    if priority_score is not None:
        info_lines.append(
            f"Inspection priority score: {priority_score:.1f}/100 [{priority_category or ''}]"
        )
    info_lines.append("DrainMind — Computer Vision Inspection System")

    return add_info_bar(annotated, info_lines, position="bottom")
