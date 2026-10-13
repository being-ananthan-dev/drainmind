"""Condition assessment helpers for DrainMind."""

from __future__ import annotations

from typing import Dict, Any, Optional, List


# Classes that indicate potential drainage obstruction
DEBRIS_LIKE_CLASSES = {
    "backpack", "handbag", "suitcase", "bottle", "cup", "bowl",
    "banana", "apple", "orange", "sandwich", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "potted plant", "vase",
    "book", "scissors", "teddy bear", "toothbrush", "hair drier",
    "cell phone", "umbrella", "tie", "sports ball", "frisbee",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "fork", "knife", "spoon",
    "debris", "trash", "garbage", "leaves", "leaf", "block", "obstruction",
}

WATER_LIKE_CLASSES = {"water", "flood", "puddle", "standing water", "wet"}


def classify_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Separate detections into drainage-relevant categories.

    Returns a dict with:
        debris_detections, water_detections, other_detections,
        has_debris, has_water
    """
    debris = []
    water = []
    other = []

    for d in detections:
        cls = d.get("class_name", "").lower().strip()
        if cls in DEBRIS_LIKE_CLASSES or any(kw in cls for kw in ["debris", "trash", "garbage", "leaf", "block"]):
            debris.append(d)
        elif cls in WATER_LIKE_CLASSES or any(kw in cls for kw in ["water", "flood", "puddle", "wet"]):
            water.append(d)
        else:
            other.append(d)

    return {
        "debris_detections": debris,
        "water_detections": water,
        "other_detections": other,
        "has_debris": len(debris) > 0,
        "has_water": len(water) > 0,
    }


def assess_quality(
    detections: List[Dict[str, Any]],
    coverage_method: Optional[str],
    image_size: Optional[tuple] = None,
) -> str:
    """
    Estimate assessment quality based on available evidence.

    Returns one of: 'good', 'acceptable', 'low', 'unknown'
    """
    if not detections and coverage_method in (None, "none"):
        return "low"

    confidences = [d.get("confidence", 0.0) for d in detections]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    if avg_conf >= 0.6 and coverage_method == "yolo_segmentation":
        return "good"
    elif avg_conf >= 0.4 or coverage_method in ("bbox_projection", "opencv_heuristic"):
        return "acceptable"
    elif avg_conf > 0 or coverage_method is not None:
        return "low"

    return "unknown"
