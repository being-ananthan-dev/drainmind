"""Coverage estimation for DrainMind.

Computes visible obstruction and water coverage within a region of interest (ROI).

Coverage formula:
    coverage% = intersection_pixels(mask, roi_mask) / roi_pixels * 100

All measurements are of VISIBLE pixels in the image only.
They do NOT represent hydraulic blockage or physical drain capacity.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from src.utils.logging_config import get_logger

logger = get_logger("analysis.coverage")


def define_central_roi(
    img_shape: Tuple[int, int],
    fraction: float = 0.6
) -> Tuple[int, int, int, int]:
    """
    Return a central rectangular ROI as (x1, y1, x2, y2).
    Defaults to the central 60% of the image.
    """
    h, w = img_shape[:2]
    margin_x = int(w * (1 - fraction) / 2)
    margin_y = int(h * (1 - fraction) / 2)
    x1 = margin_x
    y1 = margin_y
    x2 = w - margin_x
    y2 = h - margin_y
    return x1, y1, x2, y2


def roi_mask(img_shape: Tuple[int, int], roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Create a binary mask (0/255) for the given ROI."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = roi
    mask[y1:y2, x1:x2] = 255
    return mask


def compute_coverage(
    feature_mask: Optional[np.ndarray],
    drain_roi: Optional[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int],
) -> Optional[float]:
    """
    Compute what percentage of the drain ROI is covered by feature_mask.

    Returns None if:
    - feature_mask is None
    - drain_roi is None
    - ROI pixel count is zero

    Returns float in [0, 100].
    """
    if feature_mask is None:
        return None

    h, w = img_shape[:2]

    if drain_roi is None:
        drain_roi = define_central_roi(img_shape)

    x1, y1, x2, y2 = drain_roi
    roi_h = y2 - y1
    roi_w = x2 - x1
    if roi_h <= 0 or roi_w <= 0:
        logger.warning("ROI has zero area — coverage unavailable")
        return None

    # Crop and resize mask to match img dimensions if needed
    if feature_mask.shape[:2] != (h, w):
        feature_mask = cv2.resize(feature_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Ensure binary
    bin_mask = (feature_mask > 0).astype(np.uint8)

    roi_region = bin_mask[y1:y2, x1:x2]
    roi_pixels = roi_h * roi_w
    intersection_pixels = int(roi_region.sum())

    if roi_pixels == 0:
        return None

    coverage = (intersection_pixels / roi_pixels) * 100.0
    return round(float(coverage), 2)


def estimate_coverage_from_detections(
    detections: list,
    img_shape: Tuple[int, int],
    drain_roi: Optional[Tuple[int, int, int, int]] = None,
    debris_classes: set = None,
    water_classes: set = None,
) -> Dict[str, Any]:
    """
    Approximate coverage by projecting detection bounding boxes onto the ROI.

    This is a bounding-box approximation, not a pixel-accurate mask.
    Results are labelled accordingly.
    """
    if debris_classes is None:
        # Broad set of COCO-class names that could indicate debris-like objects
        debris_classes = {
            "backpack", "handbag", "suitcase", "bottle", "cup", "bowl",
            "banana", "apple", "orange", "sandwich", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "potted plant", "vase",
            "book", "scissors", "teddy bear", "toothbrush", "hair drier",
            "cell phone", "remote", "keyboard", "mouse", "umbrella",
            "tie", "sports ball", "frisbee", "skis", "snowboard",
            "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "fork", "knife", "spoon",
        }
    if water_classes is None:
        water_classes = set()  # COCO has no water class

    h, w = img_shape[:2]
    if drain_roi is None:
        drain_roi = define_central_roi(img_shape)
    x1r, y1r, x2r, y2r = drain_roi
    roi_area = max(1, (x2r - x1r) * (y2r - y1r))

    debris_mask = np.zeros((h, w), dtype=np.uint8)
    water_mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        bx1 = int(max(0, det["x1"]))
        by1 = int(max(0, det["y1"]))
        bx2 = int(min(w, det["x2"]))
        by2 = int(min(h, det["y2"]))
        cls = det.get("class_name", "").lower()

        if cls in {c.lower() for c in debris_classes}:
            debris_mask[by1:by2, bx1:bx2] = 255
        elif cls in {c.lower() for c in water_classes}:
            water_mask[by1:by2, bx1:bx2] = 255

    debris_cov = compute_coverage(debris_mask, drain_roi, img_shape)
    water_cov = compute_coverage(water_mask, drain_roi, img_shape)

    return {
        "debris_coverage": debris_cov,
        "water_coverage": water_cov,
        "method": "bbox_projection",
        "quality": "approximate",
        "note": (
            "Coverage is approximated from bounding-box areas. "
            "This is not a pixel-accurate segmentation."
        )
    }
