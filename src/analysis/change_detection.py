"""Temporal change detection and image alignment for DrainMind."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

from src.utils.logging_config import get_logger

logger = get_logger("analysis.change_detection")

MIN_MATCHES_FOR_ALIGNMENT = 10
ALIGNMENT_QUALITY_THRESHOLD = 0.15  # max allowed mean reprojection error normalised


def try_align_images(
    img_ref: np.ndarray,
    img_query: np.ndarray,
    max_features: int = 500,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Attempt feature-based image alignment using ORB + BFMatcher + homography.

    Returns (warped_query, alignment_info).
    warped_query is None if alignment fails.
    """
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_q = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_q, des_q = orb.detectAndCompute(gray_q, None)

    if des_ref is None or des_q is None or len(des_ref) < 4 or len(des_q) < 4:
        return None, {
            "aligned": False,
            "quality": "poor",
            "reason": "Insufficient keypoints",
            "match_count": 0,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_ref, des_q)
    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = [m for m in matches if m.distance < 60]

    if len(good_matches) < MIN_MATCHES_FOR_ALIGNMENT:
        return None, {
            "aligned": False,
            "quality": "poor",
            "reason": f"Too few good matches ({len(good_matches)} < {MIN_MATCHES_FOR_ALIGNMENT})",
            "match_count": len(good_matches),
        }

    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_q = np.float32([kp_q[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_q, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        return None, {
            "aligned": False,
            "quality": "poor",
            "reason": "Homography estimation failed",
            "match_count": len(good_matches),
        }

    inlier_ratio = float(mask.sum()) / len(mask) if len(mask) > 0 else 0.0
    h, w = img_ref.shape[:2]
    warped = cv2.warpPerspective(img_query, H, (w, h))

    quality = "good" if inlier_ratio >= 0.5 and len(good_matches) >= 30 else "acceptable"

    return warped, {
        "aligned": True,
        "quality": quality,
        "inlier_ratio": round(inlier_ratio, 3),
        "match_count": len(good_matches),
        "reason": "Alignment successful",
    }


def compute_coverage_change(
    current_coverage: Optional[float],
    previous_coverage: Optional[float],
) -> Dict[str, Any]:
    """
    Compute the signed change in coverage percentage.

    Returns:
        change: float | None
        direction: 'improving' | 'worsening' | 'stable' | 'unavailable'
        reliable: bool
        note: str
    """
    if current_coverage is None or previous_coverage is None:
        return {
            "change": None,
            "direction": "unavailable",
            "reliable": False,
            "note": "One or both coverage values are unavailable.",
        }

    delta = current_coverage - previous_coverage

    if abs(delta) < 2.0:
        direction = "stable"
    elif delta > 0:
        direction = "worsening"
    else:
        direction = "improving"

    return {
        "change": round(delta, 2),
        "direction": direction,
        "reliable": True,
        "note": f"Coverage changed by {delta:+.1f}%.",
    }


def compare_inspections(
    inspection_a: Dict[str, Any],
    inspection_b: Dict[str, Any],
    img_a: Optional[np.ndarray] = None,
    img_b: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Full comparison between two inspections.
    Optionally attempts image alignment if both images are provided.
    """
    obs_a = inspection_a.get("observation", {}) or {}
    obs_b = inspection_b.get("observation", {}) or {}

    cov_a = obs_a.get("debris_coverage")
    cov_b = obs_b.get("debris_coverage")
    water_a = obs_a.get("water_coverage")
    water_b = obs_b.get("water_coverage")
    score_a = obs_a.get("priority_score")
    score_b = obs_b.get("priority_score")

    debris_change = compute_coverage_change(cov_b, cov_a)  # b is newer
    water_change = compute_coverage_change(water_b, water_a)
    score_diff = (
        round(score_b - score_a, 1)
        if score_a is not None and score_b is not None
        else None
    )

    alignment_info = {"aligned": False, "quality": "not_attempted", "reason": "No images provided"}
    warped_img = None

    if img_a is not None and img_b is not None:
        warped_img, alignment_info = try_align_images(img_a, img_b)

    return {
        "debris_change": debris_change,
        "water_change": water_change,
        "score_difference": score_diff,
        "alignment": alignment_info,
        "warped_image": warped_img,
        "reliable": (
            debris_change.get("reliable", False)
            and alignment_info.get("quality", "poor") in ("good", "acceptable", "not_attempted")
        ),
        "note": (
            "Comparison may be unreliable if images were taken from different angles or "
            "under significantly different lighting conditions."
            if not alignment_info.get("aligned")
            else ""
        ),
    }
