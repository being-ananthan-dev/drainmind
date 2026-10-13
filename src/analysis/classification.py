"""Backward-compatibility re-export of condition assessment routines."""

from src.analysis.condition_assessment import (
    classify_detections,
    assess_quality,
    DEBRIS_LIKE_CLASSES,
    WATER_LIKE_CLASSES,
)

__all__ = [
    "classify_detections",
    "assess_quality",
    "DEBRIS_LIKE_CLASSES",
    "WATER_LIKE_CLASSES",
]
