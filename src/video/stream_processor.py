"""Video stream processor for DrainMind.

Processes video files frame-by-frame (sampled) with high-accuracy drain vision
analytics and aggregates multi-frame results for reporting and database storage.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Callable, Tuple

import cv2
import numpy as np

from src.utils.logging_config import get_logger
from src.video.capture import open_video, release_video, sample_frames
from src.vision.segmenter import compute_masks
from src.analysis.coverage import compute_coverage, define_central_roi
from src.vision.visualization import draw_detections

logger = get_logger("video.stream_processor")


def process_video_file(
    video_path: str,
    detector,
    sample_interval: int = 10,
    target_size: Optional[Tuple[int, int]] = (640, 480),
    max_frames: int = 500,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> Dict[str, Any]:
    """
    Process a video file and return aggregated inspection results.
    """
    cap, info = open_video(video_path)
    if cap is None or not info["opened"]:
        return {
            "success": False,
            "error": f"Could not open video: {video_path}",
            "video_info": info,
        }

    total_frames_in_video = info.get("frame_count", 0)
    expected_samples = (
        min(max_frames, total_frames_in_video // sample_interval)
        if total_frames_in_video > 0
        else max_frames
    )

    frames_processed = 0
    all_detections: List[Dict[str, Any]] = []
    class_counter: Dict[str, int] = defaultdict(int)
    coverage_history: List[float] = []
    water_history: List[float] = []

    best_annotated_frame: Optional[np.ndarray] = None
    max_blockage_frame: Optional[np.ndarray] = None
    max_blockage_pct: float = 0.0
    max_water_frame: Optional[np.ndarray] = None
    max_water_pct: float = 0.0

    fps_samples: deque = deque(maxlen=20)
    t_start = time.perf_counter()

    try:
        for frame_idx, frame in sample_frames(
            cap, sample_interval=sample_interval,
            max_frames=max_frames, target_size=target_size
        ):
            t_frame_start = time.perf_counter()
            frames_processed += 1
            h, w = frame.shape[:2]
            roi = define_central_roi(frame.shape)

            # 1. Run detection ensemble
            detections, backend = detector.detect(frame)
            for d in detections:
                all_detections.append(d)
                class_counter[d["class_name"]] += 1

            # 2. Run multi-cue drain segmentation
            mask_dict = compute_masks(frame)
            debris_mask = mask_dict.get("debris_mask")
            water_mask = mask_dict.get("water_mask")

            deb_cov = compute_coverage(debris_mask, roi, frame.shape)
            wat_cov = compute_coverage(water_mask, roi, frame.shape)

            # Fallback/blend with box area if mask is minimal but objects detected
            if detections:
                det_area = sum(
                    (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
                    for d in detections
                )
                box_cov = min(100.0, (det_area / (h * w)) * 100.0)
                frame_cov = max(deb_cov if deb_cov is not None else 0.0, box_cov)
            else:
                frame_cov = deb_cov if deb_cov is not None else 0.0

            coverage_history.append(round(frame_cov, 1))
            water_history.append(round(wat_cov if wat_cov is not None else 0.0, 1))

            # Track peak blockage keyframe
            if frame_cov >= max_blockage_pct:
                max_blockage_pct = frame_cov
                max_blockage_frame = frame.copy()

            # Track peak water keyframe
            if wat_cov is not None and wat_cov >= max_water_pct:
                max_water_pct = wat_cov
                max_water_frame = frame.copy()

            # Keep annotated frame
            if detections:
                annotated = draw_detections(frame, detections)
                best_annotated_frame = annotated
            elif best_annotated_frame is None:
                best_annotated_frame = frame.copy()

            t_frame_end = time.perf_counter()
            fps_samples.append(1.0 / max(t_frame_end - t_frame_start, 1e-6))

            if progress_callback:
                current_fps = float(np.mean(fps_samples)) if fps_samples else 0.0
                progress_callback(
                    frames_processed,
                    expected_samples,
                    {
                        "fps": round(current_fps, 1),
                        "detections_total": len(all_detections),
                        "class_counter": dict(class_counter),
                        "current_coverage": round(frame_cov, 1),
                    },
                )

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {
            "success": False,
            "error": str(e),
            "frames_processed": frames_processed,
            "video_info": info,
        }
    finally:
        release_video(cap)

    elapsed = time.perf_counter() - t_start
    avg_fps = frames_processed / elapsed if elapsed > 0 else 0.0

    # Aggregate coverage
    avg_coverage = float(np.mean(coverage_history)) if coverage_history else 0.0
    peak_coverage = float(np.max(coverage_history)) if coverage_history else 0.0
    avg_water = float(np.mean(water_history)) if water_history else 0.0
    peak_water = float(np.max(water_history)) if water_history else 0.0

    # If annotated frame was never set, annotate max blockage frame
    if best_annotated_frame is None and max_blockage_frame is not None:
        best_annotated_frame = max_blockage_frame

    return {
        "success": True,
        "video_info": info,
        "frames_processed": frames_processed,
        "total_detections": len(all_detections),
        "class_summary": dict(class_counter),
        "avg_coverage_estimate": round(avg_coverage, 1),
        "max_coverage_estimate": round(peak_coverage, 1),
        "avg_water_estimate": round(avg_water, 1),
        "max_water_estimate": round(peak_water, 1),
        "coverage_history": coverage_history,
        "water_history": water_history,
        "avg_fps": round(avg_fps, 1),
        "elapsed_seconds": round(elapsed, 1),
        "last_annotated_frame": best_annotated_frame,
        "max_blockage_frame": max_blockage_frame,
        "backend": detector.backend if detector else "drain_vision_ensemble",
    }
