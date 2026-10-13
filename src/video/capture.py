"""Video capture utilities for DrainMind."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Generator

import cv2
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("video.capture")


def save_upload_to_temp(file_obj) -> Tuple[str, bool]:
    """
    Save a Streamlit UploadedFile to a temporary path.
    Returns (temp_path, success).
    """
    try:
        suffix = Path(getattr(file_obj, "name", "video.mp4")).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        file_obj.seek(0)
        return tmp_path, True
    except Exception as e:
        logger.error(f"Failed to save uploaded video to temp: {e}")
        return "", False


def cleanup_temp(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {path}: {e}")


def open_video(source) -> Tuple[Optional[cv2.VideoCapture], dict]:
    """
    Open a video source (path string or camera index).
    Returns (cap, info_dict).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None, {"opened": False, "source": str(source)}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, {
        "opened": True,
        "source": str(source),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps > 0 and frame_count > 0 else None,
    }


def release_video(cap: Optional[cv2.VideoCapture]):
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def sample_frames(
    cap: cv2.VideoCapture,
    sample_interval: int = 10,
    max_frames: int = 300,
    target_size: Optional[Tuple[int, int]] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, frame_bgr) at every `sample_interval`-th frame.
    """
    frame_idx = 0
    emitted = 0

    while emitted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            if target_size and target_size[0] > 0 and target_size[1] > 0:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            yield frame_idx, frame
            emitted += 1

        frame_idx += 1


def check_webcam(index: int = 0) -> dict:
    """
    Test whether a webcam is accessible and can deliver frames.
    Returns diagnostic dict.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    if not cap.isOpened():
        return {"available": False, "index": index, "error": "VideoCapture did not open."}

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return {
            "available": False,
            "index": index,
            "error": "Camera opened but could not read a frame.",
        }

    h, w = frame.shape[:2]
    return {
        "available": True,
        "index": index,
        "frame_width": w,
        "frame_height": h,
        "error": None,
    }
