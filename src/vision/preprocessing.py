"""Image preprocessing pipeline for DrainMind."""

import io
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError
from src.utils.logging_config import get_logger

logger = get_logger("vision.preprocessing")


def load_image_from_upload(file_obj) -> Optional[np.ndarray]:
    """
    Load a Streamlit UploadedFile into a BGR numpy array (OpenCV convention).
    Returns None on failure.
    """
    try:
        import cv2
        data = file_obj.read()
        file_obj.seek(0)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as e:
        logger.error(f"Failed to load image from upload: {e}")
        return None


def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """Load raw image bytes into a BGR numpy array."""
    try:
        import cv2
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        return None


def load_image_from_path(path: str) -> Optional[np.ndarray]:
    """Load image from a filesystem path into BGR numpy array."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"cv2.imread could not read: {path}")
    return img


def resize_for_inference(
    img: np.ndarray, target_size: int = 640
) -> np.ndarray:
    """Resize image so longest edge = target_size, preserving aspect ratio."""
    import cv2
    h, w = img.shape[:2]
    if max(h, w) <= target_size:
        return img
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1].copy()


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1].copy()


def numpy_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img_bgr))


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return rgb_to_bgr(np.array(img.convert("RGB")))


def save_image(img: np.ndarray, path: str) -> bool:
    """Save BGR numpy array to disk. Returns True on success."""
    import cv2
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), img)
        if not ok:
            logger.error(f"cv2.imwrite failed for: {path}")
        return ok
    except Exception as e:
        logger.error(f"save_image error: {e}")
        return False


def image_to_bytes(img: np.ndarray, fmt: str = ".png") -> Optional[bytes]:
    """Encode image to bytes for download."""
    import cv2
    try:
        ok, buf = cv2.imencode(fmt, img)
        if ok:
            return buf.tobytes()
        return None
    except Exception as e:
        logger.error(f"image_to_bytes error: {e}")
        return None
