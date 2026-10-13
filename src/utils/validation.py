"""File and input validation for DrainMind."""

import os
from pathlib import Path
from typing import Tuple, Optional

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def validate_image_file(
    file_obj,
    max_size_mb: float = 20.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded image file object (Streamlit UploadedFile).
    Returns (is_valid, error_message).
    """
    if file_obj is None:
        return False, "No file provided."

    name = getattr(file_obj, "name", "")
    ext = Path(name).suffix.lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    size_bytes = getattr(file_obj, "size", None)
    if size_bytes is None:
        # Try to measure by reading
        try:
            data = file_obj.read()
            file_obj.seek(0)
            size_bytes = len(data)
        except Exception:
            size_bytes = 0

    max_bytes = int(max_size_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        return False, (
            f"File size {size_bytes / (1024*1024):.1f} MB exceeds "
            f"maximum {max_size_mb:.0f} MB."
        )

    return True, None


def validate_video_file(
    file_obj,
    max_size_mb: float = 200.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded video file object.
    Returns (is_valid, error_message).
    """
    if file_obj is None:
        return False, "No file provided."

    name = getattr(file_obj, "name", "")
    ext = Path(name).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    size_bytes = getattr(file_obj, "size", None)
    if size_bytes is None:
        try:
            data = file_obj.read()
            file_obj.seek(0)
            size_bytes = len(data)
        except Exception:
            size_bytes = 0

    max_bytes = int(max_size_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        return False, (
            f"File size {size_bytes / (1024*1024):.1f} MB exceeds "
            f"maximum {max_size_mb:.0f} MB."
        )

    return True, None


def validate_path_string(path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
    """Validate a filesystem path string."""
    if not path or not path.strip():
        return False, "Path is empty."
    p = Path(path)
    if must_exist and not p.exists():
        return False, f"Path does not exist: {path}"
    return True, None


def validate_confidence(value: float) -> Tuple[bool, Optional[str]]:
    if not (0.0 < value <= 1.0):
        return False, f"Confidence threshold must be between 0 and 1 (exclusive 0). Got {value}."
    return True, None


def validate_positive_int(value: int, name: str = "value") -> Tuple[bool, Optional[str]]:
    if not isinstance(value, int) or value <= 0:
        return False, f"{name} must be a positive integer. Got {value}."
    return True, None


def validate_uploaded_file(file_bytes: bytes, filename: str, max_size_mb: float = 20.0) -> Tuple[bool, Optional[str]]:
    """Validate uploaded image raw bytes and filename."""
    if not file_bytes:
        return False, "File is empty."
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
    if len(file_bytes) > int(max_size_mb * 1024 * 1024):
        return False, f"File size {len(file_bytes)/(1024*1024):.1f} MB exceeds {max_size_mb:.0f} MB."
    return True, None


def validate_uploaded_video(file_bytes: bytes, filename: str, max_size_mb: float = 200.0) -> Tuple[bool, Optional[str]]:
    """Validate uploaded video raw bytes and filename."""
    if not file_bytes:
        return False, "File is empty."
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
    if len(file_bytes) > int(max_size_mb * 1024 * 1024):
        return False, f"File size {len(file_bytes)/(1024*1024):.1f} MB exceeds {max_size_mb:.0f} MB."
    return True, None


def sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters."""
    import re
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', Path(name).name)
