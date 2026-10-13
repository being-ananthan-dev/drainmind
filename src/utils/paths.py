"""Path utilities for DrainMind."""

import os
import re
import uuid
from pathlib import Path
from datetime import datetime


def safe_filename(name: str, max_len: int = 80) -> str:
    """Sanitize a string for use as a filename component."""
    name = re.sub(r'[^\w\s\-.]', '_', name)
    name = re.sub(r'[\s]+', '_', name.strip())
    return name[:max_len]


def timestamped_filename(prefix: str, extension: str) -> str:
    """Generate a filename like prefix_20240101_120000_<uuid4_short>.ext"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    ext = extension.lstrip(".")
    return f"{safe_filename(prefix)}_{ts}_{uid}.{ext}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_within(child: Path, parent: Path) -> bool:
    """Check that child is under parent (prevent path traversal)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_output_path(base_dir: str | Path, filename: str) -> Path:
    """Return an absolute path inside base_dir after sanitizing filename."""
    base = ensure_dir(base_dir)
    safe = safe_filename(Path(filename).name)
    candidate = (base / safe).resolve()
    if not is_within(candidate, base):
        raise ValueError(f"Path traversal prevented for: {filename}")
    return candidate
