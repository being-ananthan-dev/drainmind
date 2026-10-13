"""DrainMind configuration loader and manager."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent
CONFIG_FILE = ROOT_DIR / "configs" / "settings.yaml"
SETTINGS_OVERRIDE_FILE = ROOT_DIR / "configs" / "user_settings.yaml"


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """Application configuration with user override support."""

    def __init__(self):
        base = _load_yaml(CONFIG_FILE)
        user = _load_yaml(SETTINGS_OVERRIDE_FILE)
        self._cfg = _deep_merge(base, user)
        self._resolve_paths()

    def _resolve_paths(self):
        """Make relative paths absolute from ROOT_DIR."""
        db_path = self._cfg.get("database", {}).get("path", "data/drainmind.db")
        if not os.path.isabs(db_path):
            self._cfg["database"]["path"] = str(ROOT_DIR / db_path)

        ann_dir = self._cfg.get("output", {}).get("annotated_dir", "outputs/annotated")
        if not os.path.isabs(ann_dir):
            self._cfg["output"]["annotated_dir"] = str(ROOT_DIR / ann_dir)

        rep_dir = self._cfg.get("output", {}).get("reports_dir", "outputs/reports")
        if not os.path.isabs(rep_dir):
            self._cfg["output"]["reports_dir"] = str(ROOT_DIR / rep_dir)

        log_file = self._cfg.get("logging", {}).get("file", "data/drainmind.log")
        if not os.path.isabs(log_file):
            self._cfg["logging"]["file"] = str(ROOT_DIR / log_file)

    def get(self, *keys, default=None):
        obj = self._cfg
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k, default)
            else:
                return default
        return obj

    @property
    def app_name(self) -> str:
        return self.get("app", "name", default="DrainMind")

    @property
    def app_version(self) -> str:
        return self.get("app", "version", default="1.0.0")

    @property
    def model_path(self) -> str:
        return self.get("model", "path", default="")

    @property
    def model_name(self) -> str:
        return self.get("model", "name", default="yolov8n")

    @property
    def confidence_threshold(self) -> float:
        return float(self.get("model", "confidence_threshold", default=0.35))

    @property
    def iou_threshold(self) -> float:
        return float(self.get("model", "iou_threshold", default=0.45))

    @property
    def input_size(self) -> int:
        return int(self.get("model", "input_size", default=640))

    @property
    def device(self) -> str:
        return self.get("model", "device", default="cpu")

    @property
    def frame_sample_interval(self) -> int:
        return int(self.get("video", "frame_sample_interval", default=10))

    @property
    def max_video_upload_mb(self) -> int:
        return int(self.get("video", "max_upload_size_mb", default=200))

    @property
    def max_image_upload_mb(self) -> int:
        return int(self.get("image", "max_upload_size_mb", default=20))

    @property
    def db_path(self) -> str:
        return self.get("database", "path", default=str(ROOT_DIR / "data" / "drainmind.db"))

    @property
    def annotated_dir(self) -> str:
        return self.get("output", "annotated_dir", default=str(ROOT_DIR / "outputs" / "annotated"))

    @property
    def reports_dir(self) -> str:
        return self.get("output", "reports_dir", default=str(ROOT_DIR / "outputs" / "reports"))

    @property
    def scoring_weights(self) -> dict:
        return self.get("scoring", "weights", default={"debris": 0.50, "water": 0.30, "trend": 0.20})

    @property
    def scoring_thresholds(self) -> dict:
        return self.get("scoring", "thresholds", default={
            "lower_priority": 25, "monitor": 50, "inspection_recommended": 75
        })

    def update_and_save(self, updates: dict):
        """Merge updates into user settings and persist."""
        existing = _load_yaml(SETTINGS_OVERRIDE_FILE)
        merged = _deep_merge(existing, updates)
        SETTINGS_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_OVERRIDE_FILE, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False)
        # Reinitialize
        self.__init__()

    def as_dict(self) -> dict:
        return self._cfg.copy()


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config():
    global _config
    _config = Config()
    return _config
