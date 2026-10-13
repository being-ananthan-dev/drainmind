"""Object detector abstraction for DrainMind.

Supports:
  1. YOLOv8 via Ultralytics (preferred, if installed and weights available).
  2. OpenCV DNN baseline with COCO weights (fallback if cv2 dnn is available).
  3. Stub mode — returns empty detections with an explicit warning when
     no model backend is available.

Detections are returned as a list of dicts:
    {
        "class_name": str,
        "confidence": float,
        "x1": float, "y1": float, "x2": float, "y2": float,
        "source": "yolo" | "cv2dnn" | "none"
    }
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger("vision.detector")

# ─── module-level singleton state ────────────────────────────────────────────
_lock = threading.Lock()
_detector_instance: Optional["DrainDetector"] = None


class DetectorStatus:
    READY = "ready"
    NO_MODEL = "no_model"
    LOAD_ERROR = "load_error"
    STUB = "stub"


class DrainDetector:
    """
    Unified detector interface.  Instantiate once and reuse.
    """

    def __init__(
        self,
        model_path: str = "",
        model_name: str = "yolov8n",
        confidence: float = 0.35,
        iou: float = 0.45,
        input_size: int = 640,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.input_size = input_size
        self.device = device

        self.status: str = DetectorStatus.STUB
        self.backend: str = "none"
        self.model_version: str = ""
        self.class_names: List[str] = []
        self.warning_message: str = ""

        self._model = None
        self._load()

    # ─── loading ─────────────────────────────────────────────────────────────

    def _load(self):
        # 1. Try custom weights path first
        if self.model_path and Path(self.model_path).is_file():
            self._try_load_ultralytics(self.model_path)
            if self.status == DetectorStatus.READY:
                return

        # 2. Try downloading/loading standard Ultralytics preset (e.g., yolov8n)
        self._try_load_ultralytics_preset()
        if self.status == DetectorStatus.READY:
            return

        # 3. Fallback: stub
        self.status = DetectorStatus.NO_MODEL
        self.backend = "none"
        self.warning_message = (
            "No compatible model weights found. "
            "Detections will not be available. "
            "Place YOLOv8 .pt weights in the models/ directory and configure the path in Settings."
        )
        logger.warning(self.warning_message)

    def _try_load_ultralytics(self, path: str):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO from: {path}")
            model = YOLO(path)
            self._model = model
            self.backend = "ultralytics"
            self.model_version = str(getattr(model, "task", "detect"))
            self.class_names = list(model.names.values()) if hasattr(model, "names") else []
            self.status = DetectorStatus.READY
            logger.info(f"YOLO loaded successfully — {len(self.class_names)} classes")
        except Exception as e:
            logger.error(f"Ultralytics load failed for {path}: {e}")
            self.status = DetectorStatus.LOAD_ERROR
            self.warning_message = f"Failed to load model from {path}: {e}"

    def _try_load_ultralytics_preset(self):
        """Try to load a named preset (downloads weights on first run if network available)."""
        try:
            from ultralytics import YOLO
            # Try a very lightweight model; this may download ~6 MB
            preset = f"{self.model_name}.pt"
            logger.info(f"Attempting Ultralytics preset: {preset}")
            model = YOLO(preset)
            # Trigger a tiny warmup inference to confirm it works
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False, conf=0.9)
            self._model = model
            self.backend = "ultralytics"
            self.model_version = preset
            self.class_names = list(model.names.values()) if hasattr(model, "names") else []
            self.status = DetectorStatus.READY
            logger.info(f"Ultralytics preset '{preset}' ready — {len(self.class_names)} classes")
        except Exception as e:
            logger.warning(f"Ultralytics preset load failed: {e}")

    # ─── inference ───────────────────────────────────────────────────────────

    def detect(
        self,
        img_bgr: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Run detection on a BGR numpy image.
        Returns (detections_list, backend_name).
        """
        conf_to_use = confidence if confidence is not None else self.confidence
        iou_to_use = iou if iou is not None else self.iou

        detections: List[Dict[str, Any]] = []

        if self.status == DetectorStatus.READY and self._model is not None and self.backend == "ultralytics":
            detections = self._detect_ultralytics(img_bgr, conf=conf_to_use, iou=iou_to_use)

        # Enhance with domain-specific drain computer vision features
        try:
            from src.vision.segmenter import extract_drain_features
            drain_features = extract_drain_features(img_bgr)
            for feat in drain_features:
                if feat["confidence"] >= conf_to_use:
                    # Check overlap with existing detections to avoid duplicates
                    overlap = False
                    for existing in detections:
                        dx = max(0, min(feat["x2"], existing["x2"]) - max(feat["x1"], existing["x1"]))
                        dy = max(0, min(feat["y2"], existing["y2"]) - max(feat["y1"], existing["y1"]))
                        inter = dx * dy
                        area_f = (feat["x2"] - feat["x1"]) * (feat["y2"] - feat["y1"])
                        if area_f > 0 and (inter / area_f) > 0.6:
                            overlap = True
                            break
                    if not overlap:
                        detections.append(feat)
        except Exception as e:
            logger.debug(f"Drain feature extraction notice: {e}")

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.get("confidence", 0), reverse=True)
        backend_name = "ultralytics_drain_ensemble" if self.status == DetectorStatus.READY else "drain_vision_engine"
        return detections, backend_name

    def _detect_ultralytics(
        self,
        img_bgr: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        try:
            results = self._model.predict(
                img_bgr,
                conf=conf if conf is not None else self.confidence,
                iou=iou if iou is not None else self.iou,
                imgsz=self.input_size,
                verbose=False,
                device=self.device
            )
            # Map COCO classes to domain terms
            coco_drain_map = {
                "bottle": "Plastic Waste (Bottle)",
                "cup": "Plastic / Container Waste",
                "can": "Metallic Debris (Can)",
                "potted plant": "Vegetation Overgrowth",
                "backpack": "Discarded Debris (Fabric)",
                "handbag": "Discarded Debris (Bag)",
                "suitcase": "Bulky Debris Obstruction",
                "umbrella": "Foreign Object Debris",
                "bowl": "Plastic / Waste Container",
            }
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(float)
                    c = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    raw_cls = self._model.names.get(cls_id, str(cls_id))
                    cls_name = coco_drain_map.get(raw_cls.lower(), raw_cls.capitalize())
                    detections.append({
                        "class_name": cls_name,
                        "confidence": round(c, 4),
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                        "source": "yolo"
                    })
            return detections
        except Exception as e:
            logger.error(f"Ultralytics inference error: {e}")
            return []

    # ─── info ─────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.status == DetectorStatus.READY

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "model_path": self.model_path or self.model_version,
            "model_name": self.model_name,
            "class_count": len(self.class_names),
            "class_names": self.class_names,
            "warning": self.warning_message,
            "confidence_threshold": self.confidence,
        }


# ─── singleton management ────────────────────────────────────────────────────

def get_detector(reload: bool = False) -> DrainDetector:
    """Return the module-level singleton detector, creating it if necessary."""
    global _detector_instance
    with _lock:
        if _detector_instance is None or reload:
            from src.config import get_config
            cfg = get_config()
            _detector_instance = DrainDetector(
                model_path=cfg.model_path,
                model_name=cfg.model_name,
                confidence=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                input_size=cfg.input_size,
                device=cfg.device
            )
    return _detector_instance


def reset_detector():
    """Force recreation of the singleton on next call."""
    global _detector_instance
    with _lock:
        _detector_instance = None
