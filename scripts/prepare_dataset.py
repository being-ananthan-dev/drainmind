"""
DrainMind — Dataset Preparation Script

This script helps prepare a dataset of drainage images for YOLO fine-tuning.
It validates annotation files, generates a dataset YAML, and provides
train/val/test split guidance.

Usage:
    python scripts/prepare_dataset.py --source /path/to/raw_data --output data/train_dataset

IMPORTANT:
    Split by LOCATION or RECORDING SESSION, not randomly by frame.
    Near-identical video frames from the same session must not appear in both
    train and val/test splits (data leakage).
"""

import argparse
import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict


RECOMMENDED_CLASSES = [
    "drain_blocked",
    "drain_partial",
    "drain_clear",
    "standing_water",
    "debris",
    "leaves",
    "trash",
]


def validate_yolo_annotation(label_path: Path, num_classes: int) -> list:
    """Check a YOLO .txt label file for common errors. Returns list of warnings."""
    warnings = []
    try:
        with open(label_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    warnings.append(f"Line {i+1}: expected 5 values, got {len(parts)}")
                    continue
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= num_classes:
                    warnings.append(f"Line {i+1}: class id {cls_id} out of range [0, {num_classes-1}]")
                coords = [float(x) for x in parts[1:]]
                for j, c in enumerate(coords):
                    if not (0.0 <= c <= 1.0):
                        warnings.append(f"Line {i+1}: coordinate {j} ({c}) out of [0,1]")
    except Exception as e:
        warnings.append(f"Cannot read file: {e}")
    return warnings


def split_sessions(sessions: list, train_frac=0.7, val_frac=0.15) -> tuple:
    """Split a list of sessions (not frames) into train/val/test."""
    random.shuffle(sessions)
    n = len(sessions)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    train = sessions[:n_train]
    val = sessions[n_train:n_train + n_val]
    test = sessions[n_train + n_val:]
    return train, val, test


def prepare_dataset(
    source_dir: str,
    output_dir: str,
    class_names: list = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42
):
    random.seed(seed)
    src = Path(source_dir)
    out = Path(output_dir)
    if class_names is None:
        class_names = RECOMMENDED_CLASSES

    print(f"\nDrainMind Dataset Preparation")
    print(f"Source: {src}")
    print(f"Output: {out}")
    print(f"Classes ({len(class_names)}): {class_names}\n")

    if not src.exists():
        print(f"ERROR: Source directory does not exist: {src}")
        return

    # Discover images
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = sorted([p for p in src.rglob("*") if p.suffix.lower() in image_exts])

    if not all_images:
        print(f"No images found in {src}")
        return

    # Group images by parent directory (= session)
    sessions_dict = defaultdict(list)
    for img in all_images:
        sessions_dict[str(img.parent)].append(img)

    sessions = list(sessions_dict.keys())
    print(f"Found {len(all_images)} images in {len(sessions)} session(s).\n")

    if len(sessions) < 3:
        print("WARNING: Fewer than 3 sessions found. Consider collecting more diverse recordings.")
        print("Proceeding with best-effort split.\n")

    train_s, val_s, test_s = split_sessions(sessions, train_frac, val_frac)

    splits = {"train": train_s, "val": val_s, "test": test_s}
    counts = {}

    for split, session_list in splits.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for sess in session_list:
            for img in sessions_dict[sess]:
                shutil.copy2(img, img_dir / img.name)
                label_file = img.with_suffix(".txt")
                if label_file.exists():
                    warns = validate_yolo_annotation(label_file, len(class_names))
                    if warns:
                        print(f"  WARNINGS in {label_file.name}:")
                        for w in warns:
                            print(f"    {w}")
                    shutil.copy2(label_file, lbl_dir / label_file.name)
                n += 1
        counts[split] = n
        print(f"{split:5s}: {n:4d} images ({len(session_list)} sessions)")

    # Generate YAML
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = out / "drain_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\nDataset YAML written to: {yaml_path}")
    print(f"\nTraining command:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640 device=cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrainMind dataset preparation")
    parser.add_argument("--source", required=True, help="Source directory with images and YOLO .txt labels")
    parser.add_argument("--output", default="data/train_dataset", help="Output directory for split dataset")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names (space-separated)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        class_names=args.classes,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed
    )
