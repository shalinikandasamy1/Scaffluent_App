#!/usr/bin/env python3
"""
Training script for Run 5: Improved weak class recall.

Changes from Run 4:
1. Higher augmentation for underrepresented classes
2. Increased image size (800 vs 640) for small object detection
3. Longer patience (20 epochs) to avoid premature stopping
4. Copy-paste augmentation enabled

Expected to improve: hose_reel (0.284), welding_sparks (0.291),
scaffold_net (0.468), tarpaulin (0.462)

Usage:
    python research/train_run5.py          # Start training
    python research/train_run5.py --dry-run  # Show config only

Prerequisites:
    - Dataset at research/merged_dataset_v3/
    - Activate venv: source .venv/bin/activate
    - GPU available (RTX 3060 recommended, ~3h for 100 epochs)
"""

import argparse
import sys
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
FIREEYE_DIR = SCRIPT_DIR.parent
DATASET_DIR = SCRIPT_DIR / "merged_dataset_v3"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
OUTPUT_DIR = Path("/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run5")
BASE_MODEL = "yolo11n.pt"

# Training configuration
TRAIN_CONFIG = {
    "data": str(DATASET_YAML),
    "epochs": 100,
    "imgsz": 800,           # Up from 640 → better small object detection
    "batch": 16,             # May need to reduce to 12 on 12GB GPU at imgsz=800
    "patience": 20,          # Up from default 10
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 5,
    "warmup_momentum": 0.5,
    "mosaic": 1.0,
    "copy_paste": 0.3,      # Copy-paste augmentation for rare objects
    "mixup": 0.15,           # Slight mixup to improve generalization
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.2,
    "scale": 0.5,
    "fliplr": 0.5,
    "project": str(OUTPUT_DIR.parent),
    "name": OUTPUT_DIR.name,
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,
    "val": True,
    "plots": True,
}


def main():
    parser = argparse.ArgumentParser(description="FireEye YOLO Run 5 training")
    parser.add_argument("--dry-run", action="store_true", help="Show config only")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size")
    args = parser.parse_args()

    TRAIN_CONFIG["batch"] = args.batch
    TRAIN_CONFIG["imgsz"] = args.imgsz

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset not found at {DATASET_YAML}")
        sys.exit(1)

    print("=" * 60)
    print("FireEye YOLO Run 5 Training Configuration")
    print("=" * 60)
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k:<20}: {v}")
    print()

    if args.dry_run:
        print("(dry run — not starting training)")
        return

    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    print(f"Starting training with {BASE_MODEL}...")
    results = model.train(**TRAIN_CONFIG)
    print(f"\nTraining complete. Results saved to: {OUTPUT_DIR}")

    # Copy best weights to models/
    best_weights = OUTPUT_DIR / "weights" / "best.pt"
    if best_weights.exists():
        dest = FIREEYE_DIR / "models" / "fireeye_yolo11n_v5.pt"
        dest.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(best_weights, dest)
        print(f"Best weights copied to: {dest}")


if __name__ == "__main__":
    main()
