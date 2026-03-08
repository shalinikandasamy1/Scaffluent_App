#!/usr/bin/env python3
"""
Compare per-class validation metrics between YOLO training runs.

Runs YOLO val on each model and compares per-class AP50, recall, and precision.

Usage:
    python research/compare_per_class.py run4 run5
    python research/compare_per_class.py  # auto-detect available runs
"""

import sys
from pathlib import Path

FIREEYE_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = FIREEYE_DIR / "research" / "merged_dataset_v3" / "dataset.yaml"

# Known run locations
RUNS = {
    "run4": Path("/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run4/weights/best.pt"),
    "run5": Path("/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run5/weights/best.pt"),
    "deployed": FIREEYE_DIR / "models" / "fireeye_yolo11n_v4.pt",
}

CLASS_NAMES = [
    "fire", "smoke", "fire_extinguisher", "gas_cylinder", "scaffold_net",
    "exit_sign", "hard_hat", "safety_vest", "welding_sparks",
    "hose_reel", "person", "tarpaulin",
]


def evaluate_model(model_path: str, name: str) -> dict:
    """Run YOLO validation and extract per-class metrics."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.val(data=str(DATASET_YAML), imgsz=640, batch=16, verbose=False)

    per_class = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(results.box.ap50):
            per_class[cls_name] = {
                "ap50": float(results.box.ap50[i]),
                "recall": float(results.box.r[i]) if hasattr(results.box, 'r') else None,
                "precision": float(results.box.p[i]) if hasattr(results.box, 'p') else None,
            }

    return {
        "name": name,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "per_class": per_class,
    }


def main():
    # Determine which runs to compare
    run_names = sys.argv[1:] if len(sys.argv) > 1 else []

    if not run_names:
        # Auto-detect
        run_names = [name for name, path in RUNS.items() if path.exists()]

    if not run_names:
        print("No model weights found to compare.")
        return

    models = {}
    for name in run_names:
        if name in RUNS and RUNS[name].exists():
            path = RUNS[name]
        else:
            path = Path(name)
            if not path.exists():
                print(f"SKIP: {name} not found")
                continue
            name = path.stem

        print(f"Evaluating {name}...")
        models[name] = evaluate_model(path, name)

    if len(models) < 1:
        print("No models evaluated.")
        return

    # Print comparison table
    names = list(models.keys())
    print(f"\n{'='*70}")
    print(f"PER-CLASS COMPARISON")
    print(f"{'='*70}\n")

    # Header
    header = f"{'Class':<20}"
    for n in names:
        header += f"  {n + ' AP50':>15}"
    print(header)
    print("-" * len(header))

    # Per-class rows
    for cls_name in CLASS_NAMES:
        row = f"{cls_name:<20}"
        for n in names:
            val = models[n]["per_class"].get(cls_name, {}).get("ap50", 0)
            row += f"  {val:>14.3f}"
        # Show delta if 2 models
        if len(names) == 2:
            v1 = models[names[0]]["per_class"].get(cls_name, {}).get("ap50", 0)
            v2 = models[names[1]]["per_class"].get(cls_name, {}).get("ap50", 0)
            delta = v2 - v1
            sign = "+" if delta >= 0 else ""
            row += f"  ({sign}{delta:.3f})"
        print(row)

    # Overall
    print("-" * len(header))
    row = f"{'mAP50':<20}"
    for n in names:
        row += f"  {models[n]['mAP50']:>14.3f}"
    if len(names) == 2:
        d = models[names[1]]["mAP50"] - models[names[0]]["mAP50"]
        sign = "+" if d >= 0 else ""
        row += f"  ({sign}{d:.3f})"
    print(row)

    row = f"{'mAP50-95':<20}"
    for n in names:
        row += f"  {models[n]['mAP50_95']:>14.3f}"
    print(row)
    print()


if __name__ == "__main__":
    main()
