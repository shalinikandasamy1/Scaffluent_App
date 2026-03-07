#!/usr/bin/env python3
"""Deploy a trained YOLO model to the FireEye models/ directory.

Copies weights, updates .env, and runs evaluation to verify.

Usage:
    python research/deploy_model.py /path/to/best.pt --name fireeye_yolo11n_v5.pt
    python research/deploy_model.py run5   # shorthand for known runs
"""

import argparse
import shutil
import sys
from pathlib import Path

FIREEYE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = FIREEYE_DIR / "models"
ENV_FILE = FIREEYE_DIR / ".env"

KNOWN_RUNS = {
    "run4": Path("/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run4/weights/best.pt"),
    "run5": Path("/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run5/weights/best.pt"),
}


def main():
    parser = argparse.ArgumentParser(description="Deploy YOLO model to FireEye")
    parser.add_argument("source", help="Path to model weights or known run name (run4, run5)")
    parser.add_argument("--name", default=None, help="Filename in models/ (default: auto)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after deploy")
    args = parser.parse_args()

    # Resolve source path
    if args.source in KNOWN_RUNS:
        source = KNOWN_RUNS[args.source]
        default_name = f"fireeye_yolo11n_{args.source.replace('run', 'v')}.pt"
    else:
        source = Path(args.source)
        default_name = f"fireeye_{source.stem}.pt"

    if not source.exists():
        print(f"ERROR: {source} does not exist")
        sys.exit(1)

    name = args.name or default_name
    dest = MODELS_DIR / name

    # Copy weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"Copied {source} -> {dest} ({size_mb:.1f} MB)")

    # Update .env
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("FIREEYE_YOLO_MODEL_NAME="):
                old = line.split("=", 1)[1]
                lines[i] = f"FIREEYE_YOLO_MODEL_NAME=models/{name}"
                print(f"Updated .env: {old} -> models/{name}")
                updated = True
                break
        if updated:
            ENV_FILE.write_text("\n".join(lines) + "\n")
    else:
        print("No .env file found; set FIREEYE_YOLO_MODEL_NAME manually")

    # Run evaluation
    if not args.no_eval:
        print("\nRunning heuristic evaluation...")
        sys.path.insert(0, str(FIREEYE_DIR))
        from evaluate import evaluate, save_results
        metrics = evaluate(heuristic_only=True)
        if metrics:
            save_results(metrics)


if __name__ == "__main__":
    main()
