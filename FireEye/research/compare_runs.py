#!/usr/bin/env python3
"""
Compare multiple YOLO training runs side by side.
Reads results.csv from each run and produces a summary.
"""

import csv
import os
import sys


def read_results(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_run(name, rows):
    """Extract key metrics from a training run."""
    if not rows:
        return None

    best_map50 = max(float(r["metrics/mAP50(B)"]) for r in rows)
    best_map5095 = max(float(r["metrics/mAP50-95(B)"]) for r in rows)
    best_epoch = next(r for r in rows if float(r["metrics/mAP50(B)"]) == best_map50)

    last = rows[-1]
    return {
        "name": name,
        "epochs": len(rows),
        "best_mAP50": best_map50,
        "best_mAP50_95": best_map5095,
        "best_epoch": int(best_epoch["epoch"]),
        "final_mAP50": float(last["metrics/mAP50(B)"]),
        "final_precision": float(last["metrics/precision(B)"]),
        "final_recall": float(last["metrics/recall(B)"]),
        "final_box_loss": float(last["train/box_loss"]),
        "final_cls_loss": float(last["train/cls_loss"]),
        "total_time": float(last["time"]),
    }


def main():
    runs = {}

    # Auto-detect runs
    local_run1 = "/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run1/results.csv"
    if os.path.exists(local_run1):
        runs["153_AdamW_run1"] = read_results(local_run1)

    local_run2_path = "/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run2/results.csv"
    if os.path.exists(local_run2_path):
        runs["153_AdamW_run2"] = read_results(local_run2_path)

    local_run3 = "/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run3/results.csv"
    if os.path.exists(local_run3):
        runs["153_AdamW_run3"] = read_results(local_run3)

    local_run4 = "/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run4/results.csv"
    if os.path.exists(local_run4):
        runs["153_AdamW_run4"] = read_results(local_run4)

    # .172 Tesla P4 SGD run
    sgd_run = "/home/evnchn/fireeye_data/runs/detect/yolo_finetune/merged_sgd_run/results.csv"
    if os.path.exists(sgd_run):
        runs["172_SGD_run"] = read_results(sgd_run)

    # Also check local copy of SGD results
    sgd_local = "/home/evnchn/Scaffluent_App/FireEye/research/sgd_run_results/results.csv"
    if not os.path.exists(sgd_run) and os.path.exists(sgd_local):
        runs["172_SGD_run"] = read_results(sgd_local)

    # Also accept CLI paths
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            name = os.path.basename(os.path.dirname(arg))
            runs[name] = read_results(arg)

    if not runs:
        print("No training results found.")
        return

    print(f"\n{'='*70}")
    print(f"TRAINING RUN COMPARISON")
    print(f"{'='*70}\n")

    summaries = []
    for name, rows in runs.items():
        s = summarize_run(name, rows)
        if s:
            summaries.append(s)

    # Header
    header = f"{'Metric':<25}"
    for s in summaries:
        header += f"{s['name']:>20}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Epochs completed", "epochs", "d"),
        ("Best mAP50", "best_mAP50", ".4f"),
        ("Best mAP50-95", "best_mAP50_95", ".4f"),
        ("Best epoch", "best_epoch", "d"),
        ("Final mAP50", "final_mAP50", ".4f"),
        ("Final Precision", "final_precision", ".4f"),
        ("Final Recall", "final_recall", ".4f"),
        ("Final box_loss", "final_box_loss", ".4f"),
        ("Final cls_loss", "final_cls_loss", ".4f"),
        ("Total time (s)", "total_time", ".0f"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<25}"
        for s in summaries:
            val = s[key]
            row += f"{val:>20{fmt}}"
        print(row)

    # Winner
    if len(summaries) > 1:
        best = max(summaries, key=lambda s: s["best_mAP50"])
        print(f"\nBest overall: {best['name']} (mAP50={best['best_mAP50']:.4f} at epoch {best['best_epoch']})")


if __name__ == "__main__":
    main()
