#!/usr/bin/env python3
"""
Monitor YOLO training progress from results.csv.
Posts updates to e-paper display.
"""

import csv
import json
import time
import urllib.request
import sys
import os


EPAPER_URL = "http://192.168.50.72:8090/api/message"


def post_epaper(header, body):
    """Post message to e-paper display."""
    try:
        data = json.dumps({"header": header, "body": body}).encode()
        req = urllib.request.Request(EPAPER_URL, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"  (e-paper post failed: {e})")


def read_results(csv_path):
    """Read training results CSV."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def format_metrics(row):
    """Format key metrics from a results row."""
    epoch = row.get("epoch", "?")
    mAP50 = float(row.get("metrics/mAP50(B)", 0))
    mAP5095 = float(row.get("metrics/mAP50-95(B)", 0))
    precision = float(row.get("metrics/precision(B)", 0))
    recall = float(row.get("metrics/recall(B)", 0))
    box_loss = float(row.get("train/box_loss", 0))
    cls_loss = float(row.get("train/cls_loss", 0))
    return {
        "epoch": epoch,
        "mAP50": mAP50,
        "mAP5095": mAP5095,
        "precision": precision,
        "recall": recall,
        "box_loss": box_loss,
        "cls_loss": cls_loss,
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/evnchn/Scaffluent_App/runs/detect/yolo_finetune/merged_run1/results.csv"
    total_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Monitoring: {csv_path}")
    print(f"Total epochs: {total_epochs}")

    last_epoch = 0
    while True:
        rows = read_results(csv_path)
        if rows:
            latest = format_metrics(rows[-1])
            current_epoch = int(latest["epoch"])

            if current_epoch > last_epoch:
                last_epoch = current_epoch
                print(f"\nEpoch {current_epoch}/{total_epochs}:")
                print(f"  mAP50={latest['mAP50']:.3f}  mAP50-95={latest['mAP5095']:.3f}")
                print(f"  P={latest['precision']:.3f}  R={latest['recall']:.3f}")
                print(f"  box_loss={latest['box_loss']:.3f}  cls_loss={latest['cls_loss']:.3f}")

                # Post to e-paper every 5 epochs
                if current_epoch % 5 == 0 or current_epoch == total_epochs:
                    header = f"YOLO Train {current_epoch}/{total_epochs}"
                    body = f"mAP50={latest['mAP50']:.2f} P={latest['precision']:.2f}\nR={latest['recall']:.2f} loss={latest['box_loss']:.2f}"
                    post_epaper(header, body)

                if current_epoch >= total_epochs:
                    print("\nTraining complete!")
                    # Final summary
                    best_map = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)
                    best_epoch = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
                    print(f"Best mAP50: {best_map:.4f} at epoch {best_epoch.get('epoch')}")
                    post_epaper("YOLO Training Done!",
                                f"Best mAP50={best_map:.2f}\nEp {best_epoch.get('epoch')}/{total_epochs}")
                    break

        time.sleep(30)


if __name__ == "__main__":
    main()
