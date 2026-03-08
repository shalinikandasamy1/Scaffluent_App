#!/usr/bin/env python3
"""
Apply NMS to YOLO label files to remove duplicate/overlapping boxes.
Useful after Grounding DINO auto-labeling which can produce duplicates
when querying with multiple synonym prompts.
"""

import glob
import os
from collections import Counter


def iou(box1, box2):
    """Calculate IoU between two YOLO boxes (cx, cy, w, h)."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def nms_labels(labels, iou_threshold=0.5):
    """Apply NMS per class to a list of (class_id, cx, cy, w, h) tuples."""
    by_class = {}
    for lbl in labels:
        cid = lbl[0]
        by_class.setdefault(cid, []).append(lbl)

    kept = []
    for cid, boxes in by_class.items():
        # Sort by box area (larger boxes first as proxy for confidence)
        boxes.sort(key=lambda b: b[3] * b[4], reverse=True)
        keep = []
        for box in boxes:
            suppress = False
            for kept_box in keep:
                if iou(box[1:], kept_box[1:]) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                keep.append(box)
        kept.extend(keep)
    return kept


def process_directory(label_dir, iou_threshold=0.5, dry_run=False):
    files = sorted(glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True))
    total_before = 0
    total_after = 0
    files_changed = 0

    for fpath in files:
        labels = []
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    labels.append((cid, *coords))

        total_before += len(labels)
        cleaned = nms_labels(labels, iou_threshold)
        total_after += len(cleaned)

        if len(cleaned) < len(labels):
            files_changed += 1
            if not dry_run:
                with open(fpath, 'w') as f:
                    for lbl in cleaned:
                        f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

    removed = total_before - total_after
    print(f"Processed {len(files)} files")
    print(f"  Before: {total_before} labels")
    print(f"  After:  {total_after} labels")
    print(f"  Removed: {removed} duplicates ({100*removed/total_before:.1f}%)")
    print(f"  Files changed: {files_changed}")
    if dry_run:
        print("  (dry run - no files modified)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Labels directory")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    process_directory(args.dir, iou_threshold=args.iou, dry_run=args.dry_run)
