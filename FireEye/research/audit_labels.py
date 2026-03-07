#!/usr/bin/env python3
"""
Audit label quality in the merged dataset.
Checks for:
- Suspiciously large/small bounding boxes
- Boxes outside image bounds
- Class distribution anomalies
- Images with extremely many detections
"""

import glob
import os
from collections import Counter, defaultdict


FIREEYE_CLASSES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}


def audit_labels(labels_dir, prefix_filter=None):
    """Audit YOLO label files for quality issues."""
    files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if prefix_filter:
        files = [f for f in files if os.path.basename(f).startswith(prefix_filter)]

    issues = defaultdict(list)
    class_counts = Counter()
    class_areas = defaultdict(list)
    boxes_per_image = []
    total_boxes = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]

        boxes_per_image.append(len(lines))
        if len(lines) > 30:
            issues["too_many_boxes"].append((fname, len(lines)))

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                issues["malformed_line"].append((fname, line))
                continue

            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            class_counts[cls_id] += 1
            total_boxes += 1
            area = w * h
            class_areas[cls_id].append(area)

            # Check bounds
            if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                issues["out_of_bounds_center"].append((fname, cls_id, cx, cy))
            if w > 0.95 and h > 0.95:
                issues["full_image_box"].append((fname, FIREEYE_CLASSES.get(cls_id, "?"), w, h))
            if w < 0.015 and h < 0.015:
                issues["tiny_box"].append((fname, FIREEYE_CLASSES.get(cls_id, "?"), w, h))
            if cls_id not in FIREEYE_CLASSES:
                issues["unknown_class"].append((fname, cls_id))

    # Report
    print(f"\n{'='*60}")
    print(f"LABEL AUDIT: {labels_dir}")
    if prefix_filter:
        print(f"Filter: {prefix_filter}*")
    print(f"{'='*60}")
    print(f"Files: {len(files)}")
    print(f"Total boxes: {total_boxes}")
    print(f"Empty files: {sum(1 for b in boxes_per_image if b == 0)}")

    if boxes_per_image:
        avg_boxes = sum(boxes_per_image) / len(boxes_per_image)
        max_boxes = max(boxes_per_image)
        print(f"Avg boxes/image: {avg_boxes:.1f}")
        print(f"Max boxes/image: {max_boxes}")

    print(f"\nClass distribution:")
    for cls_id in sorted(class_counts.keys()):
        name = FIREEYE_CLASSES.get(cls_id, f"unknown_{cls_id}")
        count = class_counts[cls_id]
        areas = class_areas[cls_id]
        avg_area = sum(areas) / len(areas) if areas else 0
        print(f"  {cls_id:2d} ({name:18s}): {count:5d}  avg_area={avg_area:.4f}")

    print(f"\nIssues found:")
    for issue_type, items in issues.items():
        print(f"  {issue_type}: {len(items)}")
        for item in items[:5]:
            print(f"    {item}")
        if len(items) > 5:
            print(f"    ... and {len(items)-5} more")

    # Missing classes
    missing = set(FIREEYE_CLASSES.keys()) - set(class_counts.keys())
    if missing:
        print(f"\nMISSING CLASSES (zero examples):")
        for cls_id in sorted(missing):
            print(f"  {cls_id}: {FIREEYE_CLASSES[cls_id]}")

    return {
        "total_files": len(files),
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "issues": {k: len(v) for k, v in issues.items()},
        "missing_classes": list(missing),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./merged_dataset/train/labels")
    parser.add_argument("--prefix", default=None, help="Filter by filename prefix")
    parser.add_argument("--by-source", action="store_true",
                        help="Break down by dataset source prefix")
    args = parser.parse_args()

    if args.by_source:
        prefixes = ["syn_", "ppe_", "dfire_", "weld_", "synt_", "augm_", "seve_"]
        for prefix in prefixes:
            audit_labels(args.dir, prefix_filter=prefix)
    else:
        audit_labels(args.dir, prefix_filter=args.prefix)
