#!/usr/bin/env python3
"""
Clean noisy labels from the merged dataset.
Fixes:
1. Remove full-image bounding boxes (>95% of image) — these are Grounding DINO false positives
2. Remove tiny boxes (<1.5% in both dimensions) — likely noise
3. Remove welding/synthetic/augmented background-only images that dilute training
"""

import glob
import os
import shutil
from collections import Counter


FIREEYE_CLASSES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}


def clean_labels(dataset_dir, dry_run=False):
    """Clean noisy labels in train and val splits."""

    stats = {
        "full_image_removed": 0,
        "tiny_removed": 0,
        "bg_images_removed": 0,
        "files_cleaned": 0,
    }

    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")

        if not os.path.exists(lbl_dir):
            continue

        label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

        for lbl_path in label_files:
            fname = os.path.basename(lbl_path)
            base = os.path.splitext(fname)[0]

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            # Check if this is a background-only image from noisy sources
            # that adds no training value
            is_empty = len(lines) == 0
            is_noisy_source = any(base.startswith(p) for p in
                                  ["weld_", "synt_", "augm_", "seve_"])

            if is_empty and is_noisy_source:
                # Remove background-only images from noisy sources
                if not dry_run:
                    os.remove(lbl_path)
                    img_path = None
                    for ext in [".jpg", ".png", ".jpeg"]:
                        p = os.path.join(img_dir, f"{base}{ext}")
                        if os.path.exists(p):
                            img_path = p
                            break
                    if img_path:
                        os.remove(img_path)
                stats["bg_images_removed"] += 1
                continue

            # Clean individual boxes
            cleaned = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Remove full-image boxes (>90% in both dims)
                if w > 0.90 and h > 0.90:
                    stats["full_image_removed"] += 1
                    continue

                # Remove tiny boxes
                if w < 0.015 and h < 0.015:
                    stats["tiny_removed"] += 1
                    continue

                cleaned.append(line)

            if cleaned != lines:
                stats["files_cleaned"] += 1
                if not dry_run:
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(cleaned))

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./merged_dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"{'DRY RUN - ' if args.dry_run else ''}Cleaning labels in {args.dir}")
    stats = clean_labels(args.dir, dry_run=args.dry_run)

    print(f"\nResults:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")
