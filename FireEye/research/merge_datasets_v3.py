#!/usr/bin/env python3
"""
Merge datasets v3: builds on v2 by adding targeted weak class data
(tarpaulin, hose_reel, welding_sparks) to address class imbalance.
"""

import os
import random
import shutil
import glob
import time
from collections import Counter

FIREEYE_CLASSES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}

PPE_TO_FIREEYE = {0: 6, 2: 7, 6: 10}
DFIRE_TO_FIREEYE = {0: 1, 1: 0}


def remap_labels(src_label, class_map, dst_label):
    if not os.path.exists(src_label):
        return 0
    lines = []
    with open(src_label) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(parts[0])
            if old_cls in class_map:
                new_cls = class_map[old_cls]
                lines.append(f"{new_cls} {' '.join(parts[1:])}")
    with open(dst_label, 'w') as f:
        f.write('\n'.join(lines))
    return len(lines)


def add_dataset(name, img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                class_map=None, prefix="", max_images=None, skip_empty=False):
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                    glob.glob(os.path.join(img_dir, "*.png")))
    if max_images and len(images) > max_images:
        random.shuffle(images)
        images = images[:max_images]

    count = 0
    labels_count = 0
    skipped = 0

    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        dst_name = f"{prefix}{base}" if prefix else base
        src_lbl = os.path.join(lbl_dir, f"{base}.txt")

        if skip_empty:
            has_labels = False
            if class_map is not None and os.path.exists(src_lbl):
                with open(src_lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) in class_map:
                            has_labels = True
                            break
            elif os.path.exists(src_lbl):
                with open(src_lbl) as f:
                    has_labels = any(l.strip() for l in f)
            if not has_labels:
                skipped += 1
                continue

        dst_img = os.path.join(out_img_dir, f"{dst_name}{ext}")
        if not os.path.exists(dst_img):
            shutil.copy2(img_path, dst_img)

        dst_lbl = os.path.join(out_lbl_dir, f"{dst_name}.txt")
        if class_map is not None:
            n = remap_labels(src_lbl, class_map, dst_lbl)
            labels_count += n
        elif os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            with open(src_lbl) as f:
                labels_count += sum(1 for line in f if line.strip())
        else:
            open(dst_lbl, 'w').close()

        count += 1

    msg = f"  {name}: {count} images, {labels_count} labels"
    if skipped:
        msg += f" ({skipped} empty skipped)"
    print(msg)
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/evnchn/Scaffluent_App/FireEye/research/merged_dataset_v3")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = args.output

    staging_img = os.path.join(out, "_staging", "images")
    staging_lbl = os.path.join(out, "_staging", "labels")
    os.makedirs(staging_img, exist_ok=True)
    os.makedirs(staging_lbl, exist_ok=True)

    total = 0
    print("Merging datasets v3 (with weak class augmentation)...\n")

    # 1. Synthetic SDXL-Turbo
    synth_img = "/home/evnchn/Scaffluent_App/FireEye/research/fireeye_dataset/all_images"
    synth_lbl = "/home/evnchn/Scaffluent_App/FireEye/research/fireeye_dataset/all_labels"
    if os.path.exists(synth_img):
        total += add_dataset("Synthetic (SDXL-Turbo)", synth_img, synth_lbl,
                             staging_img, staging_lbl, prefix="syn_")

    # 2. Construction-PPE
    ppe_base = "/home/evnchn/datasets/construction-ppe"
    for split in ["train", "val", "test"]:
        ppe_img = os.path.join(ppe_base, "images", split)
        ppe_lbl = os.path.join(ppe_base, "labels", split)
        if os.path.exists(ppe_img):
            total += add_dataset(f"Construction-PPE ({split})", ppe_img, ppe_lbl,
                                 staging_img, staging_lbl,
                                 class_map=PPE_TO_FIREEYE, prefix=f"ppe_{split}_",
                                 skip_empty=True)

    # 3. Welding frames
    weld_img = "/home/evnchn/Scaffluent_App/FireEye/research/welding_frames"
    weld_lbl = "/home/evnchn/Scaffluent_App/FireEye/research/welding_labels"
    if os.path.exists(weld_img) and os.path.exists(weld_lbl):
        total += add_dataset("Welding frames", weld_img, weld_lbl,
                             staging_img, staging_lbl, prefix="weld_",
                             skip_empty=True)

    # 4. D-Fire
    dfire_base = "/home/evnchn/Scaffluent_App/FireEye/research/DFireDataset/data"
    for split in ["train", "val"]:
        dfire_img = os.path.join(dfire_base, split, "images")
        dfire_lbl = os.path.join(dfire_base, split, "labels")
        if os.path.exists(dfire_img):
            total += add_dataset(f"D-Fire ({split})", dfire_img, dfire_lbl,
                                 staging_img, staging_lbl,
                                 class_map=DFIRE_TO_FIREEYE,
                                 prefix=f"dfire_{split}_",
                                 max_images=3000,
                                 skip_empty=True)

    # 5. Fire extinguisher targeted
    ext_img = "/home/evnchn/Scaffluent_App/FireEye/research/extinguisher_data/images"
    ext_lbl = "/home/evnchn/Scaffluent_App/FireEye/research/extinguisher_data/labels"
    if os.path.exists(ext_img):
        total += add_dataset("Fire extinguisher (targeted)", ext_img, ext_lbl,
                             staging_img, staging_lbl, prefix="ext_",
                             skip_empty=True)

    # 6. NEW in v3: Weak class targeted data
    weak_base = "/home/evnchn/Scaffluent_App/FireEye/research/weak_class_data"
    for cls_name in ["tarpaulin", "hose_reel", "welding_sparks"]:
        w_img = os.path.join(weak_base, f"{cls_name}_images")
        w_lbl = os.path.join(weak_base, f"{cls_name}_labels")
        if os.path.exists(w_img) and os.path.exists(w_lbl):
            total += add_dataset(f"Weak class: {cls_name}", w_img, w_lbl,
                                 staging_img, staging_lbl,
                                 prefix=f"weak_{cls_name}_",
                                 skip_empty=True)

    print(f"\nTotal staged: {total} images")

    # Create train/val split
    all_images = sorted(f for f in os.listdir(staging_img)
                        if f.endswith(('.jpg', '.png')))
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - args.val_fraction))
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
        split_img = os.path.join(out, split, "images")
        split_lbl = os.path.join(out, split, "labels")
        os.makedirs(split_img, exist_ok=True)
        os.makedirs(split_lbl, exist_ok=True)

        for fname in img_list:
            base = os.path.splitext(fname)[0]
            src = os.path.join(staging_img, fname)
            dst = os.path.join(split_img, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(src), dst)
            src_lbl = os.path.join(staging_lbl, f"{base}.txt")
            dst_lbl = os.path.join(split_lbl, f"{base}.txt")
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    # Class distribution
    print("\n=== Class distribution ===")
    stats = Counter()
    for split in ["train", "val"]:
        lbl_dir = os.path.join(out, split, "labels")
        for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        stats[FIREEYE_CLASSES.get(cls_id, f"unknown_{cls_id}")] += 1

    for name, count in stats.most_common():
        print(f"  {name}: {count}")

    missing = set(FIREEYE_CLASSES.values()) - set(stats.keys())
    if missing:
        print(f"\n  WARNING - Missing classes: {missing}")

    # Write dataset.yaml
    yaml_path = os.path.join(out, "dataset.yaml")
    yaml_content = f"""# FireEye Merged Training Dataset v3
# Generated: {time.strftime('%Y-%m-%d %H:%M')}
# Train: {len(train_imgs)}, Val: {len(val_imgs)}
# v3: Added targeted weak class data (tarpaulin, hose_reel, welding_sparks)

path: {os.path.abspath(out)}
train: train/images
val: val/images

names:
  0: fire
  1: smoke
  2: fire_extinguisher
  3: gas_cylinder
  4: scaffold_net
  5: exit_sign
  6: hard_hat
  7: safety_vest
  8: welding_sparks
  9: hose_reel
  10: person
  11: tarpaulin
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n{'='*60}")
    print(f"Merged dataset v3 ready!")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val:   {len(val_imgs)} images")
    print(f"  YAML:  {yaml_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
