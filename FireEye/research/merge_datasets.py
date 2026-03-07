#!/usr/bin/env python3
"""
Merge multiple datasets into a unified FireEye training dataset.

Combines:
1. Our synthetic SDXL-Turbo generated images (with Grounding DINO labels)
2. Ultralytics Construction-PPE dataset (remapped classes)
3. Real welding frames (with auto-labels)
4. D-Fire dataset (if available)

Class mapping to our 12 FireEye classes:
  0: fire, 1: smoke, 2: fire_extinguisher, 3: gas_cylinder,
  4: scaffold_net, 5: exit_sign, 6: hard_hat, 7: safety_vest,
  8: welding_sparks, 9: hose_reel, 10: person, 11: tarpaulin
"""

import os
import random
import shutil
import glob
from collections import Counter


FIREEYE_CLASSES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}

# Construction-PPE class mapping → FireEye classes
# PPE: 0=helmet, 1=gloves, 2=vest, 3=boots, 4=goggles, 5=none,
#      6=Person, 7=no_helmet, 8=no_goggle, 9=no_gloves, 10=no_boots
PPE_TO_FIREEYE = {
    0: 6,     # helmet → hard_hat
    2: 7,     # vest → safety_vest
    6: 10,    # Person → person
    # 7: 6,   # no_helmet could map to hard_hat negative — skip for now
}

# D-Fire class mapping → FireEye classes
# D-Fire: 0=fire, 1=smoke
DFIRE_TO_FIREEYE = {
    0: 0,  # fire → fire
    1: 1,  # smoke → smoke
}


def remap_labels(src_label, class_map, dst_label):
    """Read a YOLO label file, remap classes, write to dst."""
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
                class_map=None, prefix="", max_images=None):
    """Add a dataset to the merged output."""
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                    glob.glob(os.path.join(img_dir, "*.png")))

    if max_images and len(images) > max_images:
        random.shuffle(images)
        images = images[:max_images]

    count = 0
    labels_count = 0
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        dst_name = f"{prefix}{base}" if prefix else base

        # Copy image
        dst_img = os.path.join(out_img_dir, f"{dst_name}{ext}")
        if not os.path.exists(dst_img):
            shutil.copy2(img_path, dst_img)

        # Copy/remap labels
        src_lbl = os.path.join(lbl_dir, f"{base}.txt")
        dst_lbl = os.path.join(out_lbl_dir, f"{dst_name}.txt")

        if class_map is not None:
            n = remap_labels(src_lbl, class_map, dst_lbl)
            labels_count += n
        elif os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            with open(src_lbl) as f:
                labels_count += sum(1 for line in f if line.strip())
        else:
            # Create empty label file
            open(dst_lbl, 'w').close()

        count += 1

    print(f"  {name}: {count} images, {labels_count} labels")
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/evnchn/Scaffluent_App/FireEye/research/merged_dataset")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = args.output

    # Temp staging area
    staging_img = os.path.join(out, "_staging", "images")
    staging_lbl = os.path.join(out, "_staging", "labels")
    os.makedirs(staging_img, exist_ok=True)
    os.makedirs(staging_lbl, exist_ok=True)

    total = 0
    print("Merging datasets into FireEye unified training set...\n")

    # 1. Our synthetic dataset
    synth_img = "/home/evnchn/Scaffluent_App/FireEye/research/fireeye_dataset/all_images"
    synth_lbl = "/home/evnchn/Scaffluent_App/FireEye/research/fireeye_dataset/all_labels"
    if os.path.exists(synth_img):
        total += add_dataset("Synthetic (SDXL-Turbo)", synth_img, synth_lbl,
                             staging_img, staging_lbl, prefix="syn_")

    # 2. Construction-PPE dataset (remapped)
    ppe_base = "/home/evnchn/datasets/construction-ppe"
    for split in ["train", "val", "test"]:
        ppe_img = os.path.join(ppe_base, "images", split)
        ppe_lbl = os.path.join(ppe_base, "labels", split)
        if os.path.exists(ppe_img):
            total += add_dataset(f"Construction-PPE ({split})", ppe_img, ppe_lbl,
                                 staging_img, staging_lbl,
                                 class_map=PPE_TO_FIREEYE, prefix=f"ppe_{split}_")

    # 3. Welding frames (with auto-labels if available)
    weld_img = "/home/evnchn/Scaffluent_App/FireEye/research/welding_frames"
    weld_lbl = "/home/evnchn/Scaffluent_App/FireEye/research/welding_labels"
    if os.path.exists(weld_img):
        total += add_dataset("Welding frames", weld_img,
                             weld_lbl if os.path.exists(weld_lbl) else weld_img,
                             staging_img, staging_lbl, prefix="weld_")

    # 4. D-Fire dataset (if downloaded)
    dfire_base = "/home/evnchn/Scaffluent_App/FireEye/research/DFireDataset"
    for sub in ["train/images", "images"]:
        dfire_img = os.path.join(dfire_base, sub)
        dfire_lbl = dfire_img.replace("images", "labels")
        if os.path.exists(dfire_img) and any(
            f.endswith(('.jpg', '.png')) for f in os.listdir(dfire_img)
        ):
            total += add_dataset("D-Fire", dfire_img, dfire_lbl,
                                 staging_img, staging_lbl,
                                 class_map=DFIRE_TO_FIREEYE, prefix="dfire_",
                                 max_images=5000)
            break

    # 5. Earlier synthetic/augmented samples
    for subdir, name in [
        ("synthetic_bulk", "Synthetic bulk"),
        ("augmented_samples", "Augmented"),
        ("severity_variations", "Severity variations"),
    ]:
        d = f"/home/evnchn/Scaffluent_App/FireEye/research/{subdir}"
        if os.path.exists(d):
            # These don't have labels dir separate
            total += add_dataset(name, d, d, staging_img, staging_lbl,
                                 prefix=f"{subdir[:4]}_")

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
            # Symlink images
            src = os.path.join(staging_img, fname)
            dst = os.path.join(split_img, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(src), dst)
            # Copy labels
            src_lbl = os.path.join(staging_lbl, f"{base}.txt")
            dst_lbl = os.path.join(split_lbl, f"{base}.txt")
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    # Count class distribution
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

    # Write dataset.yaml
    yaml_path = os.path.join(out, "dataset.yaml")
    import time
    yaml_content = f"""# FireEye Merged Training Dataset
# Generated: {time.strftime('%Y-%m-%d %H:%M')}
# Train: {len(train_imgs)}, Val: {len(val_imgs)}
# Sources: Synthetic SDXL-Turbo, Construction-PPE, Welding frames

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
    print(f"Merged dataset ready!")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val:   {len(val_imgs)} images")
    print(f"  YAML:  {yaml_path}")
    print(f"  Train: yolo train data={yaml_path} model=yolo11n.pt epochs=50 imgsz=512")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
