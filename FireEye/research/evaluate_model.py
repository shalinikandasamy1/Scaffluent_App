#!/usr/bin/env python3
"""
Evaluate a trained FireEye YOLO model on real images.
Produces annotated output images and a detection summary report.
"""

import argparse
import glob
import os
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO
from PIL import Image


FIREEYE_CLASSES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}


def evaluate(model_path, images_dir, output_dir, conf=0.25, iou=0.5, imgsz=512):
    """Run model on all images and save annotated results."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    images = sorted(
        glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True) +
        glob.glob(os.path.join(images_dir, "**", "*.jpeg"), recursive=True)
    )
    print(f"Found {len(images)} images in {images_dir}")

    all_detections = []
    class_counts = Counter()
    per_image = []

    for img_path in images:
        results = model.predict(img_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        result = results[0]

        # Save annotated image
        rel_path = os.path.relpath(img_path, images_dir)
        safe_name = rel_path.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(output_dir, f"det_{safe_name}")
        result.save(out_path)

        # Collect stats
        boxes = result.boxes
        img_dets = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            cls_name = FIREEYE_CLASSES.get(cls_id, f"unknown_{cls_id}")
            class_counts[cls_name] += 1
            img_dets.append({
                "class": cls_name,
                "confidence": round(conf_val, 3),
                "bbox": [round(v, 1) for v in xyxy],
            })

        per_image.append({
            "image": rel_path,
            "num_detections": len(img_dets),
            "detections": img_dets,
        })
        all_detections.extend(img_dets)

        status = ", ".join(f"{d['class']}({d['confidence']:.2f})" for d in img_dets[:5])
        if len(img_dets) > 5:
            status += f" +{len(img_dets)-5} more"
        print(f"  {rel_path}: {len(img_dets)} detections - {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Images: {len(images)}")
    print(f"Total detections: {len(all_detections)}")
    print(f"\nPer-class counts:")
    for name, count in class_counts.most_common():
        avg_conf = sum(d["confidence"] for d in all_detections if d["class"] == name) / count
        print(f"  {name}: {count} (avg conf: {avg_conf:.3f})")

    no_det = sum(1 for p in per_image if p["num_detections"] == 0)
    print(f"\nImages with no detections: {no_det}/{len(images)}")

    # Save JSON report
    report = {
        "model": model_path,
        "images_dir": images_dir,
        "conf_threshold": conf,
        "total_images": len(images),
        "total_detections": len(all_detections),
        "class_counts": dict(class_counts),
        "per_image": per_image,
    }
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
    print(f"Annotated images saved to: {output_dir}")

    return report


def compare_checkpoints(checkpoints_dir, images_dir, output_base, conf=0.25):
    """Compare multiple checkpoints (best, last, epoch_N) on the same images."""
    weights = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pt")))
    print(f"Found {len(weights)} checkpoints: {[os.path.basename(w) for w in weights]}")

    results = {}
    for w in weights:
        name = os.path.splitext(os.path.basename(w))[0]
        out_dir = os.path.join(output_base, name)
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        report = evaluate(w, images_dir, out_dir, conf=conf)
        results[name] = {
            "total_detections": report["total_detections"],
            "class_counts": report["class_counts"],
        }

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"CHECKPOINT COMPARISON")
    print(f"{'='*60}")
    all_classes = set()
    for r in results.values():
        all_classes.update(r["class_counts"].keys())

    header = f"{'Class':<20}" + "".join(f"{name:>12}" for name in results.keys())
    print(header)
    print("-" * len(header))
    for cls in sorted(all_classes):
        row = f"{cls:<20}"
        for name, r in results.items():
            row += f"{r['class_counts'].get(cls, 0):>12}"
        print(row)
    row = f"{'TOTAL':<20}"
    for name, r in results.items():
        row += f"{r['total_detections']:>12}"
    print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--images", required=True, help="Directory with test images")
    parser.add_argument("--output", default="./eval_results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--compare", action="store_true",
                        help="Compare all checkpoints in --model directory")
    args = parser.parse_args()

    if args.compare:
        compare_checkpoints(args.model, args.images, args.output, conf=args.conf)
    else:
        evaluate(args.model, args.images, args.output, conf=args.conf)
