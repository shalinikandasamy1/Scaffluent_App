#!/usr/bin/env python3
"""
FireEye YOLO Training Script
=============================
Train YOLO11n on the generated FireEye dataset with proper hyperparameters.

Usage:
    python3 train_yolo_fireeye.py --data ./fireeye_dataset/dataset.yaml --epochs 50
    python3 train_yolo_fireeye.py --data ./fireeye_dataset/dataset.yaml --epochs 100 --resume
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n for FireEye")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--project", type=str, default="./yolo_finetune")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (0 to disable)")
    args = parser.parse_args()

    # Auto-name based on dataset size and epochs
    if args.name is None:
        args.name = f"fireeye_e{args.epochs}_{time.strftime('%m%d_%H%M')}"

    from ultralytics import YOLO

    # Check model path
    model_path = args.model
    if not os.path.exists(model_path):
        # Try common locations
        for candidate in [
            os.path.join(os.path.dirname(__file__), "..", "..", "yolo11n.pt"),
            os.path.expanduser("~/yolo11n.pt"),
            "yolo11n.pt",
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break

    print(f"Training FireEye YOLO detector")
    print(f"  Model: {model_path}")
    print(f"  Data:  {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch: {args.batch}")
    print(f"  Patience: {args.patience}")
    print(f"  Output: {args.project}/{args.name}")
    print()

    model = YOLO(model_path)

    results = model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # Augmentation settings tuned for construction/fire scenes
        hsv_h=0.015,      # hue variation
        hsv_s=0.7,        # saturation variation (fire colors vary a lot)
        hsv_v=0.4,        # value/brightness variation
        degrees=10,       # slight rotation
        translate=0.1,    # translation
        scale=0.5,        # scale variation
        flipud=0.0,       # no vertical flip (construction scenes have gravity)
        fliplr=0.5,       # horizontal flip ok
        mosaic=1.0,       # mosaic augmentation
        mixup=0.1,        # slight mixup
        copy_paste=0.1,   # copy-paste augmentation
        # Training settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,         # final lr = lr0 * lrf
        warmup_epochs=3,
        cos_lr=True,
        # Save settings
        save=True,
        save_period=10,   # save checkpoint every 10 epochs
        plots=True,
        verbose=True,
    )

    print(f"\nTraining complete!")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")

    # Validate on val set
    print("\nRunning validation...")
    val_results = model.val(data=os.path.abspath(args.data))
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

    # Test on real images if available
    real_images_dir = "/home/evnchn/Scaffluent_App/Images dataset/Real"
    if os.path.exists(real_images_dir):
        import glob
        real_imgs = glob.glob(os.path.join(real_images_dir, "*", "*.png"))
        if real_imgs:
            print(f"\nTesting on {len(real_imgs)} real images...")
            for img_path in real_imgs[:10]:  # test first 10
                preds = model.predict(img_path, conf=0.15, verbose=False)
                dets = []
                for r in preds:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = r.names[cls_id]
                        dets.append(f"{cls_name}({conf:.2f})")
                base = os.path.basename(img_path)
                print(f"  {base}: {', '.join(dets) if dets else 'no detections'}")


if __name__ == "__main__":
    main()
