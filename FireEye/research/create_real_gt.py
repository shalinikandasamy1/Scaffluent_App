#!/usr/bin/env python3
"""
Create semi-automatic ground truth labels for real test images.
Uses Grounding DINO with aggressive per-class prompts tuned for
HK construction fire scenes.

Run this AFTER training completes (needs GPU).
Review labels manually for accuracy.
"""

import glob
import os
import torch
from PIL import Image
from collections import Counter


# Tuned prompts for real HK construction fire images
REAL_IMAGE_QUERIES = {
    0: ["fire", "flames", "burning", "blaze"],
    1: ["smoke", "smoke plume", "fumes", "thick smoke"],
    4: ["scaffold net", "safety net", "green net", "debris net", "construction netting"],
    10: ["person", "people", "pedestrian"],
}

CLASS_NAMES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}


def create_gt(images_dir, output_dir, threshold=0.20):
    """Create ground truth labels using Grounding DINO."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    os.makedirs(output_dir, exist_ok=True)

    print("Loading Grounding DINO...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

    images = sorted(
        glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True) +
        glob.glob(os.path.join(images_dir, "**", "*.jpeg"), recursive=True)
    )
    print(f"Labeling {len(images)} real images...")

    stats = Counter()

    for idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        yolo_lines = []

        for class_id, queries in REAL_IMAGE_QUERIES.items():
            text_query = ". ".join(queries) + "."

            inputs = processor(images=img, text=text_query, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=[(h, w)]
            )[0]

            for score, box in zip(results["scores"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                # Skip full-image boxes
                if bw > 0.90 and bh > 0.90:
                    continue

                cx = max(0, min(1, (x1 + x2) / 2 / w))
                cy = max(0, min(1, (y1 + y2) / 2 / h))
                bw = max(0.01, min(1, bw))
                bh = max(0.01, min(1, bh))
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats[CLASS_NAMES[class_id]] += 1

        # Save with sanitized filename
        rel_path = os.path.relpath(img_path, images_dir)
        safe_name = rel_path.replace("/", "_").replace(" ", "_")
        base = os.path.splitext(safe_name)[0]

        with open(os.path.join(output_dir, f"{base}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))

        n = len(yolo_lines)
        print(f"  [{idx+1}/{len(images)}] {os.path.basename(img_path)}: {n} labels")

    print(f"\nTotal: {sum(stats.values())} labels across {len(images)} images")
    for name, count in stats.most_common():
        print(f"  {name}: {count}")

    print(f"\nLabels saved to: {output_dir}")
    print("IMPORTANT: Review these labels manually for accuracy!")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="/home/evnchn/Scaffluent_App/Images dataset/Real/")
    parser.add_argument("--output", default="./real_image_gt")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()

    create_gt(args.images, args.output, threshold=args.threshold)
