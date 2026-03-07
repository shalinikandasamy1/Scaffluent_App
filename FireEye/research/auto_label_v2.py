#!/usr/bin/env python3
"""
Improved auto-labeling with Grounding DINO v2.
Fixes the tokenization bug by querying each class separately,
and adds NMS to remove duplicate detections.
"""

import glob
import json
import os
import time
from collections import Counter

import torch
from PIL import Image


CLASS_QUERIES = {
    0: ["fire", "flames", "burning"],
    1: ["smoke", "fumes"],
    2: ["extinguisher", "fire extinguisher"],
    3: ["gas cylinder", "gas tank", "gas bottle"],
    4: ["safety net", "scaffold net", "debris net"],
    5: ["exit sign", "emergency exit"],
    6: ["hard hat", "safety helmet", "construction helmet"],
    7: ["safety vest", "high visibility vest", "hi-vis"],
    8: ["sparks", "welding sparks", "grinding sparks"],
    9: ["hose reel", "fire hose"],
    10: ["person", "worker", "man"],
    11: ["tarpaulin", "tarp", "covering sheet"],
}

CLASS_NAMES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}


def nms_boxes(boxes, scores, iou_threshold=0.5):
    """Simple NMS to remove overlapping boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort(descending=True)
    keep = []

    while len(order) > 0:
        i = order[0].item()
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def auto_label_v2(images_dir, labels_dir, threshold=0.20, img_size=512):
    """Auto-label images using per-class Grounding DINO queries."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    os.makedirs(labels_dir, exist_ok=True)

    print("Loading Grounding DINO (per-class query mode)...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Auto-labeling {len(images)} images (threshold={threshold})...")

    stats = Counter()
    t0 = time.time()

    for idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize((img_size, img_size))

        all_boxes = []
        all_scores = []
        all_class_ids = []

        for class_id, queries in CLASS_QUERIES.items():
            # Use the most specific single-word query first for speed
            text_query = ". ".join(queries) + "."

            inputs = processor(images=img, text=text_query, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold,
                text_threshold=threshold, target_sizes=[(img_size, img_size)]
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                all_boxes.append(box)
                all_scores.append(score)
                all_class_ids.append(class_id)

        if all_boxes:
            boxes_t = torch.stack(all_boxes)
            scores_t = torch.stack(all_scores) if isinstance(all_scores[0], torch.Tensor) else torch.tensor(all_scores)

            # NMS per class
            yolo_labels = []
            for cid in set(all_class_ids):
                mask = [i for i, c in enumerate(all_class_ids) if c == cid]
                if not mask:
                    continue
                cls_boxes = boxes_t[mask]
                cls_scores = scores_t[mask]
                keep = nms_boxes(cls_boxes, cls_scores, iou_threshold=0.5)

                for k in keep:
                    box = cls_boxes[k].tolist()
                    x1, y1, x2, y2 = box
                    cx = max(0, min(1, (x1 + x2) / 2 / img_size))
                    cy = max(0, min(1, (y1 + y2) / 2 / img_size))
                    bw = max(0.01, min(1, (x2 - x1) / img_size))
                    bh = max(0.01, min(1, (y2 - y1) / img_size))
                    yolo_labels.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    stats[CLASS_NAMES[cid]] += 1
        else:
            yolo_labels = []

        base = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(labels_dir, f"{base}.txt"), "w") as f:
            f.write("\n".join(yolo_labels))

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(images) - idx - 1) / rate
            print(f"  [{idx+1}/{len(images)}] {sum(stats.values())} labels, {rate:.1f} img/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nLabeled {len(images)} images in {elapsed:.0f}s ({len(images)/elapsed:.2f} img/s)")
    print(f"Total labels: {sum(stats.values())}")
    for name, count in stats.most_common():
        print(f"  {name}: {count}")

    del model
    torch.cuda.empty_cache()
    return dict(stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory with images")
    parser.add_argument("--labels", required=True, help="Output labels directory")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()

    auto_label_v2(args.images, args.labels, threshold=args.threshold)
