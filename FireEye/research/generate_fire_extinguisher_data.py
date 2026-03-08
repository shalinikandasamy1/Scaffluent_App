#!/usr/bin/env python3
"""
Targeted synthetic data generation for fire_extinguisher class.
This class had ZERO labels in the merged dataset, so we generate
focused images and use Grounding DINO with aggressive prompts.
"""

import os
import glob
import time
import torch
from PIL import Image
from collections import Counter


# Fire extinguisher focused prompts
PROMPTS = [
    # Wall-mounted extinguishers
    "red fire extinguisher mounted on white wall in office building, close up",
    "fire extinguisher in glass cabinet on wall, hallway, indoor",
    "two red fire extinguishers on wall brackets in warehouse",
    "fire extinguisher with sign above it on concrete wall",
    "red dry powder fire extinguisher hanging on wall mount, industrial",

    # Construction site extinguishers
    "red fire extinguisher on construction site floor near scaffolding",
    "fire extinguisher cabinet next to building entrance, construction zone",
    "portable fire extinguisher on ground near building materials",
    "fire extinguisher station on construction site with safety signs",
    "red fire extinguisher beside gas cylinders on construction site",

    # Multiple extinguishers / with other objects
    "row of fire extinguishers lined up against wall in parking garage",
    "fire extinguisher next to exit sign in corridor",
    "fire safety equipment including fire extinguisher and hose reel on wall",
    "fire extinguisher and first aid box on wall, factory floor",
    "fire extinguisher rack with three extinguishers, building lobby",

    # Different types
    "CO2 fire extinguisher silver cylinder on stand, indoor",
    "foam fire extinguisher cream colored on wall bracket",
    "large wheeled fire extinguisher on trolley, industrial",
    "small fire extinguisher under office desk",
    "ABC dry chemical fire extinguisher red, close up photo",

    # In use / with people
    "worker in hard hat checking fire extinguisher on wall",
    "fire safety inspection, person examining fire extinguisher",
    "fire drill training with fire extinguisher demonstration",
    "firefighter holding fire extinguisher in building",
    "safety officer pointing at fire extinguisher during training",

    # Varied backgrounds
    "fire extinguisher in kitchen near exit door",
    "fire extinguisher on pillar in underground car park",
    "fire extinguisher in school hallway with lockers",
    "fire extinguisher mounted in elevator lobby",
    "fire extinguisher on ship deck, maritime safety",

    # Additional variety
    "expired fire extinguisher with red tag on wall",
    "new fire extinguisher still in plastic wrap",
    "fire extinguisher inside protective box, outdoor",
    "mini fire extinguisher for car, red cylinder",
    "fire extinguisher recessed in wall niche",
    "fire extinguisher next to fire alarm pull station",
    "fire extinguisher on floor in empty room",
    "fire extinguisher behind glass break panel",
    "vintage fire extinguisher brass colored on display",
    "industrial fire extinguisher with pressure gauge close up",
]


def generate_images(output_dir, num_per_prompt=3, img_size=512):
    """Generate fire extinguisher images with SDXL-Turbo."""
    from diffusers import AutoPipelineForText2Image

    os.makedirs(output_dir, exist_ok=True)

    print("Loading SDXL-Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    count = 0
    for prompt in PROMPTS:
        for i in range(num_per_prompt):
            seed = hash(prompt + str(i)) % (2**32)
            generator = torch.Generator("cuda").manual_seed(seed)

            image = pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

            image = image.resize((img_size, img_size))
            fname = f"extinguisher_{count:04d}.jpg"
            image.save(os.path.join(output_dir, fname), quality=90)
            count += 1

            if count % 20 == 0:
                print(f"  Generated {count} images...")

    print(f"Generated {count} images in {output_dir}")
    del pipe
    torch.cuda.empty_cache()
    return count


def auto_label_extinguishers(images_dir, labels_dir, threshold=0.15):
    """Label fire extinguisher images with Grounding DINO, lower threshold for recall."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    os.makedirs(labels_dir, exist_ok=True)

    print("Loading Grounding DINO for fire extinguisher labeling...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

    # Multiple query strategies for fire extinguishers
    queries = [
        "fire extinguisher.",
        "red cylinder. extinguisher.",
        "fire safety equipment. extinguisher.",
    ]

    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Labeling {len(images)} images...")

    stats = {"labeled": 0, "empty": 0, "total_boxes": 0}

    for idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        best_boxes = []
        best_scores = []

        for query in queries:
            inputs = processor(images=img, text=query, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=[(h, w)]
            )[0]

            for score, box in zip(results["scores"], results["boxes"]):
                best_boxes.append(box.tolist())
                best_scores.append(score.item())

        # Simple NMS by removing highly overlapping boxes
        yolo_lines = []
        used = set()
        scored = sorted(range(len(best_scores)), key=lambda i: -best_scores[i])

        for i in scored:
            if i in used:
                continue
            x1, y1, x2, y2 = best_boxes[i]
            # Mark overlapping as used
            for j in scored:
                if j != i and j not in used:
                    jx1, jy1, jx2, jy2 = best_boxes[j]
                    inter_x1 = max(x1, jx1)
                    inter_y1 = max(y1, jy1)
                    inter_x2 = min(x2, jx2)
                    inter_y2 = min(y2, jy2)
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area_i = (x2 - x1) * (y2 - y1)
                        area_j = (jx2 - jx1) * (jy2 - jy1)
                        iou = inter / (area_i + area_j - inter)
                        if iou > 0.5:
                            used.add(j)

            # Convert to YOLO format — class 2 (fire_extinguisher)
            cx = max(0, min(1, (x1 + x2) / 2 / w))
            cy = max(0, min(1, (y1 + y2) / 2 / h))
            bw = max(0.01, min(1, (x2 - x1) / w))
            bh = max(0.01, min(1, (y2 - y1) / h))
            yolo_lines.append(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        base = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(labels_dir, f"{base}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))

        if yolo_lines:
            stats["labeled"] += 1
            stats["total_boxes"] += len(yolo_lines)
        else:
            stats["empty"] += 1

        if (idx + 1) % 30 == 0:
            print(f"  [{idx+1}/{len(images)}] {stats['total_boxes']} boxes found")

    print(f"\nLabeling complete:")
    print(f"  Images with detections: {stats['labeled']}")
    print(f"  Empty images: {stats['empty']}")
    print(f"  Total boxes: {stats['total_boxes']}")

    del model
    torch.cuda.empty_cache()
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./extinguisher_data")
    parser.add_argument("--num-per-prompt", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation, only label")
    args = parser.parse_args()

    img_dir = os.path.join(args.output, "images")
    lbl_dir = os.path.join(args.output, "labels")

    if not args.skip_gen:
        t0 = time.time()
        n = generate_images(img_dir, num_per_prompt=args.num_per_prompt)
        print(f"Generation took {time.time()-t0:.0f}s for {n} images")

    t0 = time.time()
    stats = auto_label_extinguishers(img_dir, lbl_dir, threshold=args.threshold)
    print(f"Labeling took {time.time()-t0:.0f}s")
