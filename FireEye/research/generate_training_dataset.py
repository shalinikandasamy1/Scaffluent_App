#!/usr/bin/env python3
"""
FireEye Training Dataset Generator
===================================
Generates a comprehensive synthetic training dataset for fire safety
object detection using SDXL-Turbo on GPU.

Produces:
  1. Text-to-image scenes across 5 risk levels (safe/low/medium/high/critical)
  2. Equipment close-ups for key detection targets
  3. Img2img severity variations from real photos
  4. Auto-labels using Grounding DINO
  5. Train/val split with YOLO dataset.yaml

Usage:
    python3 generate_training_dataset.py --count 2000 --output ./dataset
    python3 generate_training_dataset.py --count 500 --skip-autolabel  # faster
    python3 generate_training_dataset.py --severity-only  # only img2img from real photos

Requirements:
    pip install diffusers transformers accelerate torch Pillow groundingdino-py

Hardware: RTX 3060 12GB (tested), RTX 3070/4060+ recommended for larger batches
"""

import argparse
import glob
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

# ── Prompt Database ──────────────────────────────────────────────────

SCENE_PREFIXES = [
    "A photorealistic photograph of a",
    "Professional construction site photo showing",
    "Realistic image of a",
    "High quality photo of a",
    "DSLR photograph of a",
]

PROMPTS = {
    "safe": [
        "well-organized construction site with green fire-retardant scaffold nets, fire extinguisher on wall, clear exit signs, workers in hard hats and safety vests, daytime",
        "construction site interior with fire hose reel mounted on wall, green emergency exit sign illuminated, clean concrete floor, organized steel beams, bright lighting",
        "construction site stairwell with fire extinguisher at landing, emergency lighting, clear unobstructed path, exit signs visible, clean and well-maintained",
        "scaffolding with intact green safety netting on Hong Kong building exterior, bamboo scaffolding, workers visible, blue sky, safe conditions",
        "construction site entrance with safety signage, fire assembly point sign, extinguisher cabinet, security barrier, well-lit area",
        "indoor construction area with fire alarm panel on wall, hose reel, emergency exit illuminated, no hazards visible, organized workplace",
        "construction site with properly stored gas cylinders in caged area, no smoking sign, safety equipment rack, organized tools",
        "completed floor of high-rise construction, fire extinguisher near staircase, exit route marked with arrows, clean workspace",
        "construction site break area with fire safety poster on wall, fire extinguisher, first aid box, organized clean space",
        "scaffolding platform with safety rail, green debris net, worker in full PPE inspecting, clean orderly site",
    ],
    "low": [
        "construction worker performing controlled welding behind protective screen, fire extinguisher nearby, sparks contained, PPE worn, organized site",
        "construction site with hot work in progress, welding curtain deployed, fire watch person standing nearby, hot work permit visible on wall",
        "mild welding sparks behind a protective barrier at construction site, fire extinguisher within reach, worker in full PPE with face shield",
        "construction worker using gas torch for pipe work, gas cylinders stored upright with chain, no smoking sign, ventilated area",
        "small controlled flame from cutting torch at construction site, fire extinguisher and water bucket nearby, designated hot work area",
        "worker grinding metal with angle grinder at construction site, spark guard in place, fire blanket nearby, safety vest and hard hat worn",
        "construction site where welding screen separates hot work from stored materials, fire watch personnel visible, orderly setup",
        "portable welding station at construction site with proper ventilation, gas cylinders chained, fire extinguisher within 2 meters",
    ],
    "medium": [
        "construction site with wooden planks and debris scattered near welding area, some smoke visible, scaffold nets partially torn, messy workspace",
        "welding sparks landing on wooden formwork at construction site, no visible fire screen, combustible materials within 3 meters",
        "construction site with gas cylinders standing without proper restraint near work area, some loose cables, no safety signage visible",
        "scaffold netting partially damaged, some construction materials piled against exit route, dimly lit corridor",
        "construction site interior with overloaded electrical cables, temporary wiring exposed, multiple extension leads, dusty environment",
        "cigarette butts on ground near timber stack at construction site, no designated smoking area, workers without full PPE",
        "blocked emergency exit at construction site, materials stacked against door, exit sign partially obscured",
        "construction site with combustible tarpaulin covering scaffold, no fire retardant markings, near electrical work area",
        "construction site with wood shavings and sawdust near electrical panel, untidy workspace, poor housekeeping",
        "painting supplies and solvent cans left open near welding area at construction site, potential ignition risk",
    ],
    "high": [
        "small fire starting on scaffold netting at construction site, orange flames visible, smoke rising, workers nearby, Hong Kong building",
        "uncontrolled welding sparks igniting wood debris at construction site, no fire screen, fire extinguisher not visible, smoke starting",
        "gas cylinders near open flame at construction site, no safety barriers, high risk situation, industrial setting",
        "fire spreading along combustible facade covering on scaffolding, flames climbing upward, construction site exterior, smoke billowing",
        "electrical panel sparking at construction site, exposed wires, overheated cable, wisps of smoke, burning smell",
        "construction site with fire burning in debris pile, no firefighting equipment visible, blocked escape route, dangerous conditions",
        "scaffold net on fire, flames spreading vertically on building facade, Hong Kong high-rise construction, black smoke",
        "hot work area with fire spreading to nearby combustibles, no screening, gas cylinders visible nearby, emergency situation",
    ],
    "critical": [
        "major fire engulfing scaffold netting on high-rise building, massive flames and thick black smoke, construction site disaster, Hong Kong",
        "construction site fully engulfed in flames, fire spreading across multiple floors via facade covering, firefighters responding, smoke column",
        "gas cylinder explosion risk at construction site fire, flames near oxygen and acetylene tanks, critical danger, emergency evacuation",
        "building facade completely ablaze, scaffold nets acting as fuel, fire spreading rapidly upward, massive smoke plume, Hong Kong cityscape",
        "multi-floor construction site fire, flames visible from windows, melted scaffold netting, structural damage, emergency response",
    ],
}

EQUIPMENT_PROMPTS = {
    "fire_ext": [
        "red fire extinguisher mounted on concrete wall at construction site, clear view, well-lit, ABC dry powder type, pressure gauge visible",
        "fire extinguisher in red cabinet behind glass on construction site wall, break glass sign, well maintained, close-up",
        "portable fire extinguisher on floor near scaffolding at construction site, red cylinder, safety pin, nozzle visible",
        "CO2 fire extinguisher with black horn nozzle at construction site, silver cylinder, wall mounted",
    ],
    "gas_cyl": [
        "green oxygen cylinder and red acetylene cylinder chained upright in metal cage at construction site, pressure regulators, warning labels",
        "row of industrial gas cylinders stored at construction site, various colors, valve caps on, proper storage rack",
        "propane gas cylinder connected to torch at construction site, regulator and hose visible, yellow cylinder",
        "oxygen and acetylene welding gas cylinders on trolley at construction site, flashback arrestors visible",
    ],
    "exit_sign": [
        "green illuminated emergency exit sign above doorway at construction site, running man symbol, arrow pointing right",
        "emergency exit sign in dark construction corridor, battery backup light, clear escape direction",
        "exit sign with directional arrow at construction site stairwell, green and white, ISO standard",
    ],
    "hose_reel": [
        "fire hose reel mounted on wall at construction site, red wheel, connected to water supply, coiled hose",
        "dry riser inlet at construction site exterior, red connection point, fire brigade inlet sign",
        "fire hose cabinet open showing coiled hose and nozzle, construction site corridor",
    ],
    "scaffold_net": [
        "green fire-retardant safety net on bamboo scaffolding, close-up texture, Hong Kong style construction",
        "white debris netting on metal scaffolding, fire retardant label visible, construction site exterior",
        "orange scaffold safety net stretched between poles, construction site, close-up mesh detail",
    ],
    "hard_hat": [
        "yellow hard hat on construction worker at building site, safety vest, proper PPE, professional photo",
        "white safety helmet on table at construction site, chin strap visible, safety sticker",
        "construction workers group wearing different colored hard hats, safety meeting, construction site",
    ],
    "welding": [
        "welding sparks shower from angle grinder at construction site, bright orange sparks, protective face shield, close-up",
        "TIG welding arc bright blue-white light at construction site, welding mask, steel beam, sparks",
        "MIG welding at construction site, wire feed visible, bright arc, smoke, protective curtain behind",
    ],
}

# Severity variation prompts for img2img
SEVERITY_PROMPTS = {
    "calm": ("construction site, scaffolding, building under construction, peaceful, no fire, no smoke, clear day, Hong Kong", 0.55),
    "early_smoke": ("construction site with wisps of smoke, early fire signs, light haze, scaffolding, building", 0.35),
    "escalated": ("building on fire, flames spreading, scaffold netting burning, thick smoke, construction site fire", 0.55),
    "aftermath": ("building with fire damage, charred facade, broken windows, smoke residue, construction site aftermath", 0.50),
    "night": ("construction site at night, building with scaffolding, dark sky, artificial lighting, urban", 0.45),
}


# ── Class Definitions ────────────────────────────────────────────────

CLASS_MAP = {
    "fire": 0, "flames": 0, "fire flames": 0,
    "smoke": 1,
    "fire extinguisher": 2,
    "gas cylinder": 3,
    "scaffold net": 4,
    "exit sign": 5,
    "hard hat": 6,
    "safety vest": 7,
    "welding sparks": 8, "sparks": 8,
    "hose reel": 9,
    "person": 10, "construction worker": 10,
    "tarpaulin": 11,
}

CLASS_NAMES = {
    0: "fire", 1: "smoke", 2: "fire_extinguisher", 3: "gas_cylinder",
    4: "scaffold_net", 5: "exit_sign", 6: "hard_hat", 7: "safety_vest",
    8: "welding_sparks", 9: "hose_reel", 10: "person", 11: "tarpaulin",
}


def generate_text2image(pipe, output_dir, count=500, seed=42):
    """Generate text-to-image scenes across risk levels."""
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Build prompt list with distribution
    # More safe/low images to address the severity gap
    distribution = {
        "safe": 0.25, "low": 0.20, "medium": 0.20,
        "high": 0.20, "critical": 0.15,
    }

    all_prompts = []
    for category, fraction in distribution.items():
        n = int(count * fraction * 0.7)  # 70% scenes, 30% equipment
        prompts = PROMPTS[category]
        for i in range(n):
            prefix = rng.choice(SCENE_PREFIXES)
            prompt = rng.choice(prompts)
            all_prompts.append((category, f"{prefix} {prompt}", rng.randint(0, 99999)))

    # Equipment close-ups (30% of count)
    n_equip = int(count * 0.3)
    equip_keys = list(EQUIPMENT_PROMPTS.keys())
    for i in range(n_equip):
        key = rng.choice(equip_keys)
        prefix = rng.choice(SCENE_PREFIXES)
        prompt = rng.choice(EQUIPMENT_PROMPTS[key])
        all_prompts.append((f"equip_{key}", f"{prefix} {prompt}", rng.randint(0, 99999)))

    rng.shuffle(all_prompts)
    metadata = []

    print(f"Generating {len(all_prompts)} text-to-image scenes...")
    cats = Counter(c for c, _, _ in all_prompts)
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")

    t0 = time.time()
    for idx, (category, prompt, s) in enumerate(all_prompts):
        gen = torch.Generator("cuda").manual_seed(s)
        image = pipe(prompt=prompt, num_inference_steps=4,
                     guidance_scale=0.0, generator=gen).images[0]
        fname = f"{category}_{idx:04d}.jpg"
        image.save(os.path.join(output_dir, fname), quality=90)
        metadata.append({"file": fname, "category": category,
                         "prompt": prompt, "seed": s})

        if (idx + 1) % 50 == 0:
            rate = (idx + 1) / (time.time() - t0)
            eta = (len(all_prompts) - idx - 1) / rate
            print(f"  [{idx+1}/{len(all_prompts)}] {rate:.1f} img/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Generated {len(all_prompts)} images in {elapsed:.0f}s ({len(all_prompts)/elapsed:.1f} img/s)")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def generate_severity_variations(pipe_i2i, real_images_dir, output_dir, seed=42):
    """Generate severity variations from real photos using img2img."""
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Find all real images
    patterns = [
        os.path.join(real_images_dir, "Real", "TKO*", "*.png"),
        os.path.join(real_images_dir, "Real", "Central*", "*.png"),
    ]
    real_files = []
    for pat in patterns:
        real_files.extend(sorted(glob.glob(pat)))

    if not real_files:
        print("No real images found, skipping severity variations")
        return []

    print(f"Generating severity variations for {len(real_files)} real images...")
    metadata = []

    for img_idx, img_path in enumerate(real_files):
        loc = "tko" if "TKO" in img_path else "central"
        img = Image.open(img_path).convert("RGB").resize((512, 512))

        for sev_name, (prompt, strength) in SEVERITY_PROMPTS.items():
            gen = torch.Generator("cuda").manual_seed(rng.randint(0, 99999))
            result = pipe_i2i(prompt=prompt, image=img, strength=strength,
                              num_inference_steps=4, guidance_scale=0.0,
                              generator=gen).images[0]
            fname = f"{loc}_{img_idx+1:02d}_{sev_name}.jpg"
            result.save(os.path.join(output_dir, fname), quality=90)
            metadata.append({"file": fname, "source": os.path.basename(img_path),
                             "severity": sev_name, "strength": strength})

        if (img_idx + 1) % 10 == 0:
            print(f"  [{img_idx+1}/{len(real_files)}] images processed")

    print(f"Generated {len(metadata)} severity variations")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def auto_label(images_dir, labels_dir, threshold=0.20):
    """Auto-label images using Grounding DINO."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    os.makedirs(labels_dir, exist_ok=True)

    print("Loading Grounding DINO for auto-labeling...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")

    text_query = "fire. smoke. fire extinguisher. gas cylinder. scaffold net. exit sign. hard hat. safety vest. welding sparks. hose reel. person. tarpaulin."

    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Auto-labeling {len(images)} images (threshold={threshold})...")

    stats = Counter()
    t0 = time.time()

    for idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        inputs = processor(images=img, text=text_query, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=threshold,
            text_threshold=threshold, target_sizes=[(512, 512)]
        )[0]

        yolo_labels = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_clean = label.strip().lower()
            class_id = None
            for key, cid in CLASS_MAP.items():
                if key in label_clean:
                    class_id = cid
                    break
            if class_id is None:
                continue

            x1, y1, x2, y2 = box.tolist()
            cx = max(0, min(1, (x1 + x2) / 2 / 512))
            cy = max(0, min(1, (y1 + y2) / 2 / 512))
            bw = max(0.01, min(1, (x2 - x1) / 512))
            bh = max(0.01, min(1, (y2 - y1) / 512))
            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats[CLASS_NAMES[class_id]] += 1

        base = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(labels_dir, f"{base}.txt"), "w") as f:
            f.write("\n".join(yolo_labels))

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(images)}] {sum(stats.values())} labels")

    elapsed = time.time() - t0
    print(f"Labeled {len(images)} images in {elapsed:.0f}s, {sum(stats.values())} total labels")
    for name, count in stats.most_common():
        print(f"  {name}: {count}")

    del model
    torch.cuda.empty_cache()
    return dict(stats)


def create_dataset_split(images_dir, labels_dir, output_dir, val_fraction=0.2, seed=42):
    """Create train/val split and write dataset.yaml."""
    rng = random.Random(seed)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    images = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    rng.shuffle(images)
    split_idx = int(len(images) * (1 - val_fraction))
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]

    for img_list, split in [(train_imgs, "train"), (val_imgs, "val")]:
        for fname in img_list:
            base = os.path.splitext(fname)[0]
            src_img = os.path.join(images_dir, fname)
            dst_img = os.path.join(output_dir, split, "images", fname)
            if not os.path.exists(dst_img):
                os.symlink(os.path.realpath(src_img), dst_img)

            src_lbl = os.path.join(labels_dir, f"{base}.txt")
            dst_lbl = os.path.join(output_dir, split, "labels", f"{base}.txt")
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                shutil.copy2(src_lbl, dst_lbl)

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_content = f"""# FireEye Training Dataset
# Generated: {time.strftime('%Y-%m-%d %H:%M')}
# Train: {len(train_imgs)}, Val: {len(val_imgs)}

path: {os.path.abspath(output_dir)}
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
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset split: {len(train_imgs)} train, {len(val_imgs)} val")
    print(f"YAML: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="FireEye Training Dataset Generator")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of text-to-image scenes (default: 500)")
    parser.add_argument("--output", type=str, default="./fireeye_dataset",
                        help="Output directory")
    parser.add_argument("--real-images", type=str,
                        default="/home/evnchn/Scaffluent_App/Images dataset",
                        help="Directory containing real images for severity variations")
    parser.add_argument("--skip-autolabel", action="store_true",
                        help="Skip auto-labeling (faster, labels must be added manually)")
    parser.add_argument("--severity-only", action="store_true",
                        help="Only generate severity variations from real photos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="Grounding DINO confidence threshold for auto-labeling")

    args = parser.parse_args()
    output = args.output
    os.makedirs(output, exist_ok=True)

    print(f"FireEye Dataset Generator")
    print(f"Output: {output}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # Load SDXL-Turbo
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

    if not args.severity_only:
        print("Loading SDXL-Turbo (text-to-image)...")
        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        pipe_t2i = pipe_t2i.to("cuda")
        pipe_t2i.enable_attention_slicing()

        scenes_dir = os.path.join(output, "scenes")
        generate_text2image(pipe_t2i, scenes_dir, count=args.count, seed=args.seed)
        del pipe_t2i
        torch.cuda.empty_cache()

    # Load img2img variant
    print("\nLoading SDXL-Turbo (image-to-image)...")
    pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe_i2i = pipe_i2i.to("cuda")
    pipe_i2i.enable_attention_slicing()

    variations_dir = os.path.join(output, "variations")
    generate_severity_variations(pipe_i2i, args.real_images, variations_dir, seed=args.seed)
    del pipe_i2i
    torch.cuda.empty_cache()

    if args.severity_only:
        print("\nSeverity-only mode. Done.")
        return

    # Merge all images into one directory
    all_images_dir = os.path.join(output, "all_images")
    os.makedirs(all_images_dir, exist_ok=True)

    for src_dir in [os.path.join(output, "scenes"), os.path.join(output, "variations")]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                if f.endswith(".jpg"):
                    src = os.path.join(src_dir, f)
                    dst = os.path.join(all_images_dir, f)
                    if not os.path.exists(dst):
                        os.symlink(os.path.realpath(src), dst)

    total = len([f for f in os.listdir(all_images_dir) if f.endswith(".jpg")])
    print(f"\nTotal images: {total}")

    # Auto-label
    labels_dir = os.path.join(output, "all_labels")
    if not args.skip_autolabel:
        auto_label(all_images_dir, labels_dir, threshold=args.threshold)
    else:
        os.makedirs(labels_dir, exist_ok=True)
        print("Skipping auto-labeling (--skip-autolabel)")

    # Create train/val split
    print("\nCreating train/val split...")
    yaml_path = create_dataset_split(all_images_dir, labels_dir, output, seed=args.seed)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"Output: {output}")
    print(f"YAML:   {yaml_path}")
    print(f"Train with: yolo train data={yaml_path} model=yolo11n.pt epochs=50 imgsz=512")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
