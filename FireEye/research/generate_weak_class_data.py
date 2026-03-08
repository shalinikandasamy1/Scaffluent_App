#!/usr/bin/env python3
"""
Generate targeted synthetic data for weak classes identified in Run 2:
- tarpaulin (class 11): 0.05 recall, only 444 labels
- hose_reel (class 9): 0.20 recall, 2218 labels
- welding_sparks (class 8): 0.22 recall, 1475 labels

Uses SDXL-Turbo for generation + Grounding DINO for auto-labeling.
"""

import os
import time
import torch
from PIL import Image
from collections import Counter

# Prompts per class, designed for clear visual features
PROMPTS = {
    11: {  # tarpaulin
        "name": "tarpaulin",
        "queries": ["tarpaulin", "tarp", "covering sheet", "plastic sheeting"],
        "prompts": [
            "blue tarpaulin covering construction scaffolding, construction site",
            "green tarp draped over building materials at work site",
            "large plastic tarpaulin sheet on scaffold, urban construction",
            "orange tarp covering roof during renovation, close up",
            "white tarpaulin hanging from scaffolding structure",
            "heavy duty blue tarp covering construction equipment",
            "torn tarpaulin flapping on metal scaffold frame",
            "waterproof tarp protecting building facade during construction",
            "bright blue plastic sheeting covering construction work area",
            "green tarpaulin tied to scaffolding poles at building site",
            "construction site with large tarp covering concrete structure",
            "yellow tarpaulin sheet draped over wooden pallets at work site",
            "plastic tarp covering open floor of building under construction",
            "brown tarp tied around scaffolding on high rise building",
            "blue construction tarp billowing in wind on scaffold frame",
            "tarpaulin cover on scaffold next to residential building",
            "red tarp covering construction debris on site",
            "translucent plastic sheeting on commercial building scaffold",
            "dirty tarp covering scaffolding at urban construction project",
            "tarpaulin wrapped around scaffold poles, city construction",
            "close up of blue tarp with grommets on construction scaffold",
            "multiple tarps covering scaffold at night, construction site",
            "green mesh tarp on high rise scaffold, building exterior",
            "heavy tarp covering exposed rebar at construction site",
            "blue poly tarp covering scaffold platform",
            "old weathered tarpaulin on abandoned construction scaffold",
            "tarp covering large opening in building wall during renovation",
            "white plastic sheeting on scaffold, residential building",
            "blue tarp flapping in wind on construction scaffold exterior",
            "tarpaulin protecting scaffolding from rain at building site",
        ],
    },
    9: {  # hose_reel
        "name": "hose_reel",
        "queries": ["hose reel", "fire hose", "fire hose reel", "red hose reel"],
        "prompts": [
            "red fire hose reel mounted on wall, industrial building",
            "fire hose reel cabinet on concrete wall, close up",
            "industrial fire hose reel on factory wall",
            "fire hose reel with glass door in building corridor",
            "red fire hose reel station on white wall",
            "wall mounted fire hose reel in parking garage",
            "fire hose reel next to fire extinguisher on wall",
            "close up of red fire hose coiled on wall mount",
            "fire hose reel in stairwell of commercial building",
            "construction site with fire hose reel on temporary wall",
            "fire hose reel cabinet in warehouse setting",
            "red fire hose reel with nozzle visible on wall",
            "fire fighting hose reel mounted in building corridor",
            "fire safety equipment with hose reel on wall",
            "fire hose reel on scaffold at construction site",
            "industrial fire hose reel in factory environment",
            "close up fire hose reel on brick wall",
            "fire hose station with reel in underground parking",
            "red coiled fire hose on wall mounted reel",
            "fire hose reel with instructions sign on wall",
        ],
    },
    8: {  # welding_sparks
        "name": "welding_sparks",
        "queries": ["sparks", "welding sparks", "grinding sparks", "bright sparks"],
        "prompts": [
            "welder creating bright sparks on metal beam, construction site",
            "grinding sparks flying from angle grinder on steel",
            "worker welding steel producing shower of sparks, close up",
            "bright welding sparks at night on construction site scaffold",
            "sparks from metal cutting on construction site",
            "worker with welding torch creating bright sparks on metal frame",
            "shower of orange sparks from grinding metal at work site",
            "welding sparks illuminating dark construction area",
            "close up of welding arc and sparks on steel plate",
            "construction worker grinding metal pipe with sparks flying",
            "welding sparks cascading down from scaffold platform",
            "bright sparks from cutting torch on steel beam",
            "metal sparks flying in industrial workshop",
            "welder creating sparks on structural steel frame",
            "sparks fountain from angle grinder on metal at construction site",
            "night welding with bright sparks on building scaffold",
            "welding sparks reflecting off metal surface at construction site",
            "intense sparks from metal cutting in workshop",
            "worker grinding steel rebar with sparks flying everywhere",
            "welding arc producing bright orange sparks on metal joint",
        ],
    },
}


def generate_and_label(output_dir, num_per_prompt=3, threshold=0.15, img_size=512):
    from diffusers import AutoPipelineForText2Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    os.makedirs(output_dir, exist_ok=True)

    # Load SDXL-Turbo
    print("Loading SDXL-Turbo...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    # Load Grounding DINO
    print("Loading Grounding DINO...")
    gdino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(gdino_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to("cuda")

    stats = Counter()
    total_images = 0
    t0 = time.time()

    for class_id, config in PROMPTS.items():
        class_name = config["name"]
        queries = config["queries"]
        prompts = config["prompts"]

        img_dir = os.path.join(output_dir, f"{class_name}_images")
        lbl_dir = os.path.join(output_dir, f"{class_name}_labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        print(f"\n--- Generating {class_name} (class {class_id}) ---")
        print(f"  {len(prompts)} prompts x {num_per_prompt} = {len(prompts)*num_per_prompt} images")

        for pi, prompt in enumerate(prompts):
            for ni in range(num_per_prompt):
                seed = pi * 1000 + ni + class_id * 100000
                gen = torch.Generator("cuda").manual_seed(seed)
                img = pipe(
                    prompt, num_inference_steps=4, guidance_scale=0.0,
                    height=img_size, width=img_size, generator=gen
                ).images[0]

                fname = f"weak_{class_name}_{pi:03d}_{ni}.jpg"
                img.save(os.path.join(img_dir, fname))

                # Auto-label with Grounding DINO
                text_query = ". ".join(queries) + "."
                inputs = processor(images=img, text=text_query, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = gdino(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=[(img_size, img_size)]
                )[0]

                yolo_lines = []
                for score, box in zip(results["scores"], results["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    bw = (x2 - x1) / img_size
                    bh = (y2 - y1) / img_size
                    # Skip full-image false positives
                    if bw > 0.90 and bh > 0.90:
                        continue
                    # Skip tiny boxes
                    if bw < 0.015 or bh < 0.015:
                        continue
                    cx = max(0, min(1, (x1 + x2) / 2 / img_size))
                    cy = max(0, min(1, (y1 + y2) / 2 / img_size))
                    bw = max(0.01, min(1, bw))
                    bh = max(0.01, min(1, bh))
                    yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    stats[class_name] += 1

                base = os.path.splitext(fname)[0]
                with open(os.path.join(lbl_dir, f"{base}.txt"), "w") as f:
                    f.write("\n".join(yolo_lines))

                total_images += 1

            if (pi + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [{pi+1}/{len(prompts)}] {stats[class_name]} labels, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nGenerated {total_images} images in {elapsed:.0f}s")
    print(f"Labels per class:")
    for name, count in stats.most_common():
        print(f"  {name}: {count}")

    # Cleanup
    del pipe, gdino
    torch.cuda.empty_cache()
    return dict(stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./weak_class_data")
    parser.add_argument("--num-per-prompt", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    generate_and_label(args.output, num_per_prompt=args.num_per_prompt, threshold=args.threshold)
