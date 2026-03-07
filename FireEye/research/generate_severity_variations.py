#!/usr/bin/env python3
"""Generate severity variations of real construction site photos using SDXL-Turbo img2img."""

import glob
import json
import os
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# === Configuration ===
OUTPUT_DIR = "/home/evnchn/Scaffluent_App/FireEye/research/severity_variations"
IMAGE_BASE = "/home/evnchn/Scaffluent_App/Images dataset/Real"
IMG_SIZE = (512, 512)
NUM_STEPS = 4
GUIDANCE_SCALE = 0.0

SEVERITY_CONFIGS = {
    "calm": {
        "strength": 0.6,
        "prompt": "A peaceful construction site with scaffolding, clear sky, no fire, no smoke, safe working conditions, bright daylight, clean environment",
    },
    "early_smoke": {
        "strength": 0.35,
        "prompt": "Construction site scaffolding with wisps of smoke rising, early signs of fire, thin gray smoke drifting, hazy atmosphere, warning signs of danger",
    },
    "escalated": {
        "strength": 0.5,
        "prompt": "Construction site scaffolding engulfed in intense fire, large orange flames, thick black smoke billowing, fire spreading rapidly, dangerous inferno",
    },
    "aftermath": {
        "strength": 0.55,
        "prompt": "Construction site after a fire, charred black scaffolding, burnt debris, ash covered ground, smoke residue, fire damage aftermath, destroyed structure",
    },
    "night": {
        "strength": 0.45,
        "prompt": "Construction site scaffolding at night, dark sky, dim artificial lighting, nighttime scene, moonlight, shadows, same construction structures visible at night",
    },
}


def find_all_images():
    """Find all 29 real construction site images using glob to handle Unicode filenames."""
    tko_images = sorted(glob.glob(os.path.join(IMAGE_BASE, "TKO*", "*.png")))
    central_images = sorted(glob.glob(os.path.join(IMAGE_BASE, "Central*", "*.png")))
    print(f"Found {len(tko_images)} TKO images, {len(central_images)} Central images")
    print(f"Total: {len(tko_images) + len(central_images)} images")
    return tko_images, central_images


def make_output_name(location: str, img_idx: int, severity: str) -> str:
    return f"{location}_{img_idx:02d}_{severity}.jpg"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find images
    tko_images, central_images = find_all_images()
    all_images = [(p, "tko", i) for i, p in enumerate(tko_images, 1)] + [
        (p, "central", i) for i, p in enumerate(central_images, 1)
    ]
    total = len(all_images)
    assert total == 29, f"Expected 29 images, found {total}"

    # Load SDXL-Turbo
    print("\nLoading SDXL-Turbo pipeline...")
    t0 = time.time()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    metadata = {
        "model": "stabilityai/sdxl-turbo",
        "num_inference_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "image_size": list(IMG_SIZE),
        "total_source_images": total,
        "severity_configs": {
            k: {"strength": v["strength"], "prompt": v["prompt"]}
            for k, v in SEVERITY_CONFIGS.items()
        },
        "images": [],
    }

    counts = {sev: 0 for sev in SEVERITY_CONFIGS}
    gen_start = time.time()

    for idx, (img_path, location, img_num) in enumerate(all_images, 1):
        print(f"\n[{idx}/{total}] Processing {os.path.basename(img_path)} ({location} #{img_num})")

        # Load and resize
        src_img = Image.open(img_path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)

        for severity, config in SEVERITY_CONFIGS.items():
            out_name = make_output_name(location, img_num, severity)
            out_path = os.path.join(OUTPUT_DIR, out_name)

            result = pipe(
                prompt=config["prompt"],
                image=src_img,
                strength=config["strength"],
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )
            result.images[0].save(out_path, "JPEG", quality=90)
            counts[severity] += 1

            metadata["images"].append({
                "source": img_path,
                "location": location,
                "img_num": img_num,
                "severity": severity,
                "output": out_name,
                "strength": config["strength"],
                "prompt": config["prompt"],
            })

            print(f"  -> {out_name} (strength={config['strength']})")

    elapsed = time.time() - gen_start

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    metadata["generation_time_seconds"] = round(elapsed, 1)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    total_generated = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images generated: {total_generated}")
    print(f"Generation time: {elapsed:.1f}s ({elapsed/total_generated:.1f}s per image)")
    print(f"\nCounts per severity:")
    for sev, count in counts.items():
        print(f"  {sev:15s}: {count}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
