#!/usr/bin/env python3
"""
Test ControlNet depth-conditioned generation for FireEye.
Uses T2I-Adapter (lighter than ControlNet, fits RTX 3060 12GB).

Generates construction site fire scenes conditioned on depth maps
extracted from real photos, preserving structural layout.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import glob

OUTPUT_DIR = "/home/evnchn/Scaffluent_App/FireEye/research/controlnet_samples"
REAL_IMAGES_DIR = "/home/evnchn/Scaffluent_App/Images dataset/Real"

PROMPTS = [
    "construction site with scaffolding on fire, flames and smoke, fire spreading on facade netting",
    "construction site with small fire, smoke rising, workers evacuating, fire extinguisher visible",
    "construction site at night, welding sparks visible, scaffolding with safety nets, urban Hong Kong",
    "construction site with fire-retardant green scaffold net, fire extinguisher on wall, safe conditions",
    "construction site with gas cylinders near welding sparks, smoke, dangerous conditions",
]


def extract_depth_map(image, depth_estimator):
    """Extract depth map from an image using DPT."""
    inputs = depth_estimator.feature_extractor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = depth_estimator.model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    # Normalize
    output = prediction.squeeze().cpu().numpy()
    output = (output - output.min()) / (output.max() - output.min()) * 255
    return Image.fromarray(output.astype(np.uint8))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find real images
    real_files = sorted(glob.glob(os.path.join(REAL_IMAGES_DIR, "*", "*.png")))[:5]
    if not real_files:
        print("No real images found")
        return

    print(f"Found {len(real_files)} real images for depth conditioning")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Try T2I-Adapter first (lighter weight)
    try:
        from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, AutoencoderKL
        from transformers import DPTForDepthEstimation, DPTFeatureExtractor

        print("\nLoading depth estimator (DPT-Large)...")
        depth_fe = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to("cuda")

        class DepthEstimator:
            def __init__(self, fe, model):
                self.feature_extractor = fe
                self.model = model

        depth_est = DepthEstimator(depth_fe, depth_model)

        # Extract depth maps from real images
        print("Extracting depth maps...")
        depth_maps = []
        for img_path in real_files:
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            depth = extract_depth_map(img, depth_est)
            depth_maps.append((img_path, img, depth))

            # Save depth map for inspection
            base = os.path.splitext(os.path.basename(img_path))[0]
            depth.save(os.path.join(OUTPUT_DIR, f"depth_{base}.png"))

        # Free depth model
        del depth_model, depth_fe
        torch.cuda.empty_cache()

        print("\nLoading T2I-Adapter for depth conditioning...")
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-depth-sdxl-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Use SDXL-Turbo as base (already cached)
        from diffusers import AutoPipelineForText2Image
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            adapter=adapter,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        pipe.enable_attention_slicing()

        print(f"VRAM after loading: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        # Generate images
        print(f"\nGenerating {len(depth_maps) * len(PROMPTS)} depth-conditioned images...")
        t0 = time.time()
        count = 0

        for img_path, orig_img, depth_map in depth_maps:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for pidx, prompt in enumerate(PROMPTS):
                gen = torch.Generator("cuda").manual_seed(42 + pidx)
                # Convert depth to 3-channel
                depth_3ch = Image.merge("RGB", [depth_map] * 3)

                result = pipe(
                    prompt=prompt,
                    image=depth_3ch,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    adapter_conditioning_scale=0.8,
                    generator=gen,
                ).images[0]

                fname = f"depth_cond_{base}_p{pidx}.jpg"
                result.save(os.path.join(OUTPUT_DIR, fname), quality=90)
                count += 1

        elapsed = time.time() - t0
        print(f"Generated {count} images in {elapsed:.0f}s ({count/elapsed:.1f} img/s)")

    except Exception as e:
        print(f"T2I-Adapter approach failed: {e}")
        print("\nFalling back to simpler depth-guided img2img...")

        # Fallback: use depth maps as visual conditioning through img2img
        from diffusers import AutoPipelineForImage2Image

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipe.enable_attention_slicing()

        # Just use depth-colored images as input to img2img
        for img_path in real_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path).convert("RGB").resize((512, 512))

            for pidx, prompt in enumerate(PROMPTS):
                gen = torch.Generator("cuda").manual_seed(42 + pidx)
                result = pipe(
                    prompt=prompt, image=img,
                    strength=0.6,
                    num_inference_steps=4, guidance_scale=0.0,
                    generator=gen
                ).images[0]
                fname = f"depth_fallback_{base}_p{pidx}.jpg"
                result.save(os.path.join(OUTPUT_DIR, fname), quality=90)

        print("Fallback complete")

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Files: {len(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
