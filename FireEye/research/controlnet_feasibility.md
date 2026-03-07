# ControlNet SDXL Feasibility Study for RTX 3060 12GB

**Research Date:** 2026-03-08
**Hardware:** NVIDIA RTX 3060 12GB VRAM
**Goal:** Evaluate practical approaches for structurally-guided synthetic construction site image generation using ControlNet, T2I-Adapter, and IP-Adapter with SDXL on constrained VRAM.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [VRAM Budget Analysis](#2-vram-budget-analysis)
3. [Available Models (No Auth Required)](#3-available-models-no-auth-required)
4. [Approach A: ControlNet SDXL](#4-approach-a-controlnet-sdxl)
5. [Approach B: T2I-Adapter (Lightweight Alternative)](#5-approach-b-t2i-adapter-lightweight-alternative)
6. [Approach C: IP-Adapter (Reference-Image Guided)](#6-approach-c-ip-adapter-reference-image-guided)
7. [Approach D: SDXL-Turbo + ControlNet (Fastest)](#7-approach-d-sdxl-turbo--controlnet-fastest)
8. [Combined Approach: ControlNet + IP-Adapter](#8-combined-approach-controlnet--ip-adapter)
9. [Practical Code Snippets](#9-practical-code-snippets)
10. [Recommendation for FireEye](#10-recommendation-for-fireeye)
11. [Sources](#11-sources)

---

## 1. Executive Summary

**Verdict: Feasible, with the right model variant and optimizations.**

| Approach | Fits 12GB? | Speed | Structural Control | Complexity |
|---|---|---|---|---|
| ControlNet-SDXL (full) | Tight (needs offload) | ~25-35s/img | Excellent | Medium |
| ControlNet-SDXL (small) | Yes, comfortable | ~20-30s/img | Good | Medium |
| T2I-Adapter-SDXL | Yes, easily | ~18-25s/img | Good | Low |
| IP-Adapter-SDXL | Yes, easily | ~18-25s/img | Style only (no geometry) | Low |
| ControlNet + IP-Adapter | Tight | ~30-40s/img | Excellent + style | High |
| SDXL-Turbo + ControlNet | Possible but unproven | ~3-8s/img | Uncertain quality | Experimental |

**Best option for our use case:** T2I-Adapter (depth/canny) for bulk generation, ControlNet-small for high-quality structural guidance when needed.

---

## 2. VRAM Budget Analysis

### Baseline SDXL VRAM (fp16, single image 1024x1024)

| Component | VRAM (fp16) |
|---|---|
| SDXL UNet | ~5.0 GB |
| SDXL Text Encoders (CLIP-L + OpenCLIP-G) | ~1.5 GB |
| VAE (fp16-fix) | ~0.2 GB |
| Inference overhead (activations, KV cache) | ~2-4 GB |
| **Subtotal (SDXL alone)** | **~8.7-10.7 GB** |

### ControlNet Addon Cost

| ControlNet Variant | Parameters | File Size (fp16) | Additional VRAM |
|---|---|---|---|
| Full (`controlnet-canny-sdxl-1.0`) | ~1,251 M | ~2.5 GB | ~2.5-3.0 GB |
| Mid (`controlnet-canny-sdxl-1.0-mid`) | ~250 M | ~500 MB | ~0.5-0.8 GB |
| Small (`controlnet-depth-sdxl-1.0-small`) | ~179 M | ~350 MB | ~0.35-0.5 GB |

### T2I-Adapter Addon Cost

| Model | Parameters | File Size (fp16) | Additional VRAM |
|---|---|---|---|
| T2I-Adapter-SDXL (any type) | ~79 M | ~158 MB | ~0.15-0.2 GB |

### IP-Adapter Addon Cost

| Model | Parameters | File Size | Additional VRAM |
|---|---|---|---|
| IP-Adapter SDXL (base) | ~22 M | ~100 MB | ~0.1 GB (weights) |
| CLIP Image Encoder (ViT-H) | ~632 M | ~1.2 GB | ~1.2 GB |
| **IP-Adapter total** | | | **~1.3 GB** |

### Will It Fit on 12GB?

| Configuration | Estimated VRAM | Fits 12GB? | Optimization Needed |
|---|---|---|---|
| SDXL + ControlNet Full | ~11.2-13.7 GB | Marginal | cpu_offload required |
| SDXL + ControlNet Small | ~9.1-11.2 GB | Yes | fp16 sufficient |
| SDXL + T2I-Adapter | ~8.9-10.9 GB | Yes | fp16 sufficient |
| SDXL + IP-Adapter | ~10.0-12.0 GB | Yes | fp16 + maybe VAE slicing |
| SDXL + ControlNet Small + IP-Adapter | ~10.4-12.5 GB | Tight | cpu_offload recommended |
| SDXL + T2I-Adapter + IP-Adapter | ~10.1-12.1 GB | Yes | fp16 + VAE slicing |

### Key Optimization Techniques (from HuggingFace benchmarks)

| Technique | VRAM Savings | Speed Impact |
|---|---|---|
| `torch_dtype=torch.float16` | ~50% model weight reduction | None |
| `enable_model_cpu_offload()` | ~1.5 GB | Minimal (~10% slower) |
| `enable_sequential_cpu_offload()` | ~1.8 GB | Severe (~5x slower) |
| VAE slicing (`enable_vae_slicing()`) | ~6.3 GB | Minimal |
| VAE tiling (`enable_vae_tiling()`) | Prevents OOM on decode | Minimal |
| `madebyollin/sdxl-vae-fp16-fix` | Prevents NaN in fp16 VAE | None |

**Critical: Always use `madebyollin/sdxl-vae-fp16-fix`** -- the default SDXL VAE produces NaN in fp16, causing black images.

---

## 3. Available Models (No Auth Required)

All models below are publicly available on HuggingFace with no login required.

### ControlNet SDXL (by HuggingFace Diffusers team)

| Model ID | Type | Notes |
|---|---|---|
| `diffusers/controlnet-canny-sdxl-1.0` | Canny (full) | ~2.5 GB, best quality |
| `diffusers/controlnet-canny-sdxl-1.0-mid` | Canny (mid) | ~500 MB, 5x smaller |
| `diffusers/controlnet-canny-sdxl-1.0-small` | Canny (small) | ~350 MB, 7x smaller, no attention blocks |
| `diffusers/controlnet-depth-sdxl-1.0` | Depth (full) | ~2.5 GB |
| `diffusers/controlnet-depth-sdxl-1.0-mid` | Depth (mid) | ~500 MB |
| `diffusers/controlnet-depth-sdxl-1.0-small` | Depth (small) | ~350 MB |

### ControlNet SDXL (Community -- higher quality)

| Model ID | Type | Notes |
|---|---|---|
| `xinsir/controlnet-canny-sdxl-1.0` | Canny | Trained on 10M+ images, highest quality |
| `xinsir/controlnet-union-sdxl-1.0` | Union (all-in-one) | Supports 12 control types in one model, Apache 2.0 |

### T2I-Adapter SDXL (by TencentARC)

| Model ID | Type | Notes |
|---|---|---|
| `TencentARC/t2i-adapter-canny-sdxl-1.0` | Canny | 79M params, 158 MB |
| `TencentARC/t2i-adapter-depth-midas-sdxl-1.0` | Depth (MiDaS) | 79M params, 158 MB |
| `TencentARC/t2i-adapter-depth-zoe-sdxl-1.0` | Depth (ZOE) | 79M params, 158 MB |
| `TencentARC/t2i-adapter-lineart-sdxl-1.0` | Lineart | 79M params, 158 MB |
| `TencentARC/t2i-adapter-sketch-sdxl-1.0` | Sketch | 79M params, 158 MB |
| `TencentARC/t2i-adapter-openpose-sdxl-1.0` | OpenPose | 79M params, 158 MB |

### IP-Adapter SDXL (by h94)

| Model ID | Subfolder | Weight Name | Notes |
|---|---|---|---|
| `h94/IP-Adapter` | `sdxl_models` | `ip-adapter_sdxl.bin` | Base, ~100 MB |
| `h94/IP-Adapter` | `sdxl_models` | `ip-adapter-plus_sdxl_vit-h.safetensors` | Plus (patch embeddings) |
| `h94/IP-Adapter` | `sdxl_models` | `ip-adapter-plus-face_sdxl_vit-h.safetensors` | Face-specific |

---

## 4. Approach A: ControlNet SDXL

### How it works

ControlNet adds a copy of the UNet encoder blocks with "zero convolution" layers that are trained while the original model stays frozen. During inference, both the main UNet and the ControlNet copy run at every denoising step, which is why it uses significant VRAM and is slower.

### Strengths for construction site generation

- **Depth conditioning** preserves 3D structure: scaffolding positions, building geometry, floor/ceiling
- **Canny conditioning** preserves edges: scaffold poles, net boundaries, doorframes
- Extract depth/canny from the 29 real photos, then generate dozens of risk-level variations per photo
- Multi-ControlNet (canny + depth together) gives the tightest structural control

### Weaknesses

- Full ControlNet adds ~2.5 GB VRAM -- tight on 12GB
- Runs at every denoising step, so ~30-50% slower than base SDXL
- Multi-ControlNet (2 models) is very tight on 12GB, may require cpu_offload

### Recommended variant for RTX 3060

Use **`controlnet-depth-sdxl-1.0-small`** (7x smaller, ~350 MB) for most work. The small variant handles typical conditioning well and leaves ~2 GB headroom. Switch to the full model with `enable_model_cpu_offload()` only when higher fidelity is needed.

---

## 5. Approach B: T2I-Adapter (Lightweight Alternative)

### How it works

T2I-Adapter inserts lightweight adapter weights into the UNet's downsampling blocks. Unlike ControlNet, it runs **only once** at the start of the denoising process (not at every step), making it both faster and more memory-efficient.

### Size comparison

| | ControlNet-SDXL | T2I-Adapter-SDXL | Reduction |
|---|---|---|---|
| Parameters | 1,251 M | 79 M | **93.7%** |
| File size (fp16) | 2.5 GB | 158 MB | **94%** |
| Additional VRAM | ~2.5-3.0 GB | ~0.15-0.2 GB | **~94%** |
| Runs per denoising step | Every step | Once | Much faster |

### Strengths for construction site generation

- Fits comfortably on 12GB alongside SDXL
- Can combine multiple T2I-Adapters (canny + depth) without VRAM issues
- Available for the same conditioning types we need: canny, depth-midas, depth-zoe
- Faster generation enables higher volume

### Weaknesses

- Slightly less precise structural control than ControlNet (especially for complex scenes)
- Fewer community-trained variants available
- Less battle-tested than ControlNet for SDXL

### Verdict

T2I-Adapter is the **best default choice** for bulk synthetic data generation on RTX 3060 12GB. The 94% reduction in adapter size means it essentially runs "for free" on top of SDXL.

---

## 6. Approach C: IP-Adapter (Reference-Image Guided)

### How it works

IP-Adapter adds decoupled cross-attention layers that process image features (from a CLIP image encoder) separately from text features. This allows you to pass a reference image that guides the style, composition, or subject of the generated image.

### Strengths for construction site generation

- Feed a real construction site photo as the reference to maintain visual consistency
- Combine with text prompts for risk-level variation: same reference photo + "fire hazard" prompt
- ~100 MB weights, very lightweight
- Can precompute image embeddings and reuse across batch generation

### Weaknesses

- Provides **style/subject** guidance, not **structural/geometric** guidance
- Needs CLIP ViT-H image encoder loaded (~1.2 GB additional VRAM)
- Less precise spatial control than ControlNet or T2I-Adapter
- Best used in combination with structural controls, not alone

### Best use case for FireEye

Use IP-Adapter alongside T2I-Adapter or ControlNet to transfer the "look and feel" of real construction site photos while maintaining structural layout.

---

## 7. Approach D: SDXL-Turbo + ControlNet (Fastest)

### Status: Experimental / Not Recommended

SDXL-Turbo uses Adversarial Diffusion Distillation (ADD) to generate images in 1-4 steps (vs. 20-50 for SDXL). Inference does not use classifier-free guidance, reducing VRAM.

**However:**
- There are **no official ControlNet models trained specifically for SDXL-Turbo**
- Using SDXL ControlNet models with SDXL-Turbo may work but produces degraded quality because:
  - The ControlNet was trained with multi-step denoising, Turbo uses 1-4 steps
  - The conditioning signal doesn't have enough steps to guide the generation properly
- SDXL-Turbo generates at 512x512 natively, not 1024x1024
- Quality at 1-4 steps is notably lower than full SDXL at 30 steps

### If you want speed, consider instead:

- **LCM-LoRA with SDXL** (4-8 step generation, compatible with ControlNet)
- **T2I-Adapter** (single-pass, faster than ControlNet by design)
- Reducing steps to 20 with DPM++ 2M Karras scheduler

---

## 8. Combined Approach: ControlNet + IP-Adapter

The diffusers library supports combining IP-Adapter with ControlNet in the same pipeline. This gives both structural control (from depth/canny maps) and style transfer (from reference images).

### VRAM estimate on RTX 3060

| Component | VRAM |
|---|---|
| SDXL base (fp16) | ~6.5 GB |
| ControlNet-small (fp16) | ~0.35 GB |
| IP-Adapter weights | ~0.1 GB |
| CLIP ViT-H encoder | ~1.2 GB |
| Inference overhead | ~2-3 GB |
| **Total** | **~10.2-11.2 GB** |

This fits in 12GB with fp16 and VAE slicing. Use `enable_model_cpu_offload()` for safety margin.

---

## 9. Practical Code Snippets

### 9a. ControlNet-Small + Depth Map (Recommended for quality)

```python
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)

# --- Step 1: Generate depth map from real photo ---
depth_estimator = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas"
).to("cuda")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    inputs = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(inputs).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

real_photo = Image.open("path/to/construction_site.jpg").resize((1024, 1024))
depth_image = get_depth_map(real_photo)

# Free depth model VRAM before loading SDXL
del depth_estimator, feature_extractor
torch.cuda.empty_cache()

# --- Step 2: Load SDXL + ControlNet-small pipeline ---
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()  # Saves ~1.5 GB
pipe.enable_vae_slicing()        # Prevents OOM on VAE decode
pipe.enable_vae_tiling()         # Further VAE memory reduction

# --- Step 3: Generate construction site variants ---
prompt = (
    "A photorealistic photograph of a construction site interior, "
    "scaffolding with scaffold nets, fire extinguisher on wall, "
    "emergency exit sign, industrial lighting, 8K, sharp focus"
)
negative_prompt = "cartoon, illustration, painting, blurry, low quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=depth_image,
    controlnet_conditioning_scale=0.5,  # 0.5-0.7 recommended
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("generated_construction_site.png")
```

### 9b. T2I-Adapter + Canny (Recommended for bulk generation)

```python
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)

# --- Step 1: Extract canny edges from real photo ---
real_photo = Image.open("path/to/construction_site.jpg").resize((1024, 1024))
image_array = np.array(real_photo)
canny = cv2.Canny(image_array, 100, 200)
canny_image = Image.fromarray(np.stack([canny] * 3, axis=2))

# --- Step 2: Load T2I-Adapter pipeline ---
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler",
)

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# --- Step 3: Generate ---
prompt = (
    "A photorealistic photograph of a construction site with medium "
    "fire risk, hot work near combustible materials, scaffold nets "
    "not pulled back, no welding screen, industrial setting"
)
negative_prompt = "cartoon, illustration, painting, blurry, low quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    adapter_conditioning_scale=0.8,  # 0.7-0.9 typical range
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("generated_medium_risk.png")
```

### 9c. IP-Adapter + ControlNet Combined

```python
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image

# Load ControlNet (small for VRAM savings)
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)

# Load IP-Adapter ON TOP of the ControlNet pipeline
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)
pipe.set_ip_adapter_scale(0.5)  # Balance between text prompt and reference image

# IMPORTANT: enable_model_cpu_offload AFTER loading IP-Adapter
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Reference image: a real construction site photo for style transfer
reference_image = load_image("path/to/real_construction_photo.jpg")

# Depth map extracted from same or different photo
depth_image = load_image("path/to/depth_map.png")

image = pipe(
    prompt="construction site with fire hazard, gas cylinders near heat source",
    negative_prompt="cartoon, illustration, blurry",
    image=depth_image,                    # ControlNet conditioning
    ip_adapter_image=reference_image,     # IP-Adapter style guidance
    controlnet_conditioning_scale=0.5,
    num_inference_steps=30,
).images[0]
```

### 9d. Multi-ControlNet (Canny + Depth) -- tight on 12GB

```python
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)

# Use SMALL variants to fit in 12GB
controlnets = [
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-small",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ),
]

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnets,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Both conditioning images must be 1024x1024
images = [depth_image.resize((1024, 1024)), canny_image.resize((1024, 1024))]

image = pipe(
    prompt="photorealistic construction site, scaffolding, fire extinguisher",
    negative_prompt="cartoon, blurry, low quality",
    image=images,
    controlnet_conditioning_scale=[0.6, 0.4],  # Depth > Canny for scene structure
    num_inference_steps=30,
).images[0]
```

### 9e. Batch Generation Loop (for bulk synthetic data)

```python
import os
import torch
from pathlib import Path

# Assume pipe is already loaded (any of the above pipelines)

PROMPTS_BY_RISK = {
    "safe": (
        "well-organized construction site, fire extinguisher on wall, "
        "emergency exit signs illuminated, clean work area, no flames"
    ),
    "low_risk": (
        "construction site with controlled hot work, worker welding "
        "behind welding screen, fire extinguisher nearby, proper PPE"
    ),
    "medium_risk": (
        "construction site with hot work without welding screens, "
        "combustible materials within 5 meters of sparks, scaffold nets "
        "near hot work area"
    ),
    "high_risk": (
        "dangerous construction site, open flame near scaffold, "
        "gas cylinders near ignition source, blocked emergency exit, "
        "no fire extinguishers visible"
    ),
    "critical": (
        "construction site fire emergency, flames spreading along "
        "scaffold nets, heavy smoke, gas cylinders exposed to heat"
    ),
}

output_dir = Path("synthetic_controlnet_output")
output_dir.mkdir(exist_ok=True)

# Iterate over conditioning images (depth maps from real photos)
conditioning_dir = Path("depth_maps")

for depth_file in sorted(conditioning_dir.glob("*.png")):
    depth_image = Image.open(depth_file).resize((1024, 1024))
    base_name = depth_file.stem

    for risk_level, prompt_text in PROMPTS_BY_RISK.items():
        for seed in range(5):  # 5 variations per combination
            generator = torch.Generator("cuda").manual_seed(seed)

            image = pipe(
                prompt=f"A photorealistic photograph of a {prompt_text}, "
                       f"industrial lighting, 8K, sharp focus",
                negative_prompt="cartoon, illustration, painting, blurry",
                image=depth_image,
                controlnet_conditioning_scale=0.5,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]

            filename = f"{base_name}_{risk_level}_seed{seed}.png"
            image.save(output_dir / filename)
            print(f"Saved: {filename}")

    # Clear CUDA cache between base images
    torch.cuda.empty_cache()
```

---

## 10. Recommendation for FireEye

### Primary Strategy: T2I-Adapter for Bulk, ControlNet-Small for Quality

1. **Bulk generation (80% of images):** Use `T2I-Adapter-SDXL` (canny or depth-midas) for rapid generation. At 79M parameters and 158 MB, it leaves ample VRAM headroom and runs only once per generation (not per denoising step). Expected speed: ~18-25s per image at 1024x1024.

2. **High-fidelity structural guidance (20% of images):** Use `controlnet-depth-sdxl-1.0-small` with `enable_model_cpu_offload()`. This gives tighter structural control for complex scenes where scaffolding geometry and object placement matter most.

3. **Style consistency (optional):** Add IP-Adapter with a real construction site photo as reference. This is most useful if generated images look too "generic" and need to match the visual character of our real 29-photo dataset.

4. **Skip SDXL-Turbo** for ControlNet work -- there are no compatible ControlNet models, and the 1-4 step inference degrades structural guidance quality.

### Pipeline Order

```
Real construction photo (from 29-image dataset)
    |
    +---> MiDaS depth estimation ---> depth map (save to disk, reuse)
    |
    +---> OpenCV Canny (100, 200) ---> canny edges (save to disk, reuse)
    |
    v
For each (depth_map, risk_level_prompt, seed):
    T2I-Adapter pipeline ---> generated image ---> save with metadata
    |
    v
Manual QA pass ---> keep realistic images, discard artifacts
    |
    v
Auto-annotation (Grounding DINO) + manual correction in CVAT
    |
    v
YOLO training dataset (bridged transfer: pretrain synthetic, finetune real)
```

### Expected Output

- 29 real photos x 5 risk levels x 5 seeds = **725 structurally-guided images**
- Additional T2I (no conditioning) for diversity: **+500-1000 images**
- Total generation time: ~6-8 hours on RTX 3060 12GB
- Total VRAM peak: ~9-10 GB (T2I-Adapter) or ~10-11 GB (ControlNet-small)

---

## 11. Sources

### ControlNet SDXL Models
- [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) - Official HF canny model
- [diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0) - Official HF depth model
- [diffusers/controlnet-depth-sdxl-1.0-small](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small) - 7x smaller variant
- [diffusers/controlnet-canny-sdxl-1.0-mid](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-mid) - 5x smaller variant
- [xinsir/controlnet-canny-sdxl-1.0](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0) - Community model trained on 10M+ images
- [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) - All-in-one 12-type ControlNet

### T2I-Adapter
- [Efficient Controllable Generation for SDXL with T2I-Adapters](https://huggingface.co/blog/t2i-sdxl-adapters) - HF blog with benchmarks
- [TencentARC/T2I-Adapter GitHub](https://github.com/TencentARC/T2I-Adapter) - Source code
- [Meet T2I-Adapter-SDXL: Small and Efficient Control Models](https://www.marktechpost.com/2023/09/11/meet-t2i-adapter-sdxl-small-and-efficient-control-models/) - Overview

### IP-Adapter
- [h94/IP-Adapter on HuggingFace](https://huggingface.co/h94/IP-Adapter) - Model weights
- [IP-Adapter Diffusers Documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter) - Full usage guide
- [IP-Adapter Project Page](https://ip-adapter.github.io/) - Paper and demos

### Diffusers Documentation
- [ControlNet Guide](https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet) - Official usage docs
- [ControlNet SDXL Pipeline API](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl) - API reference
- [Memory Optimization](https://huggingface.co/docs/diffusers/optimization/memory) - cpu_offload, VAE slicing, etc.
- [Simple SDXL Optimizations](https://huggingface.co/blog/simple_sdxl_optimizations) - VRAM benchmarks

### VRAM and Hardware
- [SDXL 12GB VRAM Optimization](https://itctshop.com/sdxl-12gb-vram-optimization/) - RTX 3060 specific tips
- [SDXL VRAM 2026: 12GB Works, 24GB Is Safer](https://www.synpixcloud.com/blog/sdxl-vram-requirements-guide) - Requirements guide
- [SDXL System Requirements](https://stablediffusionxl.com/sdxl-system-requirements/) - Minimum specs
- [ControlNet VRAM Discussion](https://github.com/Mikubill/sd-webui-controlnet/discussions/212) - Community benchmarks

### SDXL-Turbo
- [stabilityai/sdxl-turbo on HuggingFace](https://huggingface.co/stabilityai/sdxl-turbo) - Model card
- [SDXL Turbo ONNX Runtime Optimization](https://huggingface.co/blog/sdxl_ort_inference) - Performance benchmarks
