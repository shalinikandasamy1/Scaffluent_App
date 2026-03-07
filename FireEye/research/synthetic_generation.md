# Synthetic Training Image Generation for FireEye

**Research Date:** 2026-03-08
**Goal:** Evaluate approaches for generating synthetic construction site fire safety images to augment a small dataset (29 real photos + welding video frames) for YOLO-based object detection training.
**Hardware:** NVIDIA RTX 3060 12GB VRAM

---

## Table of Contents

1. [Does Synthetic Data Actually Help?](#1-does-synthetic-data-actually-help)
2. [Generation Approaches Compared](#2-generation-approaches-compared)
3. [SDXL on RTX 3060 12GB -- Feasibility and Optimization](#3-sdxl-on-rtx-3060-12gb----feasibility-and-optimization)
4. [Recommended Checkpoints and Models](#4-recommended-checkpoints-and-models)
5. [ControlNet for Structural Guidance](#5-controlnet-for-structural-guidance)
6. [Cloud/API Alternatives (DALL-E, Midjourney, Flux)](#6-cloudapi-alternatives-dall-e-midjourney-flux)
7. [Sample Prompts by Risk Level](#7-sample-prompts-by-risk-level)
8. [Recommended Pipeline](#8-recommended-pipeline)
9. [Key Risks and Mitigations](#9-key-risks-and-mitigations)
10. [Sources](#10-sources)

---

## 1. Does Synthetic Data Actually Help?

**Short answer: Yes, substantially, especially for small datasets like ours.**

### Key Findings from 2025 Research

- **Up to 34% improvement:** A University of South Carolina benchmark (Oct 2025) found that models trained 100% on synthetic images outperformed models trained 100% on real images by up to 34% mAP50-95 and ~22% recall across all seven tested architectures, when evaluated on real-world imagery.

- **Bridged transfer is optimal:** A two-stage "bridged transfer" approach -- pre-train on synthetic data, then fine-tune on real data -- consistently produces higher accuracy than either pure real or pure synthetic training. This is the recommended approach for FireEye given our small real dataset.

- **200 real + 5000 synthetic works well:** Research shows that for datasets with fewer object classes (like ours: extinguisher, flame, gas cylinder, scaffold net, exit, etc.), augmenting a small real set with a large synthetic set yields strong results.

- **Segmentation-guided inpainting improves results:** Using segmentation masks to constrain where diffusion models insert objects into scenes improved mean mAP50 from 0.579 to 0.647 compared to real-only training in forestry environments (analogous low-data domain).

- **Early layers transfer well, heads diverge:** The largest similarity between real-trained and synthetic-trained detectors is in early convolutional layers; the largest difference is in the detection head. This means fine-tuning the head on real data after synthetic pre-training is critical.

### Construction Safety Specific Research

- A 2024 paper in Automation in Construction used text-to-image models to generate 3,585 synthetic images across 27 hazardous construction scenarios, demonstrating that diffusion-generated images can capture object relationships relevant to safety assessment.

- Game-engine-based approaches (Unreal Engine 5) have also been used for construction worker safety monitoring, offering perfect ground-truth annotations but requiring 3D asset creation.

### Verdict for FireEye

Given our dataset size (29 images), synthetic augmentation is not just helpful but almost mandatory. The bridged transfer approach (pre-train on ~2000-5000 synthetic images, fine-tune on real) is the recommended strategy.

---

## 2. Generation Approaches Compared

### Text-to-Image (T2I)

**How it works:** Generate images purely from text prompts describing the desired scene.

| Pros | Cons |
|------|------|
| Simplest pipeline | No spatial control over object placement |
| High diversity | Inconsistent object counts/positions |
| Easy to scale with batch prompting | May generate physically implausible scenes |
| No input images needed | Harder to get specific safety-relevant details right |

**Best for:** Initial dataset bootstrapping, generating diverse backgrounds and scenarios.

### Image-to-Image (I2I)

**How it works:** Start from a reference image (e.g., a real construction site photo) and modify it via prompt + denoising.

| Pros | Cons |
|------|------|
| Preserves scene structure from real photos | Requires source images |
| More realistic layouts | Less diversity than pure T2I |
| Can vary risk levels from same base scene | May produce artifacts at object boundaries |

**Best for:** Creating variations of our 29 real images at different risk levels. Take a safe construction site photo and progressively add hazards.

### Inpainting

**How it works:** Mask specific regions of an existing image and regenerate only those areas.

| Pros | Cons |
|------|------|
| Precise control over what changes | Requires manual or automated mask creation |
| Can add/remove specific objects (extinguisher, flame) | Boundary artifacts possible |
| Scene context preserved perfectly | Slower per-image workflow |

**Best for:** Adding specific hazard objects (flames, gas cylinders) to safe base images, or removing safety equipment to simulate higher risk.

### ControlNet-Guided Generation

**How it works:** Use structural guidance (depth maps, edge maps, segmentation) to control the generated image layout while varying content via prompts.

| Pros | Cons |
|------|------|
| Precise structural control | Additional VRAM overhead (~2GB) |
| Consistent scene geometry | Requires preprocessing reference images |
| Can reuse depth/edge from real photos | Limited SDXL ControlNet models available |
| Multi-ControlNet combines canny + depth | Slower generation |

**Best for:** Maintaining realistic construction site layouts while varying fire safety conditions. Extract depth/edge from real photos, then generate variations.

### Recommended Combination for FireEye

1. **ControlNet depth + canny** from real photos as structural base
2. **Inpainting** to add/remove specific safety objects
3. **T2I** for additional diverse scenes
4. **I2I** for risk-level variations of the same scene

---

## 3. SDXL on RTX 3060 12GB -- Feasibility and Optimization

### Verdict: Fully feasible with optimizations

SDXL generates at 1024x1024 natively. The RTX 3060 12GB sits at the minimum for comfortable SDXL operation. Here are the required optimizations:

### Essential Optimizations

| Optimization | Effect | How |
|---|---|---|
| **Tiled VAE** | Prevents OOM during image decode | Enable in A1111/ComfyUI settings |
| **FP8 weights** | Reduces model VRAM from ~6.5GB to ~3.5GB | `--fp8_e4m3fn-unet` flag |
| **--medvram-sdxl** | Reduces idle VRAM by ~40% | A1111 launch argument |
| **Unload refiner** | Saves ~6GB if not using refiner | Skip refiner or enable unload |
| **ComfyUI** | More efficient memory management than A1111 | Use ComfyUI instead of A1111 |

### What Works on 12GB

| Task | Feasible? | Notes |
|---|---|---|
| SDXL base generation 1024x1024 | Yes | With FP8 + Tiled VAE |
| SDXL + 1 ControlNet | Yes | With FP8, tight but works |
| SDXL + 2 ControlNets | Marginal | May need --lowvram |
| SDXL + Refiner | No | Requires model swapping (slow) |
| SDXL LoRA training | Yes | With gradient checkpointing, batch=1 |
| Flux.1 Dev (quantized) | Yes | FP8 or Q8 GGUF fits in 12GB |

### Expected Generation Speed

- SDXL 1024x1024, 30 steps, FP8: ~15-25 seconds per image
- With ControlNet: ~20-35 seconds per image
- Batch of 1000 images: ~6-10 hours

### Recommended Software Stack

**ComfyUI** is strongly recommended over Automatic1111 for this use case:
- Better memory management (node-based, loads/unloads on demand)
- Native batch processing with queue system
- Easier to build automated pipelines
- Better ControlNet integration for SDXL

---

## 4. Recommended Checkpoints and Models

### For Photorealistic Construction/Industrial Scenes

No construction-specific SDXL checkpoint exists. The best approach is a photorealistic base checkpoint + construction-focused prompting (and optionally a LoRA).

| Checkpoint | Why | Notes |
|---|---|---|
| **Juggernaut XL** | Best all-around photorealism, strong on environments | Most recommended for scene generation |
| **epiCRealism XL** | Excellent photorealism, good at architecture | Strong for building/structure scenes |
| **CyberRealistic XL v9** | Reliable photorealism, active development | Good fallback option |
| **Halcyon SDXL v1.9** | Photographer-trained, natural lighting | Best for realistic lighting conditions |
| **RealVisXL** | Photorealistic with good object detail | Good for equipment/object detail |

### LoRA Strategy

Since no pre-made construction fire safety LoRA exists, consider:

1. **Use base checkpoint + detailed prompting first** -- this may be sufficient
2. **Train a LoRA** on the 29 real images (feasible on RTX 3060 12GB with Kohya_ss, batch=1, gradient checkpointing) to capture construction site visual style
3. **Combine with object-specific LoRAs** from CivitAI if available (fire/flame LoRAs exist)

### ControlNet Models for SDXL

| Model | Use Case |
|---|---|
| `controlnet-canny-sdxl-1.0` | Edge-guided generation from real photo edges |
| `controlnet-depth-sdxl-1.0` | Depth-guided for 3D structure preservation |
| `controlnet-openpose-sdxl-1.0` | Worker pose control (if needed) |

---

## 5. ControlNet for Structural Guidance

### Recommended Pipeline for FireEye

```
Real construction photo
    |
    +---> Canny edge extraction ---> ControlNet Canny
    |                                     |
    +---> Depth estimation (MiDaS) --> ControlNet Depth
                                          |
                                    Multi-ControlNet
                                          |
                                    SDXL base model
                                          +
                                    Risk-level prompt
                                          |
                                    Generated image
```

### Why This Works for Safety Scenarios

- **Depth map** preserves 3D structure: scaffolding positions, building geometry, floor/ceiling relationships
- **Canny edges** preserve structural details: scaffold poles, net boundaries, doorways
- **Prompt controls risk level:** same structural base, different fire safety conditions
- One real photo can spawn 10-20 variations across risk levels

### Multi-ControlNet Tips

- Use `controlnet_conditioning_scale` of 0.5-0.7 for each when combining (lower than single-ControlNet)
- Mask conditionings so they don't overlap where possible
- Depth is more important than canny for scene structure; weight it higher

---

## 6. Cloud/API Alternatives (DALL-E, Midjourney, Flux)

### Comparison Matrix

| Platform | Quality | Cost | Speed | Local? | API? | Construction Scenes |
|---|---|---|---|---|---|---|
| **SDXL (local)** | Good | Free (electricity) | ~20s/img | Yes | N/A | Good with prompting |
| **Flux.1 Dev (local)** | Very Good | Free (electricity) | ~30-45s/img | Yes (quantized) | N/A | Better prompt adherence |
| **Flux.1 Pro (API)** | Excellent | ~$0.04-0.06/img | ~4.5s/img | No | Yes | Best open realism |
| **DALL-E 3** | Very Good | ~$0.04-0.08/img | ~10s/img | No | Yes (OpenAI) | Best text understanding |
| **Midjourney v6** | Excellent (artistic) | $10-30/month | ~30s/img | No | Limited | Artistic, less photorealistic |
| **Google Imagen 3** | Very Good | ~$0.02-0.06/img | Fast | No | Yes | Good realism |
| **Gemini 2.5 Flash** | Good | ~$0.00-0.01/img | Fast | No | Yes (OpenRouter) | Already in our stack |

### Cost Analysis for 2000 Synthetic Images

| Approach | Cost | Notes |
|---|---|---|
| Local SDXL (RTX 3060) | ~$2-3 electricity | 8-10 hours generation time |
| Local Flux.1 Dev (quantized) | ~$3-5 electricity | 15-25 hours, better quality |
| Flux.1 Pro API | ~$80-120 | Fast, highest quality |
| DALL-E 3 API | ~$80-160 | Great prompt understanding |
| Google Imagen 3 API | ~$40-120 | Good middle ground |
| Gemini 2.5 Flash (OpenRouter) | ~$5-20 | Already configured in project |

### Recommendation

**Hybrid approach:**
1. **Primary: Local SDXL** for bulk generation (free, controllable, ControlNet support)
2. **Secondary: Gemini 2.5 Flash via OpenRouter** for quick supplementary images (already in our stack, very cheap)
3. **Validation: Flux.1 Pro API** for a small set of high-quality reference images to compare against local generation quality

### Note on Gemini as Image Generator

Since FireEye already uses Gemini 2.5 Flash via OpenRouter for LLM analysis, and Gemini models now support image generation, this is the lowest-friction option for generating a small supplementary set. However, it offers no ControlNet-style structural guidance.

---

## 7. Sample Prompts by Risk Level

### General Construction Site Prefix

All prompts should start with a scene-setting prefix:

```
A photorealistic photograph of a construction site interior/exterior,
scaffolding with scaffold nets, concrete floors, exposed steel beams,
construction materials, industrial lighting, [time of day],
high detail, professional photography, 8K, sharp focus
```

### Safe (Risk Level 0)

```
Prompt: A photorealistic photograph of a well-organized construction site
interior. Bright red fire extinguisher mounted on wall clearly visible.
Emergency exit signs illuminated in green above clear unobstructed
doorways. Fire-retardant scaffold nets (green/grey) properly installed
on scaffolding. Clean work area, no combustible materials scattered.
Safety signage visible. No flames, no smoke, no hot work. Professional
construction photography, natural daylight, sharp focus.

Negative prompt: fire, flame, smoke, sparks, welding, messy, cluttered,
damaged, dark, blurry, cartoon, illustration, painting
```

### Low Risk (Risk Level 1)

```
Prompt: A photorealistic photograph of a construction site with
controlled hot work in progress. A worker in full PPE performing
welding behind a proper welding screen/curtain. Fire watch personnel
visible nearby. Hot work permit sign posted. Fire extinguisher within
arm's reach. Minimum 6 meter clearance from combustible materials.
Scaffold nets pulled back from welding area. Spark containment blanket
on floor. Construction site, industrial setting.

Negative prompt: uncontrolled fire, spreading flames, panic, no PPE,
blurry, cartoon, illustration, painting
```

### Medium Risk (Risk Level 2)

```
Prompt: A photorealistic photograph of a construction site showing
medium fire risk hazards. Hot work (welding/cutting) being performed
WITHOUT proper welding screens or curtains. Combustible materials
(timber, cardboard, insulation) stacked within 3-5 meters of sparks.
Scaffold nets near hot work area not pulled back. Some fire
extinguishers present but not immediately adjacent. Partially blocked
emergency exit with construction materials. Industrial construction
interior, realistic lighting.

Negative prompt: active fire, large flames, explosion, cartoon,
illustration, painting, blurry
```

### High Risk (Risk Level 3)

```
Prompt: A photorealistic photograph of a dangerous construction site
with high fire risk. Uncontrolled open flame visible near scaffold
structure. Propane gas cylinders stored near an ignition source with
no separation barrier. Emergency exit completely blocked by stacked
construction materials and debris. No fire extinguishers visible in
the area. Combustible scaffold nets hanging near heat source. Dark
industrial construction site, ominous lighting, smoke haze in air.

Negative prompt: cartoon, illustration, painting, blurry, safe,
organized, clean
```

### Critical Risk (Risk Level 4)

```
Prompt: A photorealistic photograph of a construction site fire
emergency in progress. Fire actively spreading along facade scaffold
nets, flames climbing up exterior scaffolding. Burning debris falling.
Gas cylinders exposed to radiant heat from nearby fire, visible heat
distortion. Heavy black smoke billowing. Multiple fire hazards
simultaneously present. Emergency evacuation scenario. Chaotic
construction site, dramatic urgent lighting, intense orange and red
fire glow against dark smoke.

Negative prompt: cartoon, illustration, painting, blurry, calm,
peaceful, clean, safe
```

### Prompt Engineering Tips

1. **Be specific about object relationships:** "gas cylinders stored within 2 meters of welding sparks" is better than "gas cylinders near fire"
2. **Include camera perspective:** "eye-level view," "CCTV camera angle from above," "wide-angle lens" to match deployment camera views
3. **Vary lighting:** daylight, overcast, artificial lighting, dusk -- detector must handle all
4. **Vary weather for exteriors:** clear, overcast, rain (wet surfaces reflect differently)
5. **Include negative prompts** to avoid non-photorealistic outputs
6. **Use prompt weighting** for critical objects: `(fire extinguisher:1.3)`, `(blocked exit:1.2)`

---

## 8. Recommended Pipeline

### Phase 1: Infrastructure Setup (Day 1)

1. Install ComfyUI with SDXL support
2. Download Juggernaut XL or epiCRealism XL checkpoint (~6.5GB)
3. Download ControlNet SDXL canny + depth models (~5GB each)
4. Verify generation works on RTX 3060 with FP8 + Tiled VAE
5. Test a few manual prompts from each risk level

### Phase 2: Guided Generation from Real Photos (Days 2-3)

1. Extract canny edges and depth maps from all 29 real photos
2. Use Multi-ControlNet (canny + depth) with risk-level prompts
3. Generate 10-20 variations per real photo across risk levels
4. Target: ~300-500 structurally-guided images

### Phase 3: Pure T2I Diversity Expansion (Days 3-5)

1. Create prompt matrix: 5 risk levels x 10 scene types x 5 variations
2. Scene types: interior scaffold, exterior facade, basement, rooftop, corridor, stairwell, material storage, welding bay, elevator shaft, parking structure
3. Use ComfyUI batch queue for automated generation
4. Target: ~1000-2000 additional images

### Phase 4: Annotation (Days 5-7)

1. Use Grounding DINO or similar open-vocabulary detector for auto-annotation
2. Manual review and correction in CVAT or Label Studio
3. Export in YOLO format

### Phase 5: Training (Day 8+)

1. Bridged transfer: pre-train YOLO on synthetic dataset
2. Fine-tune on real dataset
3. Evaluate on held-out real test set

### Total Timeline: ~8-10 days for a usable augmented dataset

---

## 9. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| SD generates physically implausible scenes | Detector learns wrong patterns | Manual QA pass, remove unrealistic images |
| Objects too small/large relative to scene | Poor detection at deployment scale | Control resolution, use ControlNet for scale consistency |
| Style gap between synthetic and real | Detector fails on real images | Bridged transfer (pre-train synthetic, fine-tune real) |
| Flame/fire rendering unrealistic | Poor fire detection | Supplement with real fire images from public datasets (e.g., FireNet, FLAME) |
| ControlNet adds too much VRAM overhead | OOM on RTX 3060 | Use FP8 for both base model and ControlNet, or use single ControlNet only |
| Annotation of synthetic images is expensive | Bottleneck in pipeline | Use auto-annotation (Grounding DINO) + manual correction |
| Prompt engineering is iterative | Slow initial progress | Start with sample prompts above, iterate based on visual inspection |

---

## 10. Sources

### Synthetic Data Effectiveness
- [The Impact of Synthetic Data on Object Detection Model Performance](https://arxiv.org/html/2510.12208v1) - Oct 2025 comparative analysis
- [Better Than Real? Apple-Orchard Benchmark on Synthetic Data](https://www.edge-ai-vision.com/2025/12/better-than-real-what-an-apple-orchard-benchmark-really-says-about-synthetic-data-for-vision-ai/) - Dec 2025, up to 34% improvement
- [Proof: Synthetic Data Outperforms Real-World Training by 34%](https://synetic.ai/proof/) - Synetic AI benchmark
- [The Big Data Myth: Using Diffusion Models for Dataset Generation](https://arxiv.org/abs/2306.09762) - Foundational paper
- [ODGEN: Domain-specific Object Detection Data Generation](https://arxiv.org/html/2405.15199v1) - Domain-specific generation pipeline
- [Generative AI and Simulation-Based Data Augmentation for Forestry](https://www.mdpi.com/1999-4907/17/3/302) - Low-data domain augmentation

### Synthetic Data for Construction Safety
- [Image generation of hazardous situations in construction sites using text-to-image](https://www.sciencedirect.com/science/article/abs/pii/S0926580524003510) - 3585 images across 27 hazardous scenarios
- [Generative AI-driven data augmentation for construction hazard detection](https://www.sciencedirect.com/science/article/abs/pii/S0926580525003577) - 2025
- [Game engine-driven synthetic data for construction worker safety](https://www.sciencedirect.com/science/article/abs/pii/S0926580523003205) - UE5 approach
- [LoFT: LoRA-Fused Training Dataset Generation](https://arxiv.org/html/2505.11703v1) - LoRA fusion for data generation

### Stable Diffusion and ControlNet
- [Synthetic Data Generation with Stable Diffusion: A Guide](https://blog.roboflow.com/synthetic-data-with-stable-diffusion-a-guide/) - Roboflow practical guide
- [ControlNet for SDXL](https://stable-diffusion-art.com/controlnet-sdxl/) - Setup and usage guide
- [ControlNet SDXL on HuggingFace](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl) - API documentation
- [Synthetic Data for Computer Vision - Edge AI](https://www.edge-ai-vision.com/2025/07/synthetic-data-for-computer-vision/) - Overview article

### Hardware and Optimization
- [SDXL 12GB VRAM Optimization](https://itctshop.com/sdxl-12gb-vram-optimization/) - RTX 3060 specific
- [2025 GPU Guide for ComfyUI](https://www.promptus.ai/blog/2025-gpu-guide-for-comfyui) - Hardware recommendations
- [RTX 3060 12GB vs RTX 4060 8GB for AI](https://www.bestgpusforai.com/gpu-comparison/3060-vs-4060) - VRAM advantage analysis
- [How to run Flux AI with low VRAM](https://stable-diffusion-art.com/flux-forge/) - Flux optimization

### Model Comparisons
- [Midjourney vs DALL-E vs Stable Diffusion vs Flux 2026](https://freeacademy.ai/blog/midjourney-vs-dalle-vs-stable-diffusion-vs-flux-comparison-2026) - Full comparison
- [The 9 Best AI Image Generation Models in 2026](https://www.gradually.ai/en/ai-image-models/) - Current landscape
- [DALL-E vs Midjourney vs Stable Diffusion 2026](https://aloa.co/ai/comparisons/ai-image-comparison/dalle-vs-midjourney-vs-stable-diffusion) - Feature comparison

### ComfyUI Pipelines
- [ComfyUI Batch Processing Guide 2026](https://apatero.com/blog/comfyui-batch-processing-workflow-automation-2026) - Automation guide
- [ComfyUI-DataSet nodes](https://github.com/daxcay/ComfyUI-DataSet) - Dataset preparation tools
- [Batch Process 1000+ Images ComfyUI](https://apatero.com/blog/batch-process-1000-images-comfyui-guide-2025) - Scaling generation

### Checkpoints
- [epiCRealism XL on CivitAI](https://civitai.com/models/277058/epicrealism-xl)
- [Halcyon SDXL Photorealism on CivitAI](https://civitai.com/models/299933/halcyon-sdxl-photorealism)
- [CyberRealistic XL on CivitAI](https://civitai.com/models/312530/cyberrealistic-xl)
