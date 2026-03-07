# Image-to-Image Editing for Fire Severity Variations

Research date: 2026-03-08

## Problem Statement

Our dataset has 29 real HK construction site photos (17 scaffold-net building shots, 12 TKO active fire incident). Feedback indicates too much "fire already burning hot" -- we need calm/low-severity versions to cover the full severity spectrum: clean site, minor hazard, wisps of smoke, active fire, aftermath/damage.

---

## 1. InstructPix2Pix and Successors

### Original InstructPix2Pix (Brooks et al., CVPR 2023)
- Conditional diffusion model trained on synthetic instruction-edit pairs (GPT-3 + Stable Diffusion generated).
- Accepts natural language instructions like "add smoke to the building" or "remove the fire".
- **Controllability**: Two guidance scales (text guidance + image guidance) let you trade off edit strength vs. fidelity. Low text guidance = subtle edits (good for "add wisps of smoke"); high = dramatic changes.
- **Limitations**: Based on SD 1.5 resolution (512x512). Can struggle with spatial precision -- may add smoke in wrong areas. Fire/smoke are not well-represented in its training data, so results can look painterly rather than photorealistic.

### InstructAny2Pix (NAACL 2025)
- Extends InstructPix2Pix with multi-modal LLM backbone for complex multi-object edits.
- Supports interleaved text + reference images as prompts.
- Better at following complex instructions but still limited by base model resolution.

### MagicBrush Dataset
- First large-scale manually-annotated instruction-guided editing dataset (10K triples).
- Fine-tuning InstructPix2Pix on MagicBrush significantly improves edit quality.
- Relevant because it demonstrates that domain-specific fine-tuning helps enormously.

### TurboEdit (SIGGRAPH Asia 2024)
- Sub-second editing (3 diffusion steps, 0.32s on A5000).
- Encoder-based inversion + LLM-driven attribute modification.
- Fast enough for interactive experimentation but quality is lower than full-step methods.

### Assessment for Our Use Case
- **Adding wisps of smoke to calm scene**: Moderately feasible. Results will need cherry-picking. Prompt like "add thin wisps of grey smoke rising from the scaffolding" with low text guidance (5-7) and high image guidance (1.5+).
- **Removing fire from burning scene**: Poor. InstructPix2Pix struggles with large-region removal -- tends to leave artifacts or hallucinate wrong textures underneath.
- **Overall quality**: Acceptable for data augmentation (not for publication-quality images). Expect ~40% usable outputs per batch.

---

## 2. ControlNet for Structural Preservation

### Core Concept
ControlNet adds spatial conditioning (depth maps, edge maps, pose) to diffusion models, preserving the geometric structure of the original image while allowing content changes.

### Relevant Control Types
- **Depth-conditioned**: Extract depth map from original photo (MiDaS/ZoeDepth), use it to constrain generation. Building geometry preserved perfectly while surface appearance changes. Best for "same building, different fire state".
- **Canny/HED edge**: Preserves architectural edges and scaffolding lines while allowing texture/atmosphere changes.
- **MLSD (line segments)**: Good specifically for architectural/construction scenes with many straight lines.

### Workflow: ControlNet + Inpainting
1. Extract depth map from original construction photo.
2. Create mask over fire/smoke region (or area where fire should appear).
3. Use depth-conditioned ControlNet with inpainting model.
4. Prompt for desired severity level.
5. The depth map ensures buildings, scaffolding, and ground stay geometrically correct.

### VRAM Impact
- ControlNet adds 2-3 GB per model on top of base model.
- SDXL (8 GB) + ControlNet (3 GB) = ~11 GB -- tight but feasible on RTX 3060 12GB.
- Using FP8 quantization (--fp8_e4m3fn-unet) is essential for stability.
- FLUX + ControlNet would exceed 12 GB even with quantization -- not recommended for our hardware.

### Assessment
- **Best approach for maintaining site identity.** The construction site will look like the same site across all severity levels.
- Depth + Canny dual-conditioning gives excellent structural preservation.
- Recommended as the backbone of our workflow.

---

## 3. Inpainting Approaches

### Strategy A: Remove Fire (High -> Low Severity)
- Mask the fire/smoke region in the TKO active fire photos.
- Inpaint with prompts like "construction site scaffolding, clear sky, no fire, no smoke, normal conditions".
- **Challenge**: The inpainted region must match the surrounding construction materials, lighting, and perspective.
- Works best with dedicated inpainting models (not regular checkpoints with inpainting pipeline).

### Strategy B: Add Fire/Smoke (Low -> High Severity)
- Mask a region on the calm scaffold-net photos (e.g., lower floors, near debris).
- Inpaint with prompts like "small fire starting on construction debris, thin smoke, orange flames".
- **Advantage**: Easier than removal because you're adding detail, not reconstructing hidden structure.
- Use low denoising strength (0.4-0.6) for subtle smoke; higher (0.7-0.9) for visible flames.

### Model Options (ranked by quality for our use case)

1. **SDXL Inpainting** (diffusers/stable-diffusion-xl-1.0-inpainting)
   - 1024x1024 native resolution.
   - Good photorealism, well-supported ecosystem.
   - ~8 GB VRAM -- fits RTX 3060 12GB comfortably.
   - **Recommended starting point.**

2. **FLUX-Fill** (Black Forest Labs inpainting model)
   - Superior quality and prompt adherence.
   - 12B parameter model -- needs FP8/GGUF quantization for 12GB VRAM.
   - ~60-80 seconds per image on RTX 3060 12GB with quantized weights.
   - **Best quality but slowest on our hardware.**

3. **SD 1.5 Inpainting** (runwayml/stable-diffusion-inpainting)
   - 512x512 native resolution (too low for detail).
   - Fast and lightweight (~4 GB VRAM).
   - Useful for rapid prototyping but not final dataset images.

4. **SDXL + ControlNet Inpainting** (alimama-creative/FLUX-Controlnet-Inpainting or SD-based)
   - Combines inpainting with structural conditioning.
   - Best for preserving building geometry while editing fire regions.
   - SDXL variant fits in 12GB with FP8; FLUX variant does not.

### Practical Tips
- Inpainting prompts should describe ONLY the masked region, not the whole image.
- Format: `[effect description], [physical properties], [quality modifiers]`
  - Example: "thin grey smoke wisps, translucent, backlit by sun, photorealistic, 8k"
- Feather mask edges (5-15 px) for natural blending.
- Generate 8-10 candidates per edit, manually select best 2-3.

---

## 4. IP-Adapter / Reference-Based Generation

### How It Works
IP-Adapter uses a decoupled cross-attention mechanism to inject image features alongside text features. The reference image provides style/content guidance while text prompts control modifications.

### Three IP-Adapter Types
- **Style**: Captures color palette, lighting, atmosphere. Use a fire photo as style reference to "warm up" a calm scene.
- **Content**: Captures nearly everything visible -- architecture, objects, scenery. Use to maintain site identity.
- **Character**: Not relevant for our use case.

### Workflow for Severity Variations
1. Use a calm scaffold photo as Content reference (preserves site identity).
2. Use a fire photo as Style reference (transfers fire atmosphere/lighting).
3. Control the blend weight (0.0-1.0) to dial severity:
   - Weight 0.1-0.2: Subtle orange cast, hint of haze (pre-fire atmosphere).
   - Weight 0.3-0.5: Visible smoke, warm lighting, early fire.
   - Weight 0.6-0.8: Strong fire presence, heavy smoke.
4. Combine with text prompt for additional control.

### Assessment
- **Strength**: Excellent for atmospheric changes (lighting shifts, haze, color temperature).
- **Weakness**: Poor at placing fire in specific locations -- it's a global style transfer, not localized editing.
- **Best used in combination** with inpainting (IP-Adapter sets the mood, inpainting adds specific fire/smoke).

---

## 5. FLUX.1 Kontext -- The Most Promising New Option

### Overview (Released mid-2025)
FLUX.1 Kontext is a 12B parameter multimodal flow transformer for in-context image generation and editing. It unifies generation and editing in a single architecture via sequence concatenation of input images and text.

### Why It Stands Out for Our Use Case
- **Instruction-based editing**: Natural language instructions like "add small wisps of smoke to the left side of the scaffolding" or "remove the fire, show the building 10 minutes before the incident".
- **Reference preservation**: Specifically designed to maintain object/scene consistency across edits. Robust multi-turn editing with minimal visual drift.
- **No mask required**: Unlike inpainting, Kontext can understand spatial instructions without manual masking.
- **Multi-turn editing**: Can progressively increase severity across multiple edits of the same image.

### Hardware Feasibility
- Base model needs 24 GB VRAM.
- **NVIDIA TensorRT optimization reduces to ~7 GB VRAM** (released July 2025), making it feasible on RTX 3060 12GB.
- GGUF Q5 quantized model: ~12 GB disk, fits in 12 GB VRAM.
- Speed: ~60-80 seconds per image on RTX 3060 12GB.

### Assessment
- **Most promising single model for our workflow.** Combines the benefits of InstructPix2Pix (instruction-following), ControlNet (structure preservation), and IP-Adapter (reference consistency) in one model.
- The TensorRT-optimized version makes it practical on our hardware.
- **Recommended as primary approach** if we go the image editing route.

---

## 6. OmniGen2 -- Alternative Unified Approach

### Overview (CVPR 2025 / mid-2025 release)
OmniGen2 is a unified multimodal generation model with dual-pathway decoding. Supports precise local editing via natural language instructions without plugins or intermediate steps.

### Key Capabilities
- Arbitrarily interleaved text + image inputs.
- Subject-driven generation (use our construction site as subject reference).
- Precise local editing with natural language.
- Simpler architecture than ControlNet pipelines (just VAE + transformer).

### Assessment
- Strong alternative to FLUX Kontext but less battle-tested.
- Community tooling (ComfyUI nodes) available but less mature.
- Worth evaluating if FLUX Kontext results are unsatisfactory.

---

## 7. Temporal Simulation: The "Walk Forward/Backward in Time" Approach

### The Vision
Create a coherent sequence from a single photo: clean site -> minor hazard -> smoke -> fire -> aftermath.

### Approach A: Sequential Instruction Editing
Use FLUX Kontext or InstructPix2Pix in multi-turn mode:
1. Start with clean site photo.
2. "Add a small pile of flammable debris near the scaffolding" (hazard introduction).
3. "A thin wisp of smoke is rising from the debris pile" (ignition).
4. "The smoke is thicker, small orange flames visible at the base" (early fire).
5. "The fire has spread to the scaffold netting, heavy black smoke" (active fire).
6. "The fire is dying down, charred scaffolding, water damage visible" (aftermath).

**Advantage**: Each step preserves context from the previous one. FLUX Kontext specifically designed for this multi-turn consistency.

### Approach B: Image Editing via Video Generation (2025 Research)
- Recent work (Garibi et al., 2025) proposes treating image editing as video generation: source image is frame 0, target edit is the final frame, and the model generates smooth temporal progression between them.
- Could produce a smooth 5-10 frame sequence showing fire development.
- **Status**: Research-stage; no easy-to-use implementation available yet.

### Approach C: Video Interpolation
- Given two edited images (e.g., "no fire" and "active fire"), use video interpolation (FILM, RIFE) to generate intermediate frames.
- **Problem**: Video interpolation models interpolate motion, not physical processes. Fire progression is not simple motion -- it involves new smoke appearing, flames growing, materials changing color. Results would be unrealistic blends.
- **Verdict**: Not recommended.

### Assessment
- **Sequential instruction editing (Approach A) is the most practical path.**
- Expect to spend 5-10 minutes per severity sequence (generating candidates + manual selection).
- Each sequence produces 4-6 usable images from 1 original.

---

## 8. Synthetic Fire Data Generation (Academic Context)

### Existing Approaches in Literature
- **SYN-FIRE Dataset**: 2000 labeled images of simulated indoor industrial fires created using NVIDIA Omniverse (3D rendering). High fidelity but requires 3D scene setup.
- **Adobe Premiere compositing**: Fire/smoke assets from stock video composited onto base images. Simple but looks fake at close inspection.
- **CFD simulation**: Computational Fluid Dynamics for physically-accurate fire evolution. Overkill for our purposes and requires specialized software.
- **GAN-based augmentation**: Training GANs on fire images to generate variations. Requires large fire dataset to start with.

### Key Finding from Literature
A 2025 study (Nature Scientific Reports) found that incorporating synthetic fire data improved DiceScore by 2-16% for fire segmentation models. The effective ratio was **60% real / 40% synthetic** -- adding more synthetic data beyond 40% showed diminishing returns and could hurt generalization.

### Implication for Our Project
- We should aim for roughly **60:40 real-to-synthetic ratio**.
- With 29 real images, we should target ~19 synthetic/edited images.
- Quality matters more than quantity -- poorly edited images could teach the model wrong patterns.

---

## 9. Hardware Feasibility Summary (RTX 3060 12GB)

| Approach | VRAM Required | Time per Image | Feasible? |
|----------|--------------|----------------|-----------|
| SD 1.5 InstructPix2Pix | ~4 GB | ~5s | Yes (fast, low quality) |
| SD 1.5 Inpainting | ~4 GB | ~5s | Yes (low resolution) |
| SDXL Inpainting | ~8 GB | ~15s | Yes (recommended) |
| SDXL + ControlNet (1 model) | ~11 GB | ~20s | Yes (tight, use FP8) |
| SDXL + ControlNet (2 models) | ~14 GB | N/A | No |
| FLUX-Fill (FP8/GGUF Q5) | ~12 GB | ~70s | Yes (slow but best quality) |
| FLUX Kontext (TensorRT) | ~7 GB | ~40s | Yes (best option) |
| FLUX Kontext (GGUF Q5) | ~12 GB | ~70s | Yes |
| FLUX + ControlNet | >16 GB | N/A | No |
| OmniGen2 | ~10 GB (est.) | ~30s | Likely yes |
| IP-Adapter + SDXL | ~10 GB | ~20s | Yes |

---

## 10. Recommended Strategy

### Primary Pipeline: FLUX Kontext (TensorRT optimized)
1. Install ComfyUI + FLUX Kontext GGUF/TensorRT model.
2. For each of the 29 base photos, generate 2-4 severity variations using instruction-based editing.
3. Use sequential multi-turn editing to create coherent severity progressions.
4. Target prompts by severity level:
   - **Level 0 (Clean)**: Use as-is for scaffold-net photos; for TKO fire photos, "remove all fire and smoke, show the construction site in normal conditions before the incident".
   - **Level 1 (Hazard)**: "Add scattered flammable materials and minor debris near the scaffolding base".
   - **Level 2 (Early smoke)**: "Thin wisps of grey smoke are rising from the lower scaffolding area".
   - **Level 3 (Small fire)**: "A small fire is burning at the base of the scaffolding, orange flames visible, light smoke".
   - **Level 4 (Active fire)**: Use TKO fire photos as-is, or intensify calm photos with "heavy fire engulfing the scaffold netting, thick black smoke, bright orange flames".
   - **Level 5 (Aftermath)**: "The fire has been extinguished, charred and blackened scaffolding, water damage, debris on the ground, firefighting equipment visible".

### Fallback Pipeline: SDXL Inpainting + ControlNet
If FLUX Kontext quality is insufficient or too slow:
1. Extract depth maps from all 29 photos using MiDaS.
2. Create manual masks for fire/smoke regions (semi-automated with SAM/GroundingDINO).
3. Use SDXL inpainting with depth ControlNet conditioning.
4. Separate prompt engineering for each severity level.

### Quality Control
- Every generated image must pass manual inspection before inclusion.
- Reject images with: anatomically impossible fire behavior, wrong perspective, visible artifacts, inconsistent lighting.
- Target: 60% real (29 images) / 40% synthetic (~19 images) based on literature recommendations.
- Total target dataset: ~48 images across all severity levels.

---

## 11. Key References and Resources

### Models (HuggingFace / GitHub)
- InstructPix2Pix: https://huggingface.co/timbrooks/instruct-pix2pix
- SDXL Inpainting: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting
- FLUX Kontext dev: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- FLUX ControlNet Inpainting: https://github.com/alimama-creative/FLUX-Controlnet-Inpainting
- OmniGen2: https://github.com/VectorSpaceLab/OmniGen2
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter

### Papers
- InstructPix2Pix (Brooks et al., CVPR 2023): https://arxiv.org/abs/2211.09800
- FLUX.1 Kontext (BFL, 2025): https://arxiv.org/abs/2506.15742
- MagicBrush dataset: https://osu-nlp-group.github.io/MagicBrush/
- TurboEdit: https://arxiv.org/abs/2408.00735
- OmniGen (CVPR 2025): https://arxiv.org/abs/2409.11340
- Synthetic fire data impact (2025): https://www.nature.com/articles/s41598-025-01571-5
- Image Editing via Video Generation: https://arxiv.org/html/2411.16819v2

### Tooling
- ComfyUI (recommended frontend): https://github.com/comfyanonymous/ComfyUI
- Stable Diffusion Art guides: https://stable-diffusion-art.com/
- NVIDIA TensorRT for FLUX Kontext: https://blogs.nvidia.com/blog/rtx-ai-garage-flux-kontext-nim-tensorrt/
