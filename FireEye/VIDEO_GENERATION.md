# Video Generation for FireEye Test Data

## Status: CONCLUDED

Video generation for escalation scenarios has been paused. The available hardware (RTX 3060 12GB, Tesla P4 8GB) cannot run models powerful enough to produce temporally coherent fire escalation in video form. The **OpenRouter chain-edited frame sequences** proved to be the winning approach — no video generation needed.

**What worked:** OpenRouter image editing (Gemini 3.1 Flash Image) to iteratively escalate fire in still frames. See `test_data/escalation_v2/openrouter_frames/`.

**What to revisit if restarting:** A GPU with 24GB+ VRAM (e.g., RTX 4090, A5000) could run FLF2V 14B without INT8 quantization, which would likely produce proper start-to-end frame interpolation. The seed image pairs in `test_data/escalation_v2/seed_images/` are ready to use.

## Overview

FireEye currently processes **still images only**. These generated videos serve as source material for extracting frames that test the pipeline's ability to classify escalating fire scenarios — where early frames should classify as safe/low and later frames as high/critical.

## OpenRouter Budget

| Metric | Value |
|--------|-------|
| Total credits | $70.00 |
| Usage before video gen work | $0.178 |
| Seed image generation (10 imgs) | ~$0.14 (FLUX Klein) + ~$0.20 (Gemini 3.1 edit) |
| OpenRouter escalation frames (45 imgs) | ~$2.87 (5 FLUX + 40 Gemini edits) |
| **Estimated remaining** | **~$66.61** |
| Snapshot date | 2026-03-03 |

Check balance: `curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $FIREEYE_OPENROUTER_API_KEY"`

## Infrastructure

| Resource | Details |
|----------|---------|
| GPU machine .153 | `192.168.50.153` (RTX 3060, 12GB VRAM, 32GB RAM) — main gen box |
| GPU machine .156 | `192.168.50.156` (Tesla P4, 8GB VRAM, 16GB RAM) — 1.3B models only, needs `--attention sdpa` |
| SSH | `sshpass -p 'insecure' ssh -o StrictHostKeyChecking=no evnchn@192.168.50.{153,156}` |
| WanGP install | `/home/evnchn/pinokio/api/wan.git/app/` (both machines) |
| WanGP version | v10.952 |
| Python env | `/home/evnchn/pinokio/api/wan.git/app/env/bin/python` |
| Gradio UI | Binds to `127.0.0.1:42003` (needs SSH tunnel) |
| SSH tunnel | `sshpass -p 'insecure' ssh -o StrictHostKeyChecking=no -f -N -L 42003:127.0.0.1:42003 evnchn@192.168.50.153` |
| Image gen API | OpenRouter (key in `.env` as `FIREEYE_OPENROUTER_API_KEY`) |

## Generated Video Inventory

All videos: 832x480 (16:9), 81 frames (5.1s at 16fps), 30 inference steps.

### Batch 1: Dangerous Scenes (`test_data/dangerous_videos/`)

| File | Prompt | Status |
|------|--------|--------|
| 01_campfire.mp4 | Campfire with flickering flames and sparks | Done |
| 02_kitchen_fire.mp4 | Kitchen stovetop grease fire | Done |
| 03_building_fire.mp4 | Building engulfed in flames | Done |
| 04_house_fire.mp4 | House fire from windows/roof | Done |
| 05_forest_wildfire.mp4 | Forest wildfire spreading | Done |

### Batch 2: Safe Scenes (`test_data/safe_videos/`)

| File | Prompt | Status |
|------|--------|--------|
| 01_meeting_room.mp4 | Empty meeting room | Done |
| 02_parking_lot.mp4 | Parking lot with cars | Done |
| 03_river_landscape.mp4 | Peaceful river scene | Done |
| 04_warehouse.mp4 | Warehouse interior | Done |
| 05_green_field.mp4 | Green field countryside | Done |

### Batch 3: Escalation Scenes (`test_data/escalation_videos/`)

Generated with t2v_1.3B. QA results: **only 02 and 05 are usable**. The 1.3B t2v model struggles with temporal progression — it renders the "most dramatic interpretation" across all frames instead of showing a calm-to-fire transition.

| File | Prompt | QA |
|------|--------|----|
| 01_campfire_spreads.mp4 | Campfire embers blown by wind catching dry leaves | Bad — already a massive blaze from frame 1 |
| 02_welding_ignition.mp4 | Welder sparks igniting wooden scaffolding | Good — clear escalation arc |
| 03_controlled_burn_escapes.mp4 | Agricultural burn jumping firebreak | Bad — fire huge from start |
| 04_bbq_flareup.mp4 | BBQ grease flare spreading to furniture | Bad — static large grill fire |
| 05_fireplace_escape.mp4 | Fireplace embers scattering onto floor | Okay — spark dispersal visible |


## Escalation V2: Image-to-Video Pipeline (CONCLUDED)

Text-to-video cannot reliably produce escalation sequences. The solution is a two-stage pipeline:

1. **Generate seed images** via OpenRouter (start frame + end frame)
2. **Generate video** via WanGP using models that condition on input images

### Stage 1: Seed Image Generation (DONE)

Script: `test_data/escalation_v2/generate_seeds.py`

**Workflow used:**
1. Generate "calm start" images with **FLUX.2 Klein** (`black-forest-labs/flux.2-klein-4b`, $0.014/img)
2. Edit start images into "fire end" variants with **Gemini 3.1 Flash Image** (`google/gemini-3.1-flash-image-preview`) — preserves scene composition/camera angle

**Seed images (local):** `test_data/escalation_v2/seed_images/`
**Seed images (remote):** `/tmp/fireeye_seeds/` (includes `_720p` resized versions)

| Scene | Start Image | End Image | Continuity QA |
|-------|-------------|-----------|---------------|
| 01_campfire | Controlled campfire in stone ring | Fire spread beyond ring to grass | Excellent |
| 02_welding | Welder on wooden scaffolding | Scaffolding on fire, welder retreating | Good |
| 03_agri_burn | Controlled field burn with firebreak | Fire jumped break, massive spread | Good |
| 04_bbq | Normal BBQ on deck with furniture | Grease fire spread to furniture | Excellent |
| 05_fireplace | Cozy fireplace, carpet, blanket | Embers on carpet, blanket smoldering | Excellent |

### Stage 2: Video Generation — Model Attempts

#### Attempt 1: FLF2V 14B INT8 — FAILED (corruption)

FLF2V 14B (First+Last Frame to Video) is the ideal model — it interpolates between start and end images. However, on RTX 3060 12GB with MMGP Profile 4 + INT8 quantization, it produces **severe visual corruption** (colorful garbled pixels) in the second half of all videos. The first ~50-60% of frames look fine, then the output degrades into noise.

- 81-frame attempt: corruption starts at frame ~60
- 49-frame attempt: corruption starts at frame ~24
- Corruption is proportional — always around 50-75% through the video
- Likely cause: numerical instability from INT8 quantization + aggressive CPU offloading
- Videos saved in `test_data/escalation_v2/videos/01_campfire.mp4` and `01_campfire_v2.mp4` (kept as artifacts)

**Researched fixes to try (from WanGP GitHub issues and docs):**
1. **Force SDPA attention** — add `--attention sdpa` to CLI. Sage Attention causes artifacts on RTX 30xx.
2. **Try Profile 2** (HighRAM_LowVRAM) instead of 4 — needs 48GB+ RAM (we have 32GB, borderline).
3. **Set VAE tiling to 256** explicitly, not auto.
4. **Disable "Compile Transformer Model"** in settings.
5. **Restart app between generations** — corruption can accumulate in memory state.
6. **Use mbf16 INT8 checkpoint** (not mfp16) — BFloat16's wider dynamic range is more numerically stable.
7. **Fallback**: Generate full 81 frames, trim to first ~40 frames (before corruption starts) with ffmpeg.

#### Attempt 2: VACE 1.3B — DONE (clean output, no escalation)

VACE 1.3B uses the existing t2v_1.3B base + a 1.4GB module. Produces **clean, artifact-free** output. However, output didn't match seed images well and showed no visible fire escalation — fire stayed constant throughout.

5 videos completed in 1h 20m. Downloaded to `test_data/escalation_v2/videos/vace/`.

#### Attempt 3: FLF2V SDPA — NO CORRUPTION but no escalation

`--attention sdpa` fix **eliminates the corruption** — all 81 frames clean. However, the model doesn't interpolate to the end image; fire stays at start level throughout. The FLF2V conditioning on the end frame may not work well with MMGP offloading.

5-video batch was run but produced same no-escalation result. GPU work ceased.

#### Alternative 1: OpenRouter Chain-Edited Frames — SUCCESS

**Best approach found.** Script: `test_data/escalation_v2/generate_escalation_frames.py`

Instead of video generation, directly generates 9-frame escalation sequences using iterative image editing:
1. Frame 1: Generate calm scene with **FLUX.2 Klein** ($0.014)
2. Frames 2-9: Each frame edits the previous with **Gemini 3.1 Flash Image** (~$0.07/edit), progressively adding fire

This preserves scene composition perfectly and produces clear escalation arcs.

**Output:** `test_data/escalation_v2/openrouter_frames/` — 5 scenarios × 9 frames = 45 frames, 100MB total
**Cost:** ~$2.87 (well under $5 budget)
**Quality:** Excellent — photorealistic, consistent composition, clear escalation from calm to catastrophic

| Scenario | Frame 1 → Frame 9 |
|----------|-------------------|
| campfire_spread | Calm campsite → Full forest fire |
| welding_sparks | Welder working → Workshop engulfed |
| agricultural_burn | Controlled burn → Jumped firebreak |
| bbq_grease_fire | Patio BBQ → Deck/fence fire |
| fireplace_embers | Cozy living room → Room engulfed |

#### Alternative 2: Fire Simulator — DONE

Programmatic fire simulation at `test_data/fire_simulator/`. Cellular automaton model with:
- 3 scene types: indoor, outdoor, urban
- Configurable wind, fuel density, ignition point
- Deterministic (seeded), parametric, unlimited generation
- Pure Python (no external deps), renders via custom PNG encoder

Generated 90 sample frames (30 per scenario) in `test_data/fire_simulator/output/`.

Run: `/usr/bin/python3 test_data/fire_simulator/generate_dataset.py --all`

### OpenRouter Image Gen Models Reference

| Model | Model ID | Cost/image (1K) | Editing? | Notes |
|-------|----------|-----------------|----------|-------|
| FLUX.2 Klein 4B | `black-forest-labs/flux.2-klein-4b` | **$0.014** | Yes | Cheapest. ~71 images per $1. Fast. |
| Riverflow V2 Fast | `sourceful/riverflow-v2-fast` | **$0.02** | Yes | SOTA gen+edit. Low latency. |
| FLUX.2 Pro | `black-forest-labs/flux.2-pro` | **$0.03** gen, +$0.015/MP edit | Yes | Best for gen→edit workflow. |
| Gemini 3.1 Flash Image | `google/gemini-3.1-flash-image-preview` | **~$0.04-0.10** | Yes | Best for editing existing images. Preserves composition. |
| Gemini 2.5 Flash Image | `google/gemini-2.5-flash-image-preview` | **~$0.04** | Yes | Older version. Multi-turn edits. |

### Stage 2: Video Generation (WanGP)

Models available on the RTX 3060 12GB that handle temporal progression:

| Model | JSON `model_type` | Approach | VRAM | Speed | Checkpoint |
|-------|-------------------|----------|------|-------|------------|
| **FLF2V 720p** | `flf2v_720p` | Start image + end image → interpolated video | ~12 GB | ~15 min | Auto-downloads ~14 GB |
| **SkyReels V2 DF 1.3B** | `sky_df_1.3B` | Diffusion forcing — autoregressive, each frame independently conditioned | ~6 GB | ~10 min | Auto-downloads ~2.7 GB |
| **VACE 1.3B** | `vace_1.3B` | Reference frames at specific positions, model fills gaps | ~6 GB | ~10 min | Auto-downloads ~2.7 GB |
| **I2V 14B** | `i2v` | Single start image + text prompt → animated video | ~12 GB | ~15 min | Auto-downloads ~14 GB |
| **FusionX T2V** | `t2v_fusionix` | Cinema-grade t2v in ~8 steps (better than plain t2v) | ~12 GB | ~4 min | Auto-downloads |

**Best choice for escalation: FLF2V** — guarantees progression by defining both endpoints. **However**, FLF2V is 14B only and corrupts under INT8 + MMGP Profile 4 on 12GB VRAM. See "Model Attempts" above.

### Combined Workflow Example

```
1. OpenRouter: Generate "calm campfire with stone ring" image  →  start.png
2. OpenRouter: Edit start.png to "campfire spreading to surrounding dry grass"  →  end.png
3. WanGP FLF2V: start.png + end.png  →  campfire_escalation.mp4
4. ffmpeg: Extract frames from video  →  frame_001.jpg ... frame_081.jpg
5. FireEye: Analyze each frame independently, observe risk level progression
```


## How to Generate Videos via CLI

Create a JSON settings file. For t2v:
```json
[{"model_type": "t2v_1.3B", "prompt": "Your prompt here"}]
```

For FLF2V (start + end image):
```json
[{
  "model_type": "flf2v_720p",
  "prompt": "Description of the transition",
  "image_start": "/absolute/path/to/start.png",
  "image_end": "/absolute/path/to/end.png"
}]
```

On the remote machine (kill WanGP server first if running):
```bash
# Kill running server
pkill -f "wgp.py"

# Validate
cd /home/evnchn/pinokio/api/wan.git/app
./env/bin/python wgp.py --process /path/to/tasks.json --dry-run

# Generate (use nohup so it survives SSH disconnect)
nohup ./env/bin/python wgp.py --process /path/to/tasks.json --output-dir /tmp/output > /tmp/gen.log 2>&1 &
```

Timing: t2v_1.3B ~11 min/video, FLF2V 14B ~65 min/video on RTX 3060.

## How to Restart WanGP Server

```bash
sshpass -p 'insecure' ssh -o StrictHostKeyChecking=no evnchn@192.168.50.153 \
  "cd /home/evnchn/pinokio/api/wan.git/app && \
   nohup /home/evnchn/pinokio/api/wan.git/app/env/bin/python wgp.py --multiple-images > /dev/null 2>&1 &"
```

Or restart via Pinokio UI.

## Downloaded Checkpoints on Remote Machine

| File | Size | Model |
|------|------|-------|
| `wan2.1_text2video_1.3B_mbf16.safetensors` | 2.7 GB | T2V 1.3B (bf16) |
| `wan2.1_text2video_14B_quanto_mbf16_int8.safetensors` | 14 GB | T2V 14B (int8) |
| `wan2.1_FLF2V_720p_14B_quanto_mbf16_int8.safetensors` | 16 GB | FLF2V 720p 14B (int8) — corrupts under Profile 4 |
| `wan2.1_Vace_1_3B_module.safetensors` | 1.4 GB | VACE 1.3B module (uses t2v_1.3B base) |
| `Wan2.1_VAE.safetensors` | 485 MB | Standard VAE |
| `umt5-xxl/...int8.safetensors` | — | Text encoder (int8) |

Other model checkpoints (SkyReels, I2V, etc.) will auto-download on first use.

## WanGP Config

Current config (`wgp_config.json`): Profile 4 (LowRAM_LowVRAM), INT8 quantization, last model `t2v_1.3B`.

148+ model definitions available in `defaults/` directory. Key families: Wan2.1, Wan2.2, Hunyuan Video, LTX Video, Flux, Kandinsky 5, and more.
