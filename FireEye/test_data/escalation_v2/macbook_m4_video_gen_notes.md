# Wan2.1 Video Generation on Apple Silicon M4 (32GB Unified Memory)

Research date: 2026-03-03

## Executive Summary

Wan2.1 (and its successor Wan2.2) CAN run on Apple Silicon M4 with 32GB unified memory.
There are multiple working approaches, none of which require WanGP (NVIDIA-only).

The most practical options ranked by reliability:

1. **Wan2.1-Mac fork (PyTorch + MPS)** -- Best documented, tested on 32GB M4
2. **ComfyUI + GGUF quantized models** -- GUI-based, easiest setup
3. **Wan2.2-Mac fork (PyTorch + MPS)** -- Newer model, same approach
4. **Wan2.2-mlx (Pure MLX)** -- Experimental, removes PyTorch dependency entirely
5. **HuggingFace Diffusers WanPipeline** -- NOT reliably working on MPS yet

**Important caveat**: Generation is SLOW compared to NVIDIA GPUs. Expect 12-20 minutes
for a short 480p clip with the 1.3B model, and hours for higher quality settings.

---

## Approach 1: Wan2.1-Mac Fork (RECOMMENDED for 32GB M4)

**Repository**: https://github.com/HighDoping/Wan2.1-Mac
(Alternative fork: https://github.com/R3D347HR4Y/Wan2.1-Mac)

**Status**: WORKING. Tested on 32GB M4 Mac Mini with confirmed benchmarks.

### What it does differently from upstream Wan2.1

- Models load only when needed (T5, then diffusion model, then VAE) instead of all at once
- Models are deleted from memory immediately after use (crucial for unified memory)
- VAE tiling to reduce memory footprint
- Quantized T5 encoder support (via llama.cpp GGUF)
- Mixed precision for MPS acceleration

### Installation

```bash
# Clone the Mac fork
git clone https://github.com/HighDoping/Wan2.1-Mac.git
cd Wan2.1-Mac

# Install dependencies via Poetry
poetry install

# Install llama.cpp (needed for quantized T5)
brew install llama.cpp

# Download the 1.3B model (~5GB)
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

# Download quantized T5 encoder (saves significant memory)
huggingface-cli download HighDoping/umt5-xxl-encode-gguf --local-dir ./Wan2.1-T2V-1.3B
```

### Usage (Text-to-Video, 1.3B model)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --frame_num 17 \
  --sample_steps 25 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --tile_size 256 \
  --offload_model True \
  --t5_quant \
  --device mps \
  --prompt "A cat walks on the grass, realistic style, high quality" \
  --save_file output.mp4
```

### Performance on 32GB M4 Mac Mini

| Task       | Model | Frames | Steps | Time      | Memory   |
|------------|-------|--------|-------|-----------|----------|
| T2V        | 1.3B  | 17     | 25    | ~12 min   | ~10GB video gen + ~12GB VAE |
| T2V        | 1.3B  | 45     | 50    | ~1h 20m   | Similar  |
| I2V        | 14B   | 5      | 2     | ~13 min   | Requires disk offload |
| VACE       | 1.3B  | 17     | 50    | ~54 min   | Standard |

The 1.3B model at 480p with 17 frames fits comfortably in 32GB without swap.
The 14B model requires `--disk_offload` and `--mps_ram 10GB` flags.

### Key flags for 32GB M4

- `--offload_model True` -- Essential. Loads/unloads models sequentially.
- `--t5_quant` -- Uses quantized T5 encoder, saves ~10GB+ memory.
- `--tile_size 256` -- VAE processes video in tiles, reduces peak memory.
- `--device mps` -- Uses Metal Performance Shaders for GPU acceleration.
- `--frame_num 17` -- 17 frames = ~1 second of video at 16fps. Keep low on 32GB.
- `--disk_offload` -- Only needed for 14B model on 32GB. Creates ~60GB temp cache.

---

## Approach 2: ComfyUI + GGUF Models

**Status**: WORKING. Popular approach for Apple Silicon users.

### Why GGUF and not safetensors

The standard safetensor model files trigger `Float8_e4m3fn` / `float16 vs float32` errors
on the MPS backend. GGUF quantized models avoid these dtype issues entirely.

### Setup

1. Install ComfyUI following https://docs.comfy.org (Mac instructions available)
2. Install the ComfyUI-GGUF custom node (search "gguf" in Node Library)
3. Download GGUF model files:
   - Diffusion model: `city96/Wan2.1-T2V-14B-gguf` (pick Q4_K_M or Q8 for 32GB)
   - Text encoder: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
   - VAE: `wan_2.1_vae.safetensors`
4. Place files in the appropriate ComfyUI model directories
5. Load the Wan2.1 workflow template from ComfyUI menu

### Important ComfyUI settings for Apple Silicon

- Use the **Euler** sampler with **normal** scheduler (other combos may produce blurry output)
- Start at low resolution (320x320) to confirm it runs, then scale up
- With 32GB RAM, Q8 quantization is recommended for better quality
- If you get black video or KSampler stalls, reduce resolution or frame count

### Performance

- ~5 minutes for 2 seconds of low-resolution video on M4 Pro (14B Q4 GGUF)
- Quality is lower than full-precision models but usable for testing

### Model files (HuggingFace)

- https://huggingface.co/city96/Wan2.1-T2V-14B-gguf
- Workflow examples: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged

---

## Approach 3: Wan2.2-Mac Fork (Newer Model)

**Repository**: https://github.com/HighDoping/Wan2.2-Mac

**Status**: WORKING. Same approach as Wan2.1-Mac but for the newer Wan2.2 model family.

### Key differences from Wan2.1-Mac

- Supports TI2V-5B (Text+Image to Video, 5B params) -- a newer, more capable model
- Uses bf16 precision (note: requires `PYTORCH_ENABLE_MPS_FALLBACK=1`)
- Does NOT support the A14B (14B MoE) model on 32GB -- too large

### Installation

```bash
git clone https://github.com/HighDoping/Wan2.2-Mac.git
cd Wan2.2-Mac
poetry install

brew install llama.cpp

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
huggingface-cli download HighDoping/umt5-xxl-encode-gguf --local-dir ./Wan2.2-TI2V-5B
```

### Usage

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python generate.py \
  --task ti2v-5B \
  --size "1280*704" \
  --frame_num 41 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_quant \
  --device mps \
  --prompt "A cinematic cat walking through a garden" \
  --save_file output.mp4
```

### Performance on 32GB M4 Mac Mini

| Task | Frames | Time        |
|------|--------|-------------|
| T2V  | 41     | ~1h 37m     |
| I2V  | 25     | ~47m        |

---

## Approach 4: Wan2.2-mlx (Pure MLX, Experimental)

**Repository**: https://github.com/osama-ata/Wan2.2-mlx

**Status**: EXPERIMENTAL. Full MLX port removing all PyTorch dependencies.

This is a complete rewrite using Apple's MLX framework instead of PyTorch. All model code,
training, and inference use MLX APIs and unified memory natively.

### Installation

```bash
git clone https://github.com/osama-ata/Wan2.2-mlx.git
cd Wan2.2
uv pip install -e .
```

### Usage

```bash
uv run python generate.py -- --task t2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --prompt "A cinematic cat boxing match."
```

### Limitations

- No multi-GPU/distributed inference
- Less tested than the PyTorch MPS forks
- May require model weight conversion
- No published benchmarks for 32GB M4 specifically
- The A14B model may not fit in 32GB (it is a 14B MoE model)

### When to consider this

- If you want to avoid PyTorch entirely
- If MLX ecosystem matures and provides better performance than MPS
- For future-proofing (Apple is investing heavily in MLX)

---

## Approach 5: HuggingFace Diffusers WanPipeline (NOT RECOMMENDED on MPS)

**Status**: NOT RELIABLY WORKING on Apple Silicon as of March 2026.

### The problem

The standard diffusers `WanPipeline` uses `bfloat16` by default, which MPS does not support.
Even switching to `float16` or `float32` fails with `MLIR pass manager failed` errors due to
Conv3D operations that MPS cannot handle properly.

### What happens if you try

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float32)
pipe.to("mps")  # <-- This will likely fail or produce errors

# Common errors:
# - "MLIR pass manager failed" (Metal shader compilation failure)
# - "Conv3D is not supported on MPS" (PyTorch MPS limitation)
# - "does not have support for that dtype" (bfloat16/float8 issues)
```

### Root causes

1. PyTorch MPS backend does not support Conv3D / ConvTranspose3D in all dtypes
2. bfloat16 is not supported on MPS at all
3. The VAE decoder uses 3D convolutions extensively
4. Even with `PYTORCH_ENABLE_MPS_FALLBACK=1`, the MLIR compilation can fail

### Potential future fix

- A library called `mps-conv3d` has been proposed to patch 3D convolution to native Metal
  kernels, but it is not mature enough for production use
- PyTorch nightly builds are gradually adding Conv3D MPS support
- This may work in the future as PyTorch MPS support improves

### Workaround: CPU-only (very slow)

```python
pipe.to("cpu")  # Works but extremely slow (hours for a single video)
```

---

## Approach 6: Other Notable Tools

### mlx-video (https://github.com/Blaizzy/mlx-video)

- MLX-native video generation package for Apple Silicon
- Currently only supports LTX-2 (not Wan2.1/2.2)
- Worth monitoring for future Wan model support

### kennycason's macOS guide

- Blog post: https://kennycason.com/posts/2025-05-20-wan2.1-on-macos.html
- Tested on M4 Max 128GB (not representative of 32GB systems)
- Used ~100GB RAM with 14B model -- NOT feasible on 32GB
- The 1.3B model is the only viable option for 32GB

---

## Recommendations for 32GB M4 MacBook

### Best option: Wan2.1-Mac fork with 1.3B model

This is the most tested, best documented approach for 32GB Apple Silicon.

Key settings for 32GB:
- Use the **1.3B** model (not 14B)
- Resolution: **832x480** (480p)
- Frame count: **17** (1 second) for quick tests, up to **45** for longer clips
- Sample steps: **25** for speed, **50** for quality
- Always use `--offload_model True --t5_quant --tile_size 256`

### What to expect

- 480p, 17 frames, 25 steps: ~12 minutes
- 480p, 45 frames, 50 steps: ~1.5 hours
- Quality is decent for the 1.3B model but noticeably below the 14B model
- Text rendering in video is poor at 1.3B scale

### What is NOT feasible on 32GB

- Running the 14B model without disk offload (needs 60GB+ temp storage)
- 720p generation with 1.3B (undertrained at that resolution, unstable results)
- Real-time or near-real-time generation (minimum ~12 minutes per clip)
- Batch generation of many videos (sequential only, no parallelism)

---

## Quick Reference: Minimal Test Script (Wan2.1-Mac)

```bash
# -- SETUP (one-time) --
git clone https://github.com/HighDoping/Wan2.1-Mac.git
cd Wan2.1-Mac
poetry install
brew install llama.cpp
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download HighDoping/umt5-xxl-encode-gguf --local-dir ./Wan2.1-T2V-1.3B

# -- GENERATE (each time) --
export PYTORCH_ENABLE_MPS_FALLBACK=1
python generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --frame_num 17 \
  --sample_steps 25 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --tile_size 256 \
  --offload_model True \
  --t5_quant \
  --device mps \
  --prompt "A golden retriever running on a beach at sunset, cinematic" \
  --save_file test_output.mp4
```

Expected: ~12 minutes, output at 832x480, 17 frames (~1 second at 16fps).

---

## Sources

- https://github.com/HighDoping/Wan2.1-Mac
- https://github.com/R3D347HR4Y/Wan2.1-Mac
- https://github.com/HighDoping/Wan2.2-Mac
- https://github.com/osama-ata/Wan2.2-mlx
- https://github.com/Blaizzy/mlx-video
- https://github.com/Wan-Video/Wan2.1/issues/14
- https://github.com/Wan-Video/Wan2.1/issues/175
- https://github.com/Wan-Video/Wan2.1/issues/208
- https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/discussions/6
- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- https://kennycason.com/posts/2025-05-20-wan2.1-on-macos.html
- https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.1/wan2-1-video-model
