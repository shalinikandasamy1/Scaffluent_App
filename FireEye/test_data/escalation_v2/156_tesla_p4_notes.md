# Tesla P4 (sm_61) PyTorch Compatibility Fix for WanGP

**Machine:** 192.168.50.156
**GPU:** Tesla P4 (Pascal architecture, CUDA compute capability sm_61)
**NVIDIA Driver:** 580.126.09 (supports CUDA 13.0)
**Date:** 2026-03-03

## Problem

WanGP shipped with PyTorch 2.7.0+cu128, which only supports:
```
sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120
```

The Tesla P4 requires sm_61, so any CUDA kernel execution failed with:
```
Tesla P4 with CUDA capability sm_61 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120.
```

## Root Cause

PyTorch's CUDA 12.8 builds dropped support for Pascal (sm_60/sm_61) and older architectures.
However, the CUDA 12.6 builds of the same PyTorch version still include these architectures.

This is documented in PyTorch's deprecation plan:
- CUDA 12.6 builds: sm_61 still supported (through PyTorch 2.8.x)
- CUDA 12.8+ builds: sm_61 dropped
- CUDA 13.0: Pascal will be fully deprecated

## Fix Applied

Changed from `cu128` to `cu126` build of the **same PyTorch version** (2.7.0):

```bash
cd /home/evnchn/pinokio/api/wan.git/app
./env/bin/pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
```

### Before (broken)
- `torch==2.7.0+cu128`
- `torchvision==0.22.0+cu128`
- `torchaudio==2.7.0+cu128`
- Supported archs: `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`

### After (working)
- `torch==2.7.0+cu126`
- `torchvision==0.22.0+cu126`
- `torchaudio==2.7.0+cu126`
- Supported archs: `sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90`

## Verification

All tests passed after the change:

1. **GPU detection:** Tesla P4 detected with compute capability (6, 1)
2. **Tensor operations:** FP32, FP16, and BF16 matmul all work on GPU
3. **WanGP imports:** All critical modules import successfully:
   - torch, diffusers, transformers, accelerate, mmgp, gradio, opencv
   - All shared.* WanGP modules (asyncio_utils, match_archi, attention, utils, audio_video, loras_migration)
4. **Attention modes:** sdpa, auto, flash, xformers all listed as available

## Known Limitations

### Flash Attention (runtime, not import-time)
- `flash_attn 2.7.4+cu128torch2.7` still installed (imports OK)
- Flash Attention requires sm_80+ (Ampere) at runtime, so it will NOT work on Tesla P4
- WanGP should be configured to use `sdpa` or `auto` attention mode instead of `flash`
- This is a hardware limitation of Pascal GPUs, not a software bug

### torchcodec (pre-existing issue)
- `torchcodec 0.10.0` fails to load due to missing system FFmpeg libraries
- This was broken BEFORE the PyTorch change (FFmpeg is not installed on this machine)
- Not used directly by WanGP code; only referenced in torchvision/transformers internals
- Non-blocking for WanGP operation

### GPU Memory
- Tesla P4 has only 8 GB VRAM
- WanGP video generation models typically require 10-24 GB
- mmgp (Memory Management for GPU Poor) may help with offloading, but very large models
  may still not fit
- Recommend using the smallest model profiles and lowest resolution settings

## Revert Instructions

If this change causes issues, revert to the original PyTorch:

```bash
cd /home/evnchn/pinokio/api/wan.git/app
./env/bin/pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

A full pip freeze backup was saved at `/tmp/wangp_pip_freeze_backup.txt` on the remote machine.

## Future Considerations

- When WanGP updates PyTorch, ensure the `cu126` variant is used instead of `cu128`
- Monitor PyTorch 2.8.x and 2.9.x release notes for Pascal deprecation timeline
- Pascal support is expected to be fully removed with CUDA 13.0 toolkit builds
- Long-term, the Tesla P4 will become unsupported by modern PyTorch releases
