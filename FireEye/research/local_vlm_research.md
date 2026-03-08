# Local Vision-Language Model Research for FireEye

**Date:** 2026-03-08
**Purpose:** Identify the best open-weight multimodal (vision+language) models that can run locally on our GPUs for fire safety image analysis.
**Hardware:** RTX 3060 12GB VRAM (primary), Tesla P4 8GB VRAM (secondary)

---

## Executive Summary

The best option for the RTX 3060 12GB is **Gemma 3 12B QAT (int4)** at ~6.6GB model weight, leaving headroom for KV cache. For the Tesla P4 8GB, **Qwen2.5-VL-3B Q4_K_M** or **Qwen3-VL-2B Q4_K_M** are the safest fits. For maximum quality with acceptable speed, **Qwen2.5-VL-7B Q4_K_M** (~4.5GB weights) can fit on both GPUs but will be slow on the Tesla P4 due to Pascal architecture limitations.

All recommended models support structured JSON output and can be run via **Ollama** or **llama.cpp**.

---

## Hardware Constraints

| GPU | VRAM | Architecture | FP16 TFLOPS | Notes |
|-----|------|-------------|-------------|-------|
| RTX 3060 | 12GB GDDR6 | Ampere (SM 86) | 12.7 | Good FP16/INT8 support, Tensor Cores |
| Tesla P4 | 8GB GDDR5X | Pascal (SM 61) | 5.5 (FP32) | No native FP16 Tensor Cores; INT8 via TensorRT only. Slow for LLM inference (~2-6x slower than modern GPUs for large models) |

**Key constraint:** The Tesla P4 is Pascal-era. It works with Ollama/llama.cpp but is significantly slower than Ampere GPUs. Models should stay well under 8GB total (weights + KV cache). Users report models >6GB become very slow on Pascal GPUs.

---

## Model Comparison Table

### Tier 1: Best Quality That Fits RTX 3060 12GB

| Model | Params | Quant | Weight Size | Total VRAM (est.) | Fits 3060? | Fits P4? | Quality | JSON Support |
|-------|--------|-------|-------------|-------------------|-----------|---------|---------|-------------|
| **Gemma 3 12B QAT** | 12B | int4 (QAT) | 6.6 GB | ~8-10 GB | YES | NO | Excellent | Yes (function calling) |
| **Qwen2.5-VL-7B** | 8.3B | Q5_K_M | ~6 GB | ~8-9 GB | YES | Marginal | Excellent | Yes (native structured output) |
| **Qwen2.5-VL-7B** | 8.3B | Q4_K_M | ~4.5 GB | ~6-7 GB | YES | Tight | Very Good | Yes |
| **Qwen3-VL-8B** | 8B | Q4_K_M | ~5 GB | ~7-8 GB | YES | NO | Excellent | Yes |
| **InternVL3.5 8B** | 8B | Q4 | ~5.7 GB | ~7-8 GB | YES | NO | Excellent | Yes |

### Tier 2: Best Quality That Fits Tesla P4 8GB

| Model | Params | Quant | Weight Size | Total VRAM (est.) | Fits 3060? | Fits P4? | Quality | JSON Support |
|-------|--------|-------|-------------|-------------------|-----------|---------|---------|-------------|
| **Qwen2.5-VL-3B** | 3.8B | Q4_K_M | ~2.5 GB | ~4-5 GB | YES | YES | Good | Yes |
| **Qwen3-VL-2B** | 2B | Q4_K_M | ~1.5 GB | ~3-4 GB | YES | YES | Decent | Yes |
| **Gemma 3 4B QAT** | 4B | int4 (QAT) | ~3.3 GB | ~5-6 GB | YES | YES | Good | Yes |
| **Phi-4 Multimodal** | 5.6B | Q4_K_M | ~4.5 GB | ~6-7 GB | YES | Tight | Good | Yes (function calling) |
| **InternVL3.5 4B** | 4B | Q4 | ~3.4 GB | ~5-6 GB | YES | YES | Good | Yes |

### Tier 3: Ultra-Small / Edge Models

| Model | Params | Quant | Weight Size | Total VRAM (est.) | Quality | Notes |
|-------|--------|-------|-------------|-------------------|---------|-------|
| **SmolVLM 2.2B** | 2.2B | FP16 | ~4.4 GB | ~5-6 GB | Moderate | Great for edge, but may lack domain knowledge |
| **SmolVLM 500M** | 500M | FP16 | ~1 GB | ~1.5 GB | Low | Too small for complex fire safety analysis |
| **MiniCPM-V 2.0** | 2B | Q4 | ~1.5 GB | ~3 GB | Moderate | On-device focused, comparable to some 8B models |

---

## Detailed Model Analysis

### 1. Gemma 3 12B QAT (int4) -- RECOMMENDED for RTX 3060

- **Source:** Google DeepMind
- **License:** Gemma open license (permissive for commercial use)
- **Why it fits:** QAT (Quantization-Aware Training) shrinks 24GB BF16 model to 6.6GB int4 with minimal quality loss. Google specifically trained quantization into the model rather than post-hoc quantizing.
- **Vision capabilities:** Image interpretation, object identification, text extraction from images, chart/diagram understanding.
- **Structured output:** Supports function calling and structured output natively.
- **How to run:** `ollama run gemma3:12b` (QAT version auto-selected) or download GGUF from `google/gemma-3-12b-it-qat-q4_0-gguf` on HuggingFace.
- **Inference speed estimate:** ~10-20 tokens/sec on RTX 3060 (with quantization overhead on Ampere ~20% slower than BF16).
- **Fire safety assessment:** Strong general knowledge, should handle HK regulatory concepts well given 12B parameter count. Excellent at structured data extraction from images.
- **Caveat:** KV cache grows with context length. With a single image + prompt + JSON response, should stay within 12GB comfortably. Long conversations may push limits.

### 2. Qwen2.5-VL-7B Q4_K_M -- RECOMMENDED Best All-Rounder

- **Source:** Alibaba Qwen team
- **License:** Apache 2.0
- **Why it's strong:** Qwen2.5-VL was specifically designed for structured output (JSON/HTML) from visual inputs. The 7B version outperforms many larger competitors.
- **Vision capabilities:** Dynamic-resolution ViT, object localization with bounding boxes, structured data extraction from documents/forms, long-video QA.
- **Structured output:** Native JSON output support -- designed for invoice/form/table parsing. Generates stable JSON for coordinates and attributes.
- **How to run:** `ollama run qwen2.5vl:7b` or via llama.cpp with mmproj file.
- **VRAM:** Q4_K_M ~4.5GB weights, ~6-7GB total. Fits RTX 3060 easily. Tight on Tesla P4 but possible with short context.
- **Inference speed estimate:** ~15-25 tokens/sec on RTX 3060 at Q4_K_M.
- **Fire safety assessment:** The 3B variant "outperforms the 7B model of the previous Qwen2-VL." The 7B version is substantially better. Strong at identifying objects and spatial relationships in images.

### 3. Qwen3-VL-8B -- Best Newer Model (if llama.cpp support stable)

- **Source:** Alibaba Qwen team (October 2025)
- **License:** Apache 2.0
- **Why it's strong:** Latest generation, improved reasoning over Qwen2.5-VL. 256K native context.
- **How to run:** GGUF supported by llama.cpp since October 2025. Ollama support may still have quirks with mmproj files -- verify before committing.
- **VRAM:** Q4_K_M ~5GB weights, ~7-8GB total. Fits RTX 3060, does NOT fit Tesla P4.
- **Fire safety assessment:** Best reasoning capability among the small models tested. May be overkill for simple classification but excellent for nuanced assessments.

### 4. Qwen2.5-VL-3B Q4_K_M -- RECOMMENDED for Tesla P4

- **Source:** Alibaba Qwen team
- **License:** Apache 2.0
- **Why:** Specifically designed as an "edge AI solution." Outperforms previous-gen Qwen2-VL-7B despite being smaller.
- **VRAM:** ~2.5GB weights at Q4_K_M, ~4-5GB total. Comfortably fits Tesla P4 8GB.
- **How to run:** `ollama run qwen2.5vl:3b`
- **Inference speed estimate:** ~20-30 tokens/sec on Tesla P4 (small model compensates for slow GPU).
- **Fire safety assessment:** Adequate for structured fire safety classification. May miss subtle details that 7B+ models catch. Good enough as a fallback/secondary model.

### 5. Gemma 3 4B QAT -- Alternative for Tesla P4

- **Source:** Google DeepMind
- **How to run:** `ollama run gemma3:4b`
- **VRAM:** ~3.3GB weights, ~5-6GB total. Fits Tesla P4.
- **Fire safety assessment:** Competitive with Llama 2 27B on some benchmarks. Good vision capabilities but the 4B size limits complex reasoning.

### 6. Phi-4 Multimodal (5.6B) -- Microsoft's Compact Contender

- **Source:** Microsoft
- **License:** MIT
- **Why:** Unified text+image+audio architecture at only 5.6B params. Function calling support built in.
- **VRAM:** Q4_K_M ~4.5GB, total ~6-7GB. Fits RTX 3060 easily, tight on Tesla P4.
- **How to run:** Available on Ollama.
- **Fire safety assessment:** Strong reasoning for its size. May lack some visual grounding compared to Qwen2.5-VL which was specifically optimized for vision tasks.

### 7. InternVL3.5 -- Academic Powerhouse

- **Source:** Shanghai AI Lab / Tsinghua
- **Why:** InternVL3 "significantly outperforms both Qwen2.5-VL series and closed-source models" at the 78B level. The 4B and 8B variants are competitive.
- **VRAM:** 4B variant ~3.4GB on Ollama, 8B variant ~5.7GB.
- **How to run:** Available on Ollama via community models (`blaifa/InternVL3_5`). Also via LMDeploy.
- **Caveat:** Less mature Ollama support compared to Qwen/Gemma. Community-maintained model files.
- **Fire safety assessment:** Excellent visual perception and reasoning. Supports tool usage and GUI agents. May have slightly less polished structured output compared to Qwen2.5-VL.

---

## Inference Framework Comparison

| Framework | Vision Support | Quantization | Ease of Use | GPU Offload | Best For |
|-----------|---------------|-------------|-------------|-------------|----------|
| **Ollama** | Yes (Gemma 3, Qwen2.5-VL, LLaVA, MiniCPM-V) | GGUF (Q4, Q5, Q8) | Easiest | Auto | Quick setup, API server |
| **llama.cpp** | Yes (via mmproj) | GGUF (all levels) | Moderate | Manual `-ngl` flag | Maximum control, lowest overhead |
| **vLLM** | Yes | AWQ, GPTQ, FP8 | Moderate | Full | High-throughput serving |
| **transformers** | Yes | BitsAndBytes (4/8-bit) | Easy (Python) | Auto | Prototyping, fine-tuning |
| **LMDeploy** | Yes (InternVL) | W4A16, W8A8 | Moderate | Full | InternVL-specific optimization |

**Recommendation:** Use **Ollama** for simplicity. It provides an OpenAI-compatible API endpoint, making integration with the existing FireEye pipeline straightforward (swap the OpenRouter API URL for a local Ollama URL).

---

## Structured JSON Output Strategies

All recommended models support structured output, but reliability varies:

1. **Qwen2.5-VL** -- Best native JSON support. Designed for structured data extraction. Use system prompt with JSON schema.
2. **Gemma 3** -- Supports function calling. Use Ollama's `format: json` parameter for guaranteed valid JSON.
3. **Ollama JSON mode** -- As of 2025, Ollama supports `format: json` that forces valid JSON output from almost any model.
4. **llama.cpp grammar** -- Can constrain output to match a specific JSON schema using GBNF grammars. Most reliable but requires grammar definition.
5. **Outlines / SGLang** -- Python libraries that use compressed finite state machines for constrained decoding. Up to 2.5x throughput improvement over naive approaches.

---

## Recommended Deployment Strategy for FireEye

### Primary (RTX 3060 12GB):
```
Model: Gemma 3 12B QAT (int4) OR Qwen2.5-VL-7B Q4_K_M
Framework: Ollama
Command: ollama run gemma3:12b OR ollama run qwen2.5vl:7b
API: http://localhost:11434/api/chat (OpenAI-compatible)
Expected speed: ~10-25 tok/s
```

### Fallback (Tesla P4 8GB):
```
Model: Qwen2.5-VL-3B Q4_K_M
Framework: Ollama
Command: ollama run qwen2.5vl:3b
Expected speed: ~15-25 tok/s (smaller model compensates for slower GPU)
```

### Integration approach:
The existing FireEye pipeline uses OpenRouter API. To switch to local:
1. Install Ollama on both machines
2. Change `FIREEYE_OPENROUTER_API_KEY` → `FIREEYE_LOCAL_LLM_URL=http://localhost:11434`
3. Use the same prompt templates (YAML) -- models support the same system/user message format
4. Add `format: json` to Ollama API calls for reliable structured output
5. Implement fallback: try local first, fall back to OpenRouter if local is unavailable

---

## Quality vs Speed vs VRAM Summary

```
Quality (fire safety task):
  Gemma 3 12B QAT  ████████████████████░  (Excellent - best on 3060)
  Qwen2.5-VL-7B    ████████████████████░  (Excellent - best structured output)
  Qwen3-VL-8B      █████████████████████  (Best - if framework support stable)
  InternVL3.5 8B   ████████████████████░  (Excellent - strong vision)
  Phi-4 Multimodal  ███████████████████░  (Very Good)
  Gemma 3 4B QAT   ████████████████░░░░  (Good)
  Qwen2.5-VL-3B    ███████████████░░░░░  (Good - best for P4)
  Qwen3-VL-2B      ██████████████░░░░░░  (Decent)
  SmolVLM 2.2B     ████████████░░░░░░░░  (Moderate)

Speed on RTX 3060 (tokens/sec, estimated):
  SmolVLM 2.2B     ~40-60 tok/s  (fastest)
  Qwen2.5-VL-3B    ~30-40 tok/s
  Gemma 3 4B QAT   ~25-35 tok/s
  Qwen2.5-VL-7B    ~15-25 tok/s
  Gemma 3 12B QAT  ~10-20 tok/s  (slowest but best quality)

VRAM usage (model weights only, Q4):
  SmolVLM 2.2B     ~2.0 GB
  Qwen3-VL-2B      ~1.5 GB
  Qwen2.5-VL-3B    ~2.5 GB
  Gemma 3 4B QAT   ~3.3 GB
  Phi-4 Multimodal  ~4.5 GB
  Qwen2.5-VL-7B    ~4.5 GB
  Qwen3-VL-8B      ~5.0 GB
  InternVL3.5 8B   ~5.7 GB
  Gemma 3 12B QAT  ~6.6 GB
```

---

## Key Takeaways

1. **Largest on RTX 3060 12GB:** Gemma 3 12B QAT (6.6GB weights, ~8-10GB total). This is the quality ceiling for your primary GPU.

2. **Largest on Tesla P4 8GB:** Qwen2.5-VL-3B or Gemma 3 4B QAT. The 7B models are technically possible but will be very slow on Pascal.

3. **Smallest usable:** Qwen2.5-VL-3B (Q4_K_M, ~2.5GB). Below this size, models lack sufficient knowledge for nuanced fire safety regulatory analysis.

4. **Best for structured JSON:** Qwen2.5-VL family -- purpose-built for structured data extraction from images.

5. **Easiest deployment:** Ollama with `format: json` -- drop-in replacement for the existing OpenRouter API calls.

6. **Quality gap vs cloud models:** Local 7-12B models will be noticeably worse than Gemini Flash or GPT-4o for complex fire safety reasoning. They're adequate for classification and basic assessment but may struggle with nuanced HK regulatory interpretations. Consider a hybrid approach: local for Stage 2 (risk classification), cloud for Stage 3 (detailed assessment).

---

## Sources

- [Koyeb: Best Open Source Multimodal Vision Models 2025](https://www.koyeb.com/blog/best-multimodal-vision-models-in-2025)
- [BentoML: Multimodal AI - Open-Source Vision Language Models 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [SiliconFlow: Best Open Source Multimodal Models 2026](https://www.siliconflow.com/articles/en/best-open-source-multimodal-models-2025)
- [LocalLLM.in: Ollama VRAM Requirements 2026 Guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [Google Developers: Gemma 3 QAT Models](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- [Qwen2.5-VL HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen2.5-VL-3B HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen3-VL-8B GGUF on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Ollama Vision Models](https://ollama.com/search?c=vision)
- [Ollama Qwen2.5-VL](https://ollama.com/library/qwen2.5vl)
- [llama.cpp Multimodal Support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [InternVL3 Official Blog](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)
- [InternVL3.5 Official Blog](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- [Clarifai: Benchmarking VLMs - Gemma 3 vs MiniCPM vs Qwen 2.5 VL](https://www.clarifai.com/blog/benchmarking-best-open-source-vision-language-models)
- [SmolVLM on HuggingFace](https://huggingface.co/blog/smolvlm)
- [Microsoft Phi-4 Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [Unsloth: Qwen3-VL How to Run](https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune)
- [APXML: GPU Requirements for Gemma 3](https://apxml.com/posts/gemma-3-gpu-requirements)
- [Roboflow: Best Local Vision-Language Models](https://blog.roboflow.com/local-vision-language-models/)
- [DataCamp: Top 10 Vision Language Models 2026](https://www.datacamp.com/blog/top-vision-language-models)
