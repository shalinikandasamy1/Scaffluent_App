# Cross-Model Prompt Portability: Research Summary

**Date:** 2026-03-08
**Context:** FireEye pipeline supports both cloud (Gemini Flash via OpenRouter) and local (Qwen2.5-VL-7B via Ollama) LLM backends. This document summarizes research on maintaining consistent behavior across models.

---

## The Problem

When switching LLMs, three categories of breakage occur:

1. **Output format variance** — models differ in whether they wrap JSON in markdown fences, add preamble text, or restructure field names
2. **Instruction following fidelity** — smaller models need more explicit/repetitive instructions; larger models perform worse with over-specification
3. **Semantic drift** — identical prompts produce different reasoning chains and risk assessments across model families

## 5 Techniques, Ranked by Impact

### 1. Schema-Enforced Structured Output (Highest Impact)
- **Already implemented** via `chat_completion_json()` with `response_format` parameter
- Cloud (OpenRouter): strict JSON schema enforcement at decode level
- Local (Ollama): `format: json` — looser enforcement, schema acts as "best effort"
- **Gap**: For local models, inject the JSON schema description into the system prompt as a fallback

### 2. Master Prompt + Model Adapter Pattern (High Impact, Low Cost)
- **Already implemented** via `model_adapters.yaml` and `get_system_prompt(name, model_name)`
- Master prompts (`risk_classifier.yaml`, etc.) contain domain knowledge (HK regulations, risk scales)
- Adapters contain model-specific behavioral nudges (JSON formatting, reasoning hints)
- **Extend** adapters with: `reasoning_hint`, `temperature_override`, `image_position`, `few_shot_examples`

### 3. Pydantic Validation with Retry Loop (High Impact)
- Even with schema enforcement, models produce valid JSON with semantic errors (e.g., confidence=1.5)
- Add a Pydantic validation layer with automatic retry and error injection for self-correction
- Libraries: [Instructor](https://python.useinstructor.com/) or lightweight custom implementation

### 4. DSPy-Style Prompt Compilation (Medium Impact, Higher Complexity)
- Treats prompts as programs, not strings; compiles optimal prompt per model
- Overkill for FireEye's 2-3 model targets, but useful if scaling to 5+ models
- Consider when adapter maintenance becomes burdensome

### 5. Dynamic Few-Shot Calibration Per Model (Medium Impact)
- Add 2-3 model-specific few-shot examples to correct systematic bias
- Start with 0 examples; add only when evaluation shows consistent misjudgments
- Stored in `model_adapters.yaml` per model family

## Implementation Status in FireEye

| Technique | Status | Priority |
|-----------|--------|----------|
| Schema enforcement | 80% done (cloud full, local partial) | Close gap for local |
| Master + adapter pattern | Implemented (json_instruction only) | Extend adapter fields |
| Pydantic validation retry | Not implemented | Next priority |
| DSPy compilation | Not implemented | Future if needed |
| Few-shot calibration | Not implemented | Add as evaluation reveals bias |

## Evaluation Results

| Backend | Model | Accuracy | Avg Time | Cost |
|---------|-------|----------|----------|------|
| Cloud (OpenRouter) | Gemini 3 Flash Preview | **100%** (22/22) | 5.1s/img | ~$0.001/img |
| Local (Ollama) | Qwen2.5-VL-7B Q4 | **90.9%** (20/22) | 10.7s/img | $0.00/img |
| Heuristic | N/A (deterministic) | **100%** (22/22) | 0.09s/img | $0.00/img |

Local model failures: 2 dangerous images classified as "low" — both had subtle visual cues (smoke without visible flame, welding with ambiguous context). These are candidates for few-shot calibration examples.

## Recommended Next Steps

1. Inject schema description into system prompt for local models (closes the Ollama gap)
2. Extend `model_adapters.yaml` with `reasoning_hint` for smaller models
3. Add Pydantic validation + retry in `openrouter_client.py`
4. After more evaluation data, add few-shot examples for the 2 failure cases

## Sources

- [DSPy](https://dspy.ai/) — Prompt compilation framework
- [Instructor](https://python.useinstructor.com/) — Structured output with validation
- [LangChain Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/) — Model-aware templating
- [Ollama JSON Mode](https://ollama.com/blog/structured-outputs) — Structured output support
- [llama.cpp GBNF Grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) — Token-level schema enforcement
