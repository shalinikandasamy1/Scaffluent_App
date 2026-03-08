# Session Summary: 2026-03-08 (17:50 - 19:00 HKT)

## PR #6 Review (Task 1)
- Posted independent code review on GitHub as Claude Code
- Found 2 high-priority issues (ZeroDivisionError, path traversal), 3 medium, 5 low
- Path traversal fix was implemented in this session

## Dashboard v2 (Task 2) — PR #7

### New Features
| Feature | Details |
|---------|---------|
| Compliance tab | Score gauge (CSS conic-gradient), regulatory flags table, issues list |
| Audit Log tab | JSONL history viewer with timing, model, and risk data |
| Backend selector | Switch between OpenRouter (cloud) and Ollama (local) from UI |
| Spatial proximity | Object distance analysis and safety concerns in Detections tab |
| Pipeline audit | Every analysis writes JSONL audit record with full metadata |

### Cross-Model Prompt Portability
| Component | Purpose |
|-----------|---------|
| `model_adapters.yaml` | Model-specific JSON instructions, reasoning hints, temperature |
| `_schema_to_description()` | Injects human-readable schema into system prompts for local models |
| `_validate_against_schema()` | Validates LLM responses; retries with error feedback |
| `reasoning_hint` | Step-by-step guidance for smaller models (Qwen, Gemma3) |

**Result: Local LLM accuracy improved from 90.9% to 100%**

### LLM Evaluation Results

| Backend | Model | Accuracy | Avg Time | Cost |
|---------|-------|----------|----------|------|
| Cloud | Gemini 3 Flash | 100% (22/22) | 5.1s/img | ~$0.001/img |
| Local | Qwen2.5-VL-7B | 100% (22/22) | 11.5s/img | ~$0.00008/img |
| Local | Qwen2.5-VL-3B | 86-91% (19-20/22) | 6.1s/img | ~$0.00004/img |
| Heuristic | N/A | 100% (22/22) | 0.09s/img | $0.00/img |

### Key Findings
- **7B is the minimum viable local model** for fire safety classification
- 3B model lacks risk calibration (never uses "critical" or "safe") and has JSON reliability issues
- Schema injection + validation retry is the key technique that bridges the cloud-local accuracy gap
- Energy break-even: ~250 images/day (RTX 3060), ~90/day (Tesla P4)
- Cold start overhead: ~19s for Ollama model loading

### Testing
- 56 unit tests passing
- Playwright visual QA: all 7 tabs verified with both cloud and local backends
- Full pipeline evaluation on 22 images across all backends

## Commits (7 on feature/dashboard-v2)
1. `c59d325` Dashboard v2: compliance, audit, dual LLM, model adapters
2. `8eb9bd4` Schema injection, validation retry, reasoning hints
3. `3f3bb34` Audit logging in UI, spatial proximity in detections
4. `658b66e` Energy cost analysis
5. `53890b5` evaluate.py backend recording fix
6. `b1dd060` 3B vs 7B benchmark comparison
7. `aa2bd4b` Updated QA screenshots

## Research Documents Created
- `research/local_vlm_research.md` — VLM recommendations for RTX 3060 and Tesla P4
- `research/cross_model_prompt_portability.md` — 5 techniques ranked by impact
- `research/energy_cost_analysis.md` — Local vs cloud economics
- `research/benchmark_results.json` — Raw benchmark data
