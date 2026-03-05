# FireEye Pipeline Performance Report

## 1. Current Pipeline Architecture

The FireEye pipeline processes images through 4 sequential stages:

```
Image Upload
    |
    v
[Stage 1] YOLO Detection          (local, CPU, yolo11n.pt)
    |
    v
[Stage 2] Risk Classifier         (VLM call via OpenRouter)
    |
    v
[Stage 3a] Present Agent          (VLM call via OpenRouter)
    |
    v
[Stage 3b] Future Agent           (VLM call via OpenRouter)
    |
    v
Result (AnalysisResult)
```

**Model**: `google/gemini-3-flash-preview` via OpenRouter ($0.50/1M input, $3.00/1M output, ~140 t/s)

**Key files**:
- `app/pipeline/orchestrator.py` — stage sequencing
- `app/pipeline/yolo_detector.py` — YOLO inference (local)
- `app/pipeline/risk_classifier.py` — VLM risk classification
- `app/pipeline/llm_agents.py` — Present + Future agents
- `app/services/openrouter_client.py` — OpenRouter API wrapper
- `app/services/image_utils.py` — base64 encoding
- `ui_app.py` — NiceGUI dashboard (runs pipeline via `asyncio.to_thread`)

### What each VLM call does

| Stage | Input | Output | System prompt size |
|-------|-------|--------|--------------------|
| Risk Classifier | Image + detection list | `{risk_level, confidence, reason}` | ~550 chars |
| Present Agent | Image + detections + risk level | `{summary, hazards[], distances[]}` | ~470 chars |
| Future Agent | Image + detections + risk + present assessment | `{scenarios[], overall_risk, recommendation}` | ~1,400 chars |

All three calls send the **full base64-encoded image** with every request.

---

## 2. Bottleneck Analysis

### 2.1 Three Sequential VLM Calls (PRIMARY BOTTLENECK)

Each VLM call involves:
1. Base64-encoding the image (~100-500 KB encoded)
2. HTTP round-trip to OpenRouter
3. OpenRouter routing to Google's API
4. Gemini processing (image tokenization + text generation)
5. JSON response parsing

**Estimated per-call latency**: 2-8 seconds depending on image size and load.
**Total VLM time**: 6-24 seconds for 3 serial calls.

YOLO inference (Stage 1) is fast (~100-500ms on CPU with yolo11n), so it's negligible.

### 2.2 Redundant Image Encoding

The image is base64-encoded **3 separate times** (`encode_image_to_data_uri` in `risk_classifier.py:92`, `llm_agents.py:91`, `llm_agents.py:143`). Each call re-reads the file from disk and re-encodes it. This wastes CPU time and I/O.

### 2.3 No Image Resizing Before VLM

`image_utils.py` has a `resize_if_needed()` function (line 49) but it's **never called** before encoding images for VLM. A 4000x3000 photo encodes to ~2MB base64, consuming excessive tokens and upload bandwidth.

### 2.4 Sequential Dependency Chain

```
Risk Classifier → Present Agent → Future Agent
```

- Present Agent uses the risk classification result, but only as context text — not a hard dependency.
- Future Agent depends on Present Agent output (summary, hazards, distances).
- Risk and Present could potentially run in parallel with minor prompt changes.

### 2.5 Verbose Future Agent Prompt

The Future Agent system prompt is ~1,400 characters with detailed calibration instructions. This is 3x longer than the other prompts, adding input tokens and processing time.

---

## 3. Recommended Optimizations

### 3.1 Merge Risk Classifier + Present Agent into a Single Call (HIGH IMPACT)

**Current**: 2 separate VLM calls, each sending the image.
**Proposed**: One call that returns both risk classification AND present assessment.

```python
# Combined schema: {risk_level, confidence, reason, summary, hazards[], distances[]}
```

**Trade-offs**:
- Saves one full VLM round-trip (~2-8s)
- Slightly larger output schema, but well within model capabilities
- Prompt can be combined: "Classify risk AND describe the current scene"
- Minor risk of reduced quality from multi-task prompting (empirically unlikely with Gemini Flash)

**Estimated savings**: 30-40% of total VLM time.

### 3.2 Cache Base64 Encoding (EASY WIN)

Encode the image once and pass the data URI to all consumers:

```python
data_uri = encode_image_to_data_uri(image_path)
# Pass data_uri to risk_classifier, present agent, future agent
```

**Estimated savings**: ~50-200ms (minor but free).

### 3.3 Resize Images Before VLM (EASY WIN)

Add image resizing before base64 encoding. Most VLMs don't benefit from >1280px input:

```python
def encode_image_to_data_uri(image_path, max_dim=1280):
    img = load_image(image_path)
    img = resize_if_needed(img, max_dim)
    _, buf = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"
```

**Trade-offs**:
- Reduces base64 payload from ~2MB to ~200KB for large images
- Fewer image tokens = faster VLM processing + lower cost
- Minimal quality impact for fire detection (doesn't need pixel-perfect detail)

**Estimated savings**: 20-50% reduction in VLM processing time for large images.

### 3.4 Parallelize Remaining Calls (MEDIUM IMPACT)

After merging Risk+Present (3.1), only 2 VLM calls remain:
1. Combined Risk+Present
2. Future Agent

These must stay sequential since Future depends on Present. But if we keep them separate AND don't merge:
- Risk Classifier and Present Agent can run in parallel (Present doesn't critically need the risk level; it can be provided as "preliminary heuristic risk" from the fast `classify_from_detections()` function).

### 3.5 Use a Faster Provider or Model (MEDIUM IMPACT)

| Provider/Model | Speed | Vision | Pricing (input/output per 1M) | Notes |
|----------------|-------|--------|-------------------------------|-------|
| **Gemini 3 Flash Preview** (current) | ~140 t/s | Yes | $0.50 / $3.00 | Good quality, moderate speed |
| **Gemini 2.0 Flash Lite** | ~200+ t/s | Yes | $0.075 / $0.30 | 10x cheaper, faster, less capable |
| **Groq Llama 3.2 11B Vision** | ~1000+ t/s | Yes | $0.16 / $0.16 | Extremely fast, smaller model |
| **Groq Llama 3.2 90B Vision** | ~300+ t/s | Yes | ~$1.40 / $1.40 | Fast, larger model |
| **GPT-4o mini** | ~97 t/s | Yes | $0.15 / $0.60 | Fast TTFT (0.56s), good quality |

**Recommendation**: For speed-critical deployments, try **Groq Llama 3.2 11B Vision** as the risk+present classifier (structured JSON output, simple task). Keep Gemini Flash for the Future Agent if quality matters.

**Alternative**: Use **Gemini 2.0 Flash Lite** via OpenRouter for all calls — 10x cheaper and faster, with acceptable quality for fire risk assessment.

**Trade-offs**:
- Groq: Blazing fast, but Llama 3.2 11B is less capable than Gemini Flash for nuanced reasoning
- Gemini 2.0 Flash Lite: Cheaper and faster, slightly less capable
- Switching providers means managing multiple API keys/clients

### 3.6 Simplify Prompts (LOW IMPACT)

The Future Agent system prompt (~1,400 chars) contains extensive calibration instructions. Consider:
- Moving calibration details to few-shot examples instead of system prompt
- Trimming redundant instructions (the likelihood/severity definitions could be shorter)
- Using structured output constraints (JSON schema) to enforce format instead of prose instructions

**Estimated savings**: 10-20% fewer input tokens, marginal speed improvement.

---

## 4. Future Agent Assessment

### What it does

The Future Agent (`llm_agents.predict_future()`) takes the scene analysis from the Present Agent and predicts:
- **Branching scenarios**: What could go wrong (fire spread, explosion, etc.)
- **Per-scenario metadata**: likelihood, severity, time_horizon
- **Overall projected risk**: Recalibrated risk level based on future analysis
- **Recommendation**: Actionable advice

### Value it provides

1. **Scenario planning**: Gives operators specific things to watch for
2. **Risk recalibration**: The `overall_risk` field may differ from the initial risk classifier
3. **Actionable recommendation**: "Move flammable materials" vs just "high risk"
4. **Time horizons**: Helps prioritize response urgency

### Can it be removed?

**Partial removal is viable; full removal causes regression.** Here's why:

**What you lose**:
- The `recommendation` field — this is the most user-visible output in the dashboard (displayed in the "Recommendation" card in the overview tab)
- The `overall_risk` recalibration — provides a second opinion on risk
- Scenario-level detail — likelihood/severity/time_horizon per scenario

**What you keep** (from Present + Risk alone):
- Risk level and reason (from Risk Classifier)
- Scene summary, hazards list, spatial distances (from Present Agent)

**Recommendation: Merge, don't remove.**

The Future Agent's most valuable outputs (recommendation, overall_risk) can be merged into the Present Agent prompt with minimal quality loss. Instead of 2 separate VLM calls for Stage 3, use one combined call:

```python
# Combined Present+Future schema:
{
    "summary": "...",
    "hazards": ["..."],
    "distances": ["..."],
    "scenarios": [{"scenario": "...", "likelihood": "...", "severity": "...", "time_horizon": "..."}],
    "overall_risk": "safe|low|medium|high|critical",
    "recommendation": "..."
}
```

This preserves all output fields while eliminating one VLM call.

---

## 5. Concrete Next Steps

### Phase 1: Quick Wins (no quality regression)

1. **Cache base64 encoding** — encode image once, pass to all stages
2. **Resize images before encoding** — cap at 1280px max dimension
3. **Use heuristic risk as preliminary** — `classify_from_detections()` is already implemented but unused in the main pipeline; it can feed the Present Agent while the LLM risk runs in parallel

### Phase 2: Architecture Changes (test for regression)

4. **Merge Risk + Present into single VLM call** — combined prompt and schema
5. **Merge Present + Future into single VLM call** — OR merge all three into one call
6. **Skip Future Agent for "safe" risk** — orchestrator already has the logic structure for this (comment says "always run for now")

### Phase 3: Provider/Model Optimization (requires experimentation)

7. **Benchmark Gemini 2.0 Flash Lite** — swap `llm_model` in config, compare quality
8. **Benchmark Groq Llama 3.2 11B Vision** — add Groq client support, compare speed+quality
9. **A/B test merged vs separate prompts** — ensure quality doesn't regress on real fire images

### Optimal End State

```
Image Upload
    |
    v
[Stage 1] YOLO Detection (local, ~200ms)
    |
    v
[Stage 2] Combined Analysis (single VLM call, ~3-5s)
           - Risk classification
           - Present assessment
           - Future scenarios + recommendation
    |
    v
Result
```

**Expected improvement**: From 3 VLM calls (~6-24s) down to 1 (~2-8s). **2-3x faster end-to-end.**

With Groq or Flash Lite, the single VLM call could drop to ~1-3s, achieving **4-8x faster** than current.

---

## 6. Summary Table

| Optimization | Effort | Speed Impact | Quality Risk | Recommended? |
|---|---|---|---|---|
| Cache base64 | Trivial | Minor (~200ms) | None | Yes |
| Resize images | Easy | Moderate (20-50%) | None | Yes |
| Merge Risk+Present | Medium | High (30-40%) | Low | Yes |
| Merge all 3 into 1 call | Medium | Very high (60-70%) | Medium | Yes, with testing |
| Skip Future for safe risk | Easy | Conditional (saves 1 call) | None | Yes |
| Switch to Flash Lite | Easy | Moderate (faster model) | Low | Test first |
| Switch to Groq | Medium | Very high (10x speed) | Medium | Test first |
| Simplify prompts | Easy | Low (10-20% tokens) | Low | Nice to have |
