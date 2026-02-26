# FireEye Stage 2 — Risk Classification

## Overview

Stage 2 takes the YOLO detections from Stage 1 and classifies the overall fire spread risk of the scene. It supports two modes: a fast **heuristic classifier** based on object co-occurrence, and a more nuanced **LLM-based classifier** that also considers the image itself.

Currently, the pipeline uses the LLM classifier by default.

Source: `app/pipeline/risk_classifier.py`

```
Detections + Image  ──►  Risk Classifier  ──►  RiskClassification
                                                  ├─ risk_level:  RiskLevel (enum)
                                                  ├─ confidence:  float (0.0 – 1.0)
                                                  └─ reason:      str
```


## Risk Levels

| Level | Description |
|-------|-------------|
| **safe** | No fire, no active ignition source present. |
| **low** | Controlled or contained flame (candle, torch, welding arc) in a clear area with no flammable materials within reach. Normal work activity. |
| **medium** | Controlled flame with flammable materials at a safe but noteworthy distance, OR small uncontrolled flame with no immediate spread path. |
| **high** | Large or actively spreading flame, OR a flame in close proximity to significant flammable materials that could ignite via radiant heat or embers. |
| **critical** | Active fire spread already occurring, OR immediate multi-vector cascade risk (explosive containers adjacent to flame, ember storm reaching fuel). |


## Heuristic Classifier

The heuristic classifier (`classify_from_detections()`) uses a simple decision tree based on which object categories are present:

```
Has ignition source AND flammable material?  ──►  HIGH   (conf 0.8)
Has hazardous object (but not both above)?   ──►  MEDIUM (conf 0.7)
Has flammable material only?                 ──►  LOW    (conf 0.6)
None of the above?                           ──►  SAFE   (conf 0.9)
```

### Object Category Definitions

| Category | Labels |
|----------|--------|
| **IGNITION_LABELS** | fire, flame, lighter, match, candle, torch, welding, spark, heater, stove |
| **FLAMMABLE_LABELS** | wood, cardboard, paper, fabric, cloth, plastic, foam, hay, straw |
| **HAZARDOUS_LABELS** | fire, flame, smoke, gas cylinder, gas tank, lighter, match, candle, torch, welding, fuel, gasoline, kerosene, propane |

The heuristic classifier is fast (no API call) and useful as a fallback, but cannot reason about spatial relationships, flame size, or scene context.


## LLM Classifier

The LLM classifier (`classify_with_llm()`) sends the image and detection summary to an LLM via OpenRouter. The LLM receives:

1. A system prompt defining the risk scale and calibration guidance
2. The list of YOLO detections with confidence scores
3. The original image (as a base64 data URI)

### Calibration Principles

The LLM prompt is designed to avoid over-classification:

- **Open flames are normal** in construction/industrial contexts — welding, cutting torches, candles, and controlled burns are routine.
- Classification is based on **fire spread risk**, not the mere presence of flame.
- The LLM considers the **combination** of: flame size/control, proximity of flammable materials, ember/spark travel, and environmental factors.


## Configuration

| Setting | Default | Env Variable | Description |
|---------|---------|-------------|-------------|
| LLM model | `google/gemini-2.5-flash` | `FIREEYE_LLM_MODEL` | Model used for LLM classification |
| Temperature | `0.0` | `FIREEYE_LLM_TEMPERATURE` | Deterministic output |
| Risk confidence threshold | `0.5` | `FIREEYE_RISK_CONFIDENCE_THRESHOLD` | Minimum confidence for heuristic results |


## Pipeline Integration

The orchestrator calls `classify_with_llm()` for all images. The resulting `RiskClassification` is:

1. Included in the final `AnalysisResult` returned to the API consumer.
2. Passed as context to both the Present Agent and Future Agent in Stage 3, so they can calibrate their assessments relative to the classified risk.
