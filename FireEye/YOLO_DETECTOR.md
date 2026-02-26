# FireEye Stage 1 — YOLO Object Detection

## Overview

Stage 1 is the entry point of the FireEye analysis pipeline. It runs a YOLO (You Only Look Once) object detection model on the input image to identify and locate objects relevant to fire safety — ignition sources, flammable materials, and other hazardous items.

Source: `app/pipeline/yolo_detector.py`

```
Input Image  ──►  YOLO Model  ──►  list[Detection]
                                      ├─ label: str        (e.g. "fire", "wood", "person")
                                      ├─ confidence: float  (0.0 – 1.0)
                                      └─ bbox: BoundingBox  (x1, y1, x2, y2 pixel coords)
```


## How It Works

1. The YOLO model (`yolo11n.pt` by default) is **lazy-loaded** on first use — weights are auto-downloaded if not present.
2. The model is forced to run on **CPU** to avoid GPU compatibility issues (specifically `sm_61` architecture mismatches).
3. Inference is run on the input image with a configurable confidence threshold (default: `0.25`).
4. Each detected object is returned as a `Detection` with its class label, confidence score, and bounding box coordinates.
5. An **annotated copy** of the image is saved with YOLO's built-in bounding box overlays for visual inspection.


## Key Functions

| Function | Purpose |
|----------|---------|
| `detect(image_path)` | Run inference, return detections only. |
| `detect_and_annotate(image_path, output_path)` | Run inference, save annotated image, return detections. |

The orchestrator uses `detect_and_annotate()` so both the structured detections and a visual reference are produced.


## Object Categories

Detected labels are not filtered at this stage — YOLO returns all objects it recognises from its training set. Downstream stages (Stage 2) group these into fire-relevant categories:

| Category | Labels | Relevance |
|----------|--------|-----------|
| **Ignition sources** | fire, flame, lighter, match, candle, torch, welding, spark, heater, stove | Active or potential sources of ignition |
| **Flammable materials** | wood, cardboard, paper, fabric, cloth, plastic, foam, hay, straw | Fuel that could catch fire |
| **Hazardous objects** | gas cylinder, gas tank, fuel, gasoline, kerosene, propane, smoke | Objects that elevate risk by their nature |

Non-fire-related detections (e.g. "person", "chair") are still passed downstream — the LLM agents may use them for spatial context.


## Configuration

| Setting | Default | Env Variable | Description |
|---------|---------|-------------|-------------|
| Model | `yolo11n.pt` | `FIREEYE_YOLO_MODEL_NAME` | YOLO model weights file |
| Confidence threshold | `0.25` | `FIREEYE_YOLO_CONFIDENCE_THRESHOLD` | Minimum detection confidence |

The `n` in `yolo11n.pt` indicates the "nano" variant — fastest inference, smallest model. Larger variants (`yolo11s.pt`, `yolo11m.pt`, etc.) offer higher accuracy at the cost of speed.


## Output Example

For an image containing a candle near a stack of wood:

```json
[
  {
    "label": "candle",
    "confidence": 0.87,
    "bbox": {"x1": 310, "y1": 280, "x2": 340, "y2": 340}
  },
  {
    "label": "wood",
    "confidence": 0.72,
    "bbox": {"x1": 400, "y1": 290, "x2": 480, "y2": 350}
  }
]
```

These detections, along with the original image, are passed to Stage 2 (Risk Classification).
