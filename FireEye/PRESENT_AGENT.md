# FireEye Stage 3a — Present Agent

## Overview

The Present Agent is the first of two LLM agents in Stage 3. It acts as an objective **scene observer** — describing what is currently visible in the image, the spatial relationships between objects, and any environmental factors. Its output provides grounded context for the Future Agent.

Source: `app/pipeline/llm_agents.py:assess_present()`

```
Image + Detections + Risk  ──►  Present Agent  ──►  PresentAssessment
                                                      ├─ summary:   str        (overall scene description)
                                                      ├─ hazards:   list[str]  (ignition sources, flammables)
                                                      └─ distances: list[str]  (spatial relationships)
```


## What It Does

The Present Agent answers three questions:

1. **What ignition sources are present?** For each, it states whether it appears controlled (torch, candle, welding arc) or uncontrolled (freely burning fire).

2. **What flammable materials are present?** For each, it notes the distance from the nearest ignition source.

3. **What environmental factors exist?** Wind indicators, embers in flight, enclosure type, and structural elements that affect fire behaviour.


## Design Principles

- **Objective, not editorial** — The agent describes what it sees without injecting alarm language or speculation. "A candle flame 0.5m from a stack of wood planks" rather than "a dangerous fire threatening nearby fuel."

- **Concise and precise** — Factual statements about positions, distances, and states. No filler.

- **Controlled vs. uncontrolled distinction** — This is a critical classification the Present Agent makes for each flame, and it directly influences the Future Agent's risk assessment.


## Inputs

The Present Agent receives:

| Input | Source | Purpose |
|-------|--------|---------|
| Image | Original uploaded image (base64 data URI) | Visual analysis of the scene |
| Detection summary | Stage 1 YOLO output | Object labels, confidences, and bounding box coordinates |
| Preliminary risk level | Stage 2 classification | Provides calibration context (e.g., "current risk: low") |


## Output Schema

```json
{
  "summary": "A small candle flame burns on a flat surface in the centre of the frame. No other objects or materials are visible within the scene.",
  "hazards": [
    "Small controlled candle flame (centre of frame)"
  ],
  "distances": [
    "No flammable materials visible — nearest object is beyond the frame"
  ]
}
```

### Field Details

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | One to three sentences describing the overall scene state. |
| `hazards` | list of strings | Each entry describes one ignition source or flammable material, including whether flames are controlled or uncontrolled. |
| `distances` | list of strings | Each entry describes a spatial relationship between an ignition source and a flammable material, or notes the absence of nearby fuel. |


## How the Future Agent Uses This

The Future Agent receives the Present Agent's output as structured text:

```
Present assessment:
  Summary: [summary]
  Ignition sources / flammables: [hazards joined]
  Distances: [distances joined]
```

This grounds the Future Agent's predictions in observed spatial facts rather than requiring it to re-interpret the raw image independently. For example, if the Present Agent reports "wood stack 80px (~0.5m) from candle flame", the Future Agent can reason about radiant heat ignition at that specific distance.


## Configuration

The Present Agent uses the same LLM configuration as all other LLM calls:

| Setting | Default | Env Variable |
|---------|---------|-------------|
| Model | `google/gemini-2.5-flash` | `FIREEYE_LLM_MODEL` |
| Temperature | `0.0` | `FIREEYE_LLM_TEMPERATURE` |

The JSON output schema is enforced via OpenRouter's structured output support, ensuring the response always conforms to the expected format.
