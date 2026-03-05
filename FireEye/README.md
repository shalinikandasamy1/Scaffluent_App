# FireEye — Predictive Fire Safety System

FireEye is an AI-powered fire safety analysis system for construction and industrial sites. It combines real-time object detection with LLM-based reasoning to assess fire spread risk and predict how hazardous situations may evolve.

## Architecture

FireEye uses a three-stage pipeline. Each stage builds on the previous one:

```
                          ┌──────────────────────┐
                          │     Input Image       │
                          │  (CCTV or mobile)     │
                          └──────────┬─────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │  Stage 1: YOLO Object Detection        │
                │  Identifies fire, flame, wood, gas,    │
                │  smoke, and other relevant objects      │
                │                                        │
                │  → list[Detection] + annotated image   │
                └────────────────────┬───────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │  Stage 2: Risk Classification          │
                │  LLM classifies overall fire spread    │
                │  risk: safe / low / medium / high /    │
                │  critical                              │
                │                                        │
                │  → RiskClassification                  │
                └────────────────────┬───────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
         ┌──────────────────────┐  ┌──────────────────────┐
         │  Stage 3a: Present   │  │  Stage 3b: Future    │
         │  Agent               │──│  Agent               │
         │                      │  │                      │
         │  Describes current   │  │  Predicts branching  │
         │  hazards & spatial   │  │  scenarios with      │
         │  relationships       │  │  likelihood, severity│
         │                      │  │  and time horizons   │
         │  → PresentAssessment │  │  → FuturePrediction  │
         └──────────────────────┘  └──────────────────────┘
```

Each stage has detailed documentation:

| Stage | Document | Source |
|-------|----------|--------|
| Stage 1 — YOLO Detection | [YOLO_DETECTOR.md](YOLO_DETECTOR.md) | `app/pipeline/yolo_detector.py` |
| Stage 2 — Risk Classification | [RISK_CLASSIFIER.md](RISK_CLASSIFIER.md) | `app/pipeline/risk_classifier.py` |
| Stage 3a — Present Agent | [PRESENT_AGENT.md](PRESENT_AGENT.md) | `app/pipeline/llm_agents.py` |
| Stage 3b — Future Agent | [FUTURE_AGENT.md](FUTURE_AGENT.md) | `app/pipeline/llm_agents.py` |


## Quick Start

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

### Setup

```bash
cd FireEye
pip install -r requirements.txt

# Create .env with your API key
echo "FIREEYE_OPENROUTER_API_KEY=your-key-here" > .env
```

### Run the API Server

```bash
python -m app.main
```

The server starts at `http://localhost:8000`. Interactive docs are available at `/docs` (Swagger UI).

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/ingest/` | Upload an image (multipart form data) — returns `image_id` |
| `POST` | `/analysis/{image_id}` | Run the full pipeline on an ingested image |
| `GET` | `/analysis/{image_id}/annotated` | Download the YOLO-annotated output image |

### Example Workflow

```bash
# 1. Upload an image
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@photo.jpg" \
  -F "source_type=mobile" \
  -F "source_id=inspector-01" \
  -F "location=Building A, Floor 3"

# Response: {"image_id": "abc123...", "message": "Image ingested successfully"}

# 2. Run analysis
curl -X POST http://localhost:8000/analysis/abc123...

# 3. View annotated image
curl http://localhost:8000/analysis/abc123.../annotated -o annotated.jpg
```


## Configuration

All settings are loaded from environment variables (prefixed with `FIREEYE_`) or a `.env` file. Defined in `app/config.py`.

| Setting | Default | Env Variable | Description |
|---------|---------|-------------|-------------|
| OpenRouter API key | *(required)* | `FIREEYE_OPENROUTER_API_KEY` | API key for LLM calls |
| LLM model | `google/gemini-2.5-flash` | `FIREEYE_LLM_MODEL` | Model used for risk classification and agents |
| LLM temperature | `0.0` | `FIREEYE_LLM_TEMPERATURE` | 0.0 for deterministic output |
| YOLO model | `yolo11n.pt` | `FIREEYE_YOLO_MODEL_NAME` | YOLO weights file (auto-downloaded) |
| YOLO confidence | `0.25` | `FIREEYE_YOLO_CONFIDENCE_THRESHOLD` | Minimum detection confidence |
| Host | `0.0.0.0` | `FIREEYE_HOST` | Server bind address |
| Port | `8000` | `FIREEYE_PORT` | Server bind port |
| Debug | `false` | `FIREEYE_DEBUG` | Enable debug logging and hot-reload |


## Testing

### End-to-End Test

Runs the full pipeline on images in `test_data/dangerous/` and `test_data/safe/`, verifying that dangerous scenes are classified as medium/high/critical and safe scenes as safe/low.

```bash
python test_e2e.py
```

### Future Agent Stress Test

Generates four parametric synthetic scenes (A through D) with escalating risk, then verifies the Future Agent correctly identifies the risk level for each:

| Scene | Description | Expected Risk |
|-------|-------------|---------------|
| A | Small candle, empty room | LOW |
| B | Small candle + wood 80px away | MEDIUM |
| C | Large fire + embers, open space | HIGH |
| D | Large fire + embers + materials + wind | CRITICAL |

```bash
python test_future_prediction.py
```

See [FUTURE_AGENT.md](FUTURE_AGENT.md) for full details on the condition-based risk model and what drives escalation between levels.


## Project Structure

```
FireEye/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings from env / .env
│   ├── models/
│   │   └── schemas.py           # Pydantic models (Detection, RiskLevel, etc.)
│   ├── pipeline/
│   │   ├── orchestrator.py      # Coordinates Stage 1 → 2 → 3
│   │   ├── yolo_detector.py     # Stage 1: YOLO object detection
│   │   ├── risk_classifier.py   # Stage 2: Risk classification
│   │   └── llm_agents.py        # Stage 3: Present + Future agents
│   ├── routers/
│   │   ├── health.py            # GET /health
│   │   ├── ingest.py            # POST /ingest/
│   │   └── analysis.py          # POST /analysis/{id}, GET annotated image
│   ├── services/
│   │   ├── openrouter_client.py # OpenRouter/OpenAI API wrapper
│   │   └── image_utils.py       # Base64 encoding, image I/O, resizing
│   └── storage/
│       └── image_store.py       # Filesystem-based image persistence
├── test_data/                   # Test images (dangerous/, safe/, synthetic/)
├── images/                      # Runtime image storage (input/, output/)
├── generate_test_scenes.py      # Parametric synthetic scene generator
├── test_future_prediction.py    # Future Agent stress test
├── test_e2e.py                  # End-to-end pipeline test
├── ui_app.py                    # NiceGUI dashboard entry point (port 8090)
├── test_ui.py                   # Playwright visual QA for the dashboard
├── qa_screenshots/              # QA baseline screenshots
├── requirements.txt             # Python dependencies
├── YOLO_DETECTOR.md             # Stage 1 documentation
├── RISK_CLASSIFIER.md           # Stage 2 documentation
├── PRESENT_AGENT.md             # Stage 3a documentation
├── FUTURE_AGENT.md              # Stage 3b documentation
├── DASHBOARD.md                 # NiceGUI dashboard documentation
├── VIDEO_GENERATION.md          # Test data generation notes
└── README.md                    # This file
```


## NiceGUI Dashboard

In addition to the FastAPI server, FireEye includes a standalone NiceGUI web dashboard for interactive demo and diagnostics. It runs the full pipeline end-to-end with live stage progress, result tabs, and visual comparisons.

```bash
source venv/bin/activate
pip install "nicegui==3.8.0"
python ui_app.py
# Open http://localhost:8090
```

See [DASHBOARD.md](DASHBOARD.md) for full details on the UI layout, result tabs, and visual QA testing.


## Dependencies

- **FastAPI** + **Uvicorn** — API server
- **Ultralytics** — YOLO object detection
- **OpenAI SDK** — OpenRouter API client (OpenAI-compatible)
- **OpenCV** + **Pillow** + **NumPy** — Image processing
- **Pydantic** — Data validation and settings management
- **NiceGUI** — Web dashboard (optional, for `ui_app.py`)

---

*FireEye is part of the [Scaffluent](../README.md) scaffolding safety evaluation platform.*
