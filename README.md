# Scaffluent

Scaffluent is a scaffolding safety evaluation platform. It combines an iOS app for on-site inspections with a Python-based fire safety analysis backend called **FireEye**.

## Repository Structure

```
Scaffluent_App/
├── Scaffluent/           # iOS app (Swift / SwiftUI / Xcode)
│   └── Scaffluent/
│       ├── App/          # App entry point, global view model
│       ├── Models/       # Data models (EvaluationSession, Issue, User, etc.)
│       ├── Screens/      # UI screens (Home, Login, Evaluation, History, Results, Help)
│       ├── Components/   # Reusable UI components (SessionCard, IssueRow, StatusChip)
│       ├── Modules/      # Analysis modules (Aggregation)
│       ├── Services/     # Camera, Auth, Thermal SDK, Session storage
│       └── Utils/        # Constants, Logging
│
└── FireEye/              # AI fire safety analysis (Python)
    ├── app/              # FastAPI backend
    │   ├── pipeline/     # 3-stage analysis pipeline (YOLO → Risk → LLM Agents)
    │   ├── routers/      # API endpoints (health, ingest, analysis)
    │   ├── models/       # Pydantic schemas
    │   ├── services/     # OpenRouter client, image utilities
    │   └── storage/      # Filesystem image store
    ├── ui_app.py         # NiceGUI dashboard (standalone demo/diagnostic tool)
    ├── test_data/        # Test images, escalation frames, fire simulator
    └── docs              # Per-stage technical documentation
```

## Scaffluent iOS App

The iOS app is built with SwiftUI and targets on-site scaffolding inspectors. It provides:

- **Login** — user authentication
- **New Evaluation** — start an inspection session, capture images via camera/LiDAR and FLIR ONE Pro thermal camera
- **Evaluation** — run analysis modules against captured data
- **Results** — view detected issues with severity levels (info / warning / critical) and pass/fail outcome
- **History** — browse and review past evaluation sessions
- **Aggregation** — combines issues from all analysis modules into a final evaluation result

The app uses an MVVM architecture with `@Observable` view models. Key services include `CameraService` (AVCapture/ARKit/LiDAR), `ThermalSDKWrapper` (FLIR ONE Pro), `AuthService`, and `SessionStore`.

**Status:** The app structure and interfaces are defined. Model files contain interface descriptions; full implementations are in progress.

### Requirements

- Xcode 16+
- iOS 17+
- FLIR ONE Pro SDK (for thermal imaging)

## FireEye

FireEye is an AI-powered fire safety analysis system for construction and industrial sites. It combines YOLO object detection with LLM-based reasoning to assess fire spread risk and predict how hazardous situations may evolve.

See [FireEye/README.md](FireEye/README.md) for full documentation.

### Pipeline Overview

```
Input Image → Stage 1: YOLO Detection → Stage 2: Risk Classification → Stage 3a: Present Agent
                                                                      → Stage 3b: Future Agent
```

1. **Stage 1 — YOLO Detection**: Identifies fire, flame, smoke, gas cylinders, wood, and other objects using YOLOv11
2. **Stage 2 — Risk Classification**: LLM classifies overall fire spread risk (safe / low / medium / high / critical)
3. **Stage 3a — Present Agent**: Describes current hazards and spatial relationships
4. **Stage 3b — Future Agent**: Predicts branching future scenarios with likelihood, severity, and time horizons

### Two Interfaces

| Interface | Port | Entry Point | Purpose |
|-----------|------|-------------|---------|
| **FastAPI** | 8000 | `python -m app.main` | REST API for programmatic access |
| **NiceGUI Dashboard** | 8090 | `python ui_app.py` | Interactive web UI for demo and diagnostics |

### Quick Start

```bash
cd FireEye
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install "nicegui==3.8.0"  # for the dashboard

echo "FIREEYE_OPENROUTER_API_KEY=your-key-here" > .env

# Run the API server
python -m app.main

# Or run the NiceGUI dashboard
python ui_app.py
```

## How They Fit Together

The Scaffluent iOS app captures images on-site (including thermal imagery from FLIR ONE Pro). These images can be sent to the FireEye backend for fire safety analysis. FireEye processes the image through its multi-stage pipeline and returns structured risk assessments that the app presents to the inspector.

The NiceGUI dashboard provides a standalone web interface for testing and demonstrating the FireEye pipeline without needing the iOS app.

## Configuration

FireEye settings are loaded from environment variables (prefixed with `FIREEYE_`) or a `.env` file in the `FireEye/` directory. The only required setting is `FIREEYE_OPENROUTER_API_KEY`.

See [FireEye/README.md](FireEye/README.md) for the full configuration reference.

## Documentation Index

| Document | Description |
|----------|-------------|
| [FireEye/README.md](FireEye/README.md) | FireEye overview, API reference, setup, and project structure |
| [FireEye/DASHBOARD.md](FireEye/DASHBOARD.md) | NiceGUI dashboard UI, tabs, pipeline execution, and visual QA |
| [FireEye/YOLO_DETECTOR.md](FireEye/YOLO_DETECTOR.md) | Stage 1 — YOLO object detection details |
| [FireEye/RISK_CLASSIFIER.md](FireEye/RISK_CLASSIFIER.md) | Stage 2 — Risk classification (heuristic + LLM) |
| [FireEye/PRESENT_AGENT.md](FireEye/PRESENT_AGENT.md) | Stage 3a — Present Agent scene assessment |
| [FireEye/FUTURE_AGENT.md](FireEye/FUTURE_AGENT.md) | Stage 3b — Future Agent prediction model |
| [FireEye/VIDEO_GENERATION.md](FireEye/VIDEO_GENERATION.md) | Test data generation: video and escalation frames |
