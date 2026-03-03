# FireEye Dashboard

A NiceGUI 3.8 web application that runs the full FireEye pipeline end-to-end and surfaces expanded diagnostics across five result tabs. Designed as a standalone demo/diagnostic tool — it imports the pipeline modules directly, bypassing the FastAPI layer.

---

## Quick Start

```bash
cd FireEye

# Create and populate .env if you haven't already
echo "FIREEYE_OPENROUTER_API_KEY=sk-or-v1-..." > .env

# Activate the venv (created separately — see Setup below)
source venv/bin/activate

# Launch
python ui_app.py
```

Open **http://localhost:8090** in your browser.

---

## Setup

The dashboard has its own virtual environment with NiceGUI, Playwright, and all backend dependencies installed.

```bash
cd FireEye

# Create venv
python3 -m venv venv

# Install everything
venv/bin/pip install "nicegui==3.8.0"
venv/bin/pip install playwright
venv/bin/playwright install chromium
venv/bin/pip install -r requirements.txt
```

---

## User Interface

```
┌──────── Header: Rengoku flame gradient · FireEye title · Dark mode toggle ────────┐
│                                                                                     │
├──── Left pane (26%) ───────────────┬──── Right pane (74%) ────────────────────────┤
│                                    │                                                │
│  Upload Image                      │  [ Overview | Images | Detections |           │
│  ┌─────────────────────────────┐   │    Assessment | Raw JSON ]                    │
│  │  Drop image here or click   │   │                                                │
│  └─────────────────────────────┘   │  (tab content rendered after analysis)        │
│                                    │                                                │
│  Source type  [mobile ▾]           │                                                │
│  Source ID    [inspector-1    ]    │                                                │
│  Location     [               ]    │                                                │
│  Notes        [               ]    │                                                │
│                                    │                                                │
│  [ ANALYZE ]                       │                                                │
│                                    │                                                │
│  Pipeline stages                   │                                                │
│  ○  YOLO Detection                 │                                                │
│  ○  Risk Classifier                │                                                │
│  ○  Present Agent                  │                                                │
│  ○  Future Agent                   │                                                │
└────────────────────────────────────┴────────────────────────────────────────────────┘
```

### Stage indicator colours

| State   | Colour |
|---------|--------|
| Pending | Grey   |
| Running | Rengoku gold `#FAA307` |
| Done    | Green — with elapsed ms |
| Error   | Red    |

---

## Result Tabs

### Overview
The primary summary view, shown automatically when analysis completes.

- **Risk badge** — level (`SAFE` / `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`) with confidence percentage
- **Stage timings** — ms elapsed for each of the four pipeline stages
- **Risk reason card** — the LLM's one-sentence justification (left-bordered in Rengoku orange)
- **Scene summary card** — factual description from the Present Agent, plus detected hazards (⚠) and spatial relationships (↔)
- **Recommendation card** — the Future Agent's action recommendation (flame-gradient left border)

### Images
Side-by-side comparison of:
- **Original** — the uploaded image
- **YOLO Annotated** — the same frame with bounding boxes drawn by the detector

Both are encoded as base64 data-URIs so no separate file-serving route is needed.

### Detections
Sortable table of every object YOLO found, with columns:

| Label | Confidence | x1 | y1 | x2 | y2 |
|-------|------------|----|----|----|----|
| fire  | 87.3%      | … | … | … | … |

### Assessment
Future Agent scenario cards. Each card shows:
- **Severity badge** — colour-coded red/orange/green
- **Likelihood badge** — unlikely / possible / likely / certain
- **Time horizon badge** — seconds / minutes / hours
- Scenario description text

Footer shows the Future Agent's `overall_risk` projection.

### Raw JSON
Full `AnalysisResult` model dump formatted as pretty-printed JSON, rendered in a syntax-highlighted code block. Useful for debugging or copying output to other tools.

---

## Pipeline Execution

All four pipeline functions are synchronous and CPU/network-bound. To keep the UI responsive they are dispatched to a thread pool via `asyncio.to_thread()`:

```
on_analyze() — async NiceGUI handler
│
├── asyncio.to_thread(yolo_detector.detect_and_annotate, ...)  → Stage 0 done ✓
├── asyncio.to_thread(risk_classifier.classify_with_llm, ...)  → Stage 1 done ✓
├── asyncio.to_thread(llm_agents.assess_present, ...)          → Stage 2 done ✓
└── asyncio.to_thread(llm_agents.predict_future, ...)          → Stage 3 done ✓
```

The stage indicator dots update live as each `await` completes, so the user can watch the pipeline progress in real time.

The uploaded image is written to `images/input/` via `image_store.store_input_image()` (same store used by the FastAPI backend) before the pipeline starts. The annotated output is written to `images/output/`.

---

## Brand: Rengoku Flame Palette

The dashboard uses the Kyojuro Rengoku (Demon Slayer) colour palette as FireEye's brand identity. The palette maps naturally to fire — purely decorative use, no anime content in the product.

| Token | Hex | Usage |
|-------|-----|-------|
| Crimson | `#C91B00` | Header gradient start |
| Orange  | `#E85D04` | Header gradient mid · Analyze button · accent card borders |
| Gold    | `#FAA307` | Header gradient end · running-stage dot |

The gradient is injected as a CSS custom-property block via `ui.add_head_html()` on page load:

```css
:root {
  --rengoku-red:    #C91B00;
  --rengoku-orange: #E85D04;
  --rengoku-gold:   #FAA307;
}
.rengoku-header {
  background: linear-gradient(135deg, #C91B00 0%, #E85D04 55%, #FAA307 100%);
}
```

---

## Visual QA

`test_ui.py` runs a headless Playwright suite against a live instance of the app.

```bash
venv/bin/python test_ui.py
```

### What it checks

| # | Check |
|---|-------|
| 1 | Page title contains "FireEye" |
| 2 | Header element contains "FireEye" |
| 3 | Rengoku gradient CSS class is applied to the header |
| 4–8 | All five result tabs are present (Overview, Images, Detections, Assessment, Raw JSON) |
| 9–12 | All four pipeline stage labels are visible |
| 13 | Analyze button is **disabled** before an image is uploaded |
| 14 | Analyze button is **enabled** after a test image is programmatically uploaded |

Screenshots are saved to `qa_screenshots/` on every run:

| File | When |
|------|------|
| `01_initial.png` | Immediately after the page loads |
| `02_image_loaded.png` | After uploading a test image |

The test script polls `localhost:8090` until the port accepts connections (up to 12 s) before launching Playwright, avoiding race conditions from slow startup.

---

## File Reference

```
FireEye/
├── ui_app.py              # NiceGUI dashboard — main entry point
├── test_ui.py             # Playwright headless visual QA
├── qa_screenshots/
│   ├── 01_initial.png     # Baseline screenshot — empty state
│   └── 02_image_loaded.png  # Baseline screenshot — image staged for analysis
└── venv/                  # Python virtual environment (not committed)
```

---

## Configuration

The dashboard inherits all settings from `app/config.py` (same `.env` file as the FastAPI backend). The only required setting is:

```
FIREEYE_OPENROUTER_API_KEY=sk-or-v1-...
```

The dashboard itself adds one constant:

| Setting | Value | Notes |
|---------|-------|-------|
| Port | `8090` | Hard-coded in `ui_app.py` to avoid clashing with the FastAPI server on `8000` |
| Host | `0.0.0.0` | Accessible on the local network |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `nicegui` | 3.8.0 | Web UI framework (Quasar + FastAPI + WebSocket) |
| `playwright` | 1.58.0 | Headless browser for visual QA |
| All `requirements.txt` deps | — | FireEye pipeline (YOLO, OpenAI SDK, OpenCV, …) |
