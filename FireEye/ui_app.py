"""FireEye NiceGUI Dashboard
End-to-end pipeline showcase with expanded diagnostics.

Run from the FireEye/ directory with the project venv:
    cd FireEye && venv/bin/python ui_app.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
import uuid as _uuid_mod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure the FireEye package root is importable when run from any CWD
sys.path.insert(0, str(Path(__file__).parent))

# Load .env before app.config is imported — Settings() is instantiated at
# import time, so the env vars must be populated first.
from dotenv import load_dotenv
_env_file = Path(__file__).parent / ".env"
load_dotenv(_env_file, override=False)   # override=False: real env vars win

from nicegui import events, ui

from app.config import settings
from app.models.schemas import AnalysisResult
from app.pipeline import llm_agents, risk_classifier, yolo_detector
from app.storage import image_store

# ── Rengoku flame palette ──────────────────────────────────────────────────────
# Inspired by Kyojuro Rengoku (Demon Slayer) — deep crimson → flame orange → warm gold
# Fair use: purely decorative palette for a fire-detection product, no anime content.
RENGOKU_CSS = """
<style>
  :root {
    --rengoku-red:    #C91B00;
    --rengoku-orange: #E85D04;
    --rengoku-gold:   #FAA307;
    --rengoku-dark:   #1A0500;
    --rengoku-cream:  #FFF8E7;
  }
  .rengoku-header {
    background: linear-gradient(
      135deg,
      var(--rengoku-red)    0%,
      var(--rengoku-orange) 55%,
      var(--rengoku-gold)   100%
    ) !important;
  }
  .rengoku-stage-running {
    color: var(--rengoku-gold) !important;
  }
  .rengoku-accent-card {
    border-left: 4px solid var(--rengoku-orange) !important;
  }
  .rengoku-recommendation-card {
    background: linear-gradient(
      90deg,
      rgba(232,93,4,0.08) 0%,
      rgba(250,163,7,0.06) 100%
    ) !important;
    border-left: 4px solid var(--rengoku-gold) !important;
  }

  /* ── Mobile-responsive overrides ─────────────────────────────── */
  @media (max-width: 820px) {
    /* Stack the splitter vertically on mobile */
    .q-splitter {
      flex-direction: column !important;
      height: auto !important;
    }
    .q-splitter > .q-splitter__panel {
      width: 100% !important;
      height: auto !important;
      flex: none !important;
    }
    .q-splitter > .q-splitter__separator {
      display: none !important;
    }
    /* Left panel: disable scroll area machinery on mobile,
       let content flow naturally in the document. */
    .mobile-scroll-area {
      min-width: 0 !important;
    }
    /* Header: single line, compact */
    .rengoku-header {
      padding-left: 12px !important;
      padding-right: 12px !important;
      gap: 4px !important;
      flex-wrap: nowrap !important;
      min-height: 48px !important;
      height: 48px !important;
    }
    .mobile-header-title {
      font-size: 0.8rem !important;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
      flex-shrink: 1;
    }
    /* Hide dark mode label text on small screens */
    .mobile-dark-toggle .q-toggle__label {
      display: none !important;
    }
    /* Tabs: make scrollable and smaller */
    .q-tabs__content {
      flex-wrap: nowrap !important;
    }
    .q-tab__label {
      font-size: 0.7rem !important;
    }
    /* Images: full width, stack vertically */
    .mobile-images-container {
      flex-direction: column !important;
      gap: 16px !important;
    }
    .mobile-images-container .q-img,
    .mobile-images-container img {
      max-width: 100% !important;
      width: 100% !important;
    }
    /* Detection table: enable horizontal scroll */
    .q-table__container {
      overflow-x: auto !important;
    }
    /* Overview timing: wrap better */
    .mobile-timing {
      margin-left: 0 !important;
      width: 100%;
    }
  }
</style>
"""

# ── constants ──────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES    = 20 * 1024 * 1024   # 20 MB hard cap
ALLOWED_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

STAGE_DEFS: list[tuple[str, str]] = [
    ("search",        "YOLO Detection"),
    ("warning",       "Risk Classifier"),
    ("psychology",    "Present Agent"),
    ("auto_awesome",  "Future Agent"),
]

RISK_BADGE_COLOR: dict[str, str] = {
    "safe":     "positive",
    "low":      "positive",
    "medium":   "warning",
    "high":     "negative",
    "critical": "negative",
}

SEVERITY_COLOR: dict[str, str] = {
    "low":      "green",
    "medium":   "orange",
    "high":     "deep-orange",
    "critical": "red",
}

LIKELIHOOD_COLOR: dict[str, str] = {
    "unlikely": "blue-grey",
    "possible": "orange",
    "likely":   "deep-orange",
    "certain":  "red",
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _path_to_b64(path: str | Path) -> str:
    """Encode an image file to a base64 data-URI suitable for ui.image()."""
    p = Path(path)
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


# ── page ───────────────────────────────────────────────────────────────────────

@ui.page("/")
def index() -> None:
    # Map Rengoku flame colours onto Quasar's design-system tokens so that
    # every component (buttons, badges, tabs, progress bars, …) automatically
    # uses the brand palette without needing per-element overrides.
    ui.colors(
        primary="#E85D04",    # rengoku-orange  — buttons, active tabs, focus rings
        secondary="#FAA307",  # rengoku-gold    — secondary accents
        accent="#C91B00",     # rengoku-red     — highlight / danger accent
        positive="#22c55e",   # keep semantic green for success states
        negative="#ef4444",   # keep semantic red for error states
        warning="#f97316",    # keep semantic orange for warnings
    )
    ui.add_head_html(RENGOKU_CSS)
    ui.add_head_html('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">')

    # On mobile, the QScrollArea's absolute-positioned internals cause 0-height.
    # This script unwraps the scroll area, keeping the content in normal flow.
    ui.add_body_html("""<script>
    document.addEventListener('DOMContentLoaded', () => {
        if (window.innerWidth > 820) return;
        const check = () => {
            const sa = document.querySelector('.mobile-scroll-area');
            if (!sa) return setTimeout(check, 100);
            const content = sa.querySelector('.q-scrollarea__content');
            if (!content) return;
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'width:100%;';
            while (content.firstChild) wrapper.appendChild(content.firstChild);
            sa.parentElement.replaceChild(wrapper, sa);
        };
        check();
    });
    </script>""")

    # ── per-session mutable state ─────────────────────────────────────────────
    class S:
        def __init__(self) -> None:
            self.image_id: Optional[_uuid_mod.UUID] = None
            self.image_name: str = ""
            self.image_bytes: bytes = b""
            self.result: Optional[AnalysisResult] = None
            self.processing: bool = False
            self.current_stage: int = -1      # 0-3, -1 = idle
            self.stage_ms: list[float] = [0.0, 0.0, 0.0, 0.0]

    s = S()

    # ── header ────────────────────────────────────────────────────────────────
    with ui.header().classes(
        "rengoku-header text-white gap-3 px-6 py-3 items-center shadow-md"
    ):
        ui.icon("local_fire_department", size="sm")
        ui.label("FireEye — Fire Hazard Intelligence Dashboard").classes(
            "text-xl font-bold tracking-wide mobile-header-title"
        )
        ui.space()
        dark = ui.dark_mode()
        ui.switch("Dark mode").bind_value(dark, "value").props("color=white").classes("mobile-dark-toggle")

    # ── body: splitter layout ─────────────────────────────────────────────────
    with ui.splitter(value=26).classes("w-full") as splitter:

        # ── LEFT: upload form + pipeline status ───────────────────────────────
        with splitter.before:
            with ui.scroll_area().classes("w-full mobile-scroll-area").style("height: calc(100vh - 56px); min-width: 280px"):
                with ui.column().classes("p-4 gap-3 w-full"):

                    ui.label("Upload Image").classes("text-base font-semibold")

                    upload_widget = ui.upload(
                        label="Drop image here or click",
                        auto_upload=True,
                        max_files=1,
                    ).props('accept="image/*"').classes("w-full")

                    src_type = ui.select(
                        ["mobile", "cctv"], value="mobile", label="Source type"
                    ).classes("w-full")
                    src_id = ui.input(
                        "Source ID", value="inspector-1"
                    ).classes("w-full")
                    loc_input = ui.input(
                        "Location", value=""
                    ).classes("w-full")
                    notes_input = ui.textarea(
                        "Notes", value=""
                    ).props("rows=2").classes("w-full")

                    with ui.row().classes("w-full items-center gap-2 mt-1"):
                        analyze_btn = ui.button(
                            "Analyze", icon="manage_search", color="primary",
                        ).classes("flex-grow")
                        ui.spinner("dots", size="sm").bind_visibility_from(s, "processing")
                    analyze_btn.disable()

                    ui.separator()

                    # ── stage status indicators ───────────────────────────────
                    ui.label("Pipeline stages").classes(
                        "text-sm font-medium text-grey-7"
                    )

                    stage_dots: list[tuple[ui.icon, ui.label, ui.label]] = []
                    for _, stage_name in STAGE_DEFS:
                        with ui.row().classes("items-center gap-2 w-full"):
                            dot = ui.icon("circle", size="xs").classes("text-grey-4")
                            lbl = ui.label(stage_name).classes(
                                "text-sm text-grey-7 flex-grow"
                            )
                            ms_lbl = ui.label("").classes("text-xs text-grey-5")
                        stage_dots.append((dot, lbl, ms_lbl))

                    err_label = ui.label("").classes("text-negative text-sm")

        # ── RIGHT: results tab panels ─────────────────────────────────────────
        with splitter.after:
            with ui.column().classes("w-full gap-0"):

                with ui.tabs().classes("w-full bg-grey-1") as tabs:
                    tab_overview   = ui.tab("Overview",   icon="dashboard")
                    tab_images     = ui.tab("Images",     icon="image")
                    tab_detections = ui.tab("Detections", icon="label")
                    tab_assessment = ui.tab("Assessment", icon="psychology")
                    tab_raw        = ui.tab("Raw JSON",   icon="data_object")

                with ui.tab_panels(tabs, value=tab_overview).classes("w-full"):

                    # Overview ─────────────────────────────────────────────────
                    with ui.tab_panel(tab_overview):
                        overview_col = ui.column().classes("w-full gap-4 p-4")
                        with overview_col:
                            ui.label(
                                "Upload an image and click Analyze."
                            ).classes("text-grey-6")

                    # Images ───────────────────────────────────────────────────
                    with ui.tab_panel(tab_images):
                        images_row = ui.row().classes("w-full gap-6 flex-wrap p-4 mobile-images-container")
                        with images_row:
                            ui.label("No images yet.").classes("text-grey-6")

                    # Detections ───────────────────────────────────────────────
                    with ui.tab_panel(tab_detections):
                        det_col = ui.column().classes("w-full p-4")
                        with det_col:
                            ui.label("No detections yet.").classes("text-grey-6")

                    # Assessment ───────────────────────────────────────────────
                    with ui.tab_panel(tab_assessment):
                        assess_col = ui.column().classes("w-full gap-4 p-4")
                        with assess_col:
                            ui.label("No assessment yet.").classes("text-grey-6")

                    # Raw JSON ─────────────────────────────────────────────────
                    with ui.tab_panel(tab_raw):
                        raw_col = ui.column().classes("w-full p-4")
                        with raw_col:
                            ui.label("No data yet.").classes("text-grey-6")

    # ── stage helpers ─────────────────────────────────────────────────────────

    def _reset_stages() -> None:
        for dot, _, ms_lbl in stage_dots:
            dot.classes(replace="text-grey-4")
            ms_lbl.set_text("")

    def _mark_stage(idx: int, status: str, elapsed_ms: float | None = None) -> None:
        dot, _, ms_lbl = stage_dots[idx]
        color = {
            "pending": "text-grey-4",
            "running": "rengoku-stage-running",
            "done":    "text-green-600",
            "error":   "text-red-600",
        }[status]
        dot.classes(replace=color)
        if elapsed_ms is not None:
            ms_lbl.set_text(f"{elapsed_ms:.0f} ms")

    # ── render helpers ─────────────────────────────────────────────────────────

    def _render_overview(result: AnalysisResult) -> None:
        overview_col.clear()
        with overview_col:
            risk  = result.risk_classification
            level = risk.risk_level.value if risk else "unknown"

            # ── hero row: risk badge + confidence + timings ────────────────────
            with ui.row().classes("items-center gap-3 flex-wrap w-full"):
                ui.badge(level.upper(), color=RISK_BADGE_COLOR.get(level, "grey")).classes(
                    "text-sm px-3 py-1"
                )
                if risk:
                    ui.label(f"Confidence: {risk.confidence:.0%}").classes(
                        "text-grey-7 text-sm"
                    )
                timing_str = "  ·  ".join(
                    f"{name}: {s.stage_ms[i]:.0f} ms"
                    for i, (_, name) in enumerate(STAGE_DEFS)
                )
                ui.label(timing_str).classes("text-xs text-grey-5 ml-auto mobile-timing")

            # ── risk reason ───────────────────────────────────────────────────
            if risk:
                with ui.card().classes("w-full rengoku-accent-card"):
                    ui.label("Risk reason").classes(
                        "text-sm font-semibold text-grey-7"
                    )
                    ui.separator()
                    ui.label(risk.reason).classes("mt-1")

            # ── present assessment ────────────────────────────────────────────
            pa = result.present_assessment
            if pa:
                with ui.card().classes("w-full"):
                    ui.label("Scene summary").classes("font-semibold")
                    ui.separator()
                    ui.label(pa.summary).classes("mt-1")

                    if pa.hazards:
                        ui.label("Detected hazards").classes(
                            "text-sm font-semibold text-orange-8 mt-3"
                        )
                        for h in pa.hazards:
                            with ui.row().classes("items-start gap-1"):
                                ui.icon("warning_amber", size="xs", color="orange")
                                ui.label(h).classes("text-sm")

                    if pa.distances:
                        ui.label("Spatial relationships").classes(
                            "text-sm font-semibold text-blue mt-3"
                        )
                        for d in pa.distances:
                            with ui.row().classes("items-start gap-1"):
                                ui.icon("social_distance", size="xs", color="blue")
                                ui.label(d).classes("text-sm")

            # ── recommendation ────────────────────────────────────────────────
            fp = result.future_prediction
            if fp:
                with ui.card().classes("w-full rengoku-recommendation-card"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("local_fire_department").style(
                            "color: var(--rengoku-orange)"
                        )
                        ui.label("Recommendation").classes("font-semibold")
                    ui.label(fp.recommendation).classes("mt-1")

    def _render_images(image_id: _uuid_mod.UUID) -> None:
        images_row.clear()
        with images_row:
            orig = image_store.get_input_image_path(image_id)
            if orig:
                with ui.column().classes("items-center gap-1"):
                    ui.label("Original").classes(
                        "text-sm font-medium text-grey-7"
                    )
                    ui.image(_path_to_b64(orig)).classes(
                        "rounded shadow max-h-96 object-contain"
                    ).style("max-width: 480px")

            ann = image_store.get_output_image_path(image_id, "annotated")
            if ann:
                with ui.column().classes("items-center gap-1"):
                    ui.label("YOLO Annotated").classes(
                        "text-sm font-medium text-grey-7"
                    )
                    ui.image(_path_to_b64(ann)).classes(
                        "rounded shadow max-h-96 object-contain"
                    ).style("max-width: 480px")

    def _render_detections(result: AnalysisResult) -> None:
        det_col.clear()
        with det_col:
            dets = result.detections
            if not dets:
                ui.label("No objects detected.").classes("text-grey-6")
                return

            ui.label(f"{len(dets)} detection(s)").classes("font-semibold mb-2")

            columns = [
                {"name": "label",      "label": "Label",      "field": "label",      "align": "left",   "sortable": True},
                {"name": "confidence", "label": "Confidence", "field": "confidence", "align": "center", "sortable": True,
                 "format": "val => (val * 100).toFixed(1) + '%'"},
                {"name": "x1",         "label": "x1",         "field": "x1",         "align": "right"},
                {"name": "y1",         "label": "y1",         "field": "y1",         "align": "right"},
                {"name": "x2",         "label": "x2",         "field": "x2",         "align": "right"},
                {"name": "y2",         "label": "y2",         "field": "y2",         "align": "right"},
            ]
            rows = [
                {
                    "id":         i,
                    "label":      d.label,
                    "confidence": d.confidence,
                    "x1":         f"{d.bbox.x1:.1f}",
                    "y1":         f"{d.bbox.y1:.1f}",
                    "x2":         f"{d.bbox.x2:.1f}",
                    "y2":         f"{d.bbox.y2:.1f}",
                }
                for i, d in enumerate(dets)
            ]
            ui.table(columns=columns, rows=rows, row_key="id").classes(
                "w-full"
            ).props("flat bordered dense")

    def _render_assessment(result: AnalysisResult) -> None:
        assess_col.clear()
        with assess_col:
            fp = result.future_prediction
            if not fp:
                ui.label("No assessment available.").classes("text-grey-6")
                return

            ui.label("Future Scenarios").classes("text-lg font-semibold")

            for sc in fp.scenarios:
                sev_col  = SEVERITY_COLOR.get(sc.severity, "grey")
                lik_col  = LIKELIHOOD_COLOR.get(sc.likelihood, "blue-grey")
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.badge(sc.severity.upper(),  color=sev_col)
                        ui.badge(sc.likelihood,        color=lik_col)
                        ui.badge(f"⏱ {sc.time_horizon}", color="grey")
                    ui.label(sc.scenario).classes("mt-2 text-sm")

            ui.separator()

            ov = fp.overall_risk.value
            with ui.row().classes("items-center gap-3 mt-1"):
                ui.label("Overall projected risk:").classes(
                    "text-sm font-semibold text-grey-7"
                )
                ui.badge(ov.upper(), color=RISK_BADGE_COLOR.get(ov, "grey")).classes(
                    "text-sm px-2 py-1"
                )

    def _render_raw(result: AnalysisResult) -> None:
        raw_col.clear()
        with raw_col:
            payload = json.dumps(
                result.model_dump(mode="json", exclude={"annotated_image_path"}),
                indent=2, default=str,
            )
            ui.code(payload, language="json").classes("w-full text-xs")

    # ── upload handler ─────────────────────────────────────────────────────────

    async def on_upload(e: events.UploadEventArguments) -> None:
        ext = Path(e.file.name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            ui.notify(f"Unsupported file type '{ext}'. Use jpg/png/webp/bmp.", type="negative")
            return
        data = await e.file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            ui.notify(
                f"File too large ({len(data):,} bytes). Maximum is 20 MB.", type="negative"
            )
            return
        s.image_bytes = data
        s.image_name  = e.file.name
        analyze_btn.enable()
        ui.notify(
            f"Loaded {e.file.name} ({len(data):,} bytes)",
            type="positive",
        )

    upload_widget.on_upload(on_upload)

    # ── analyze handler ────────────────────────────────────────────────────────

    async def on_analyze() -> None:
        if not s.image_bytes:
            ui.notify("Upload an image first.", type="warning")
            return
        if s.processing:
            return

        s.processing = True
        err_label.set_text("")
        analyze_btn.disable()
        _reset_stages()

        # Persist image to the FireEye input store
        s.image_id = _uuid_mod.uuid4()
        image_store.store_input_image(s.image_id, s.image_name, s.image_bytes)
        image_path = image_store.get_input_image_path(s.image_id)

        try:
            result = AnalysisResult(image_id=s.image_id)

            # ── Stage 0: YOLO ──────────────────────────────────────────────────
            s.current_stage = 0
            _mark_stage(0, "running")
            t0 = time.perf_counter()
            ann_path = image_store.get_annotated_path(s.image_id)
            detections = await asyncio.to_thread(
                yolo_detector.detect_and_annotate, image_path, ann_path
            )
            s.stage_ms[0] = (time.perf_counter() - t0) * 1000
            result.detections = detections
            result.annotated_image_path = str(ann_path)
            _mark_stage(0, "done", s.stage_ms[0])

            # ── Stage 1: Risk classifier ───────────────────────────────────────
            s.current_stage = 1
            _mark_stage(1, "running")
            t0 = time.perf_counter()
            risk = await asyncio.to_thread(
                risk_classifier.classify_with_llm, str(image_path), detections
            )
            s.stage_ms[1] = (time.perf_counter() - t0) * 1000
            result.risk_classification = risk
            _mark_stage(1, "done", s.stage_ms[1])

            # ── Stage 2: Present agent ─────────────────────────────────────────
            s.current_stage = 2
            _mark_stage(2, "running")
            t0 = time.perf_counter()
            present = await asyncio.to_thread(
                llm_agents.assess_present, str(image_path), detections, risk
            )
            s.stage_ms[2] = (time.perf_counter() - t0) * 1000
            result.present_assessment = present
            _mark_stage(2, "done", s.stage_ms[2])

            # ── Stage 3: Future agent ──────────────────────────────────────────
            s.current_stage = 3
            _mark_stage(3, "running")
            t0 = time.perf_counter()
            future = await asyncio.to_thread(
                llm_agents.predict_future, str(image_path), detections, risk, present
            )
            s.stage_ms[3] = (time.perf_counter() - t0) * 1000
            result.future_prediction = future
            _mark_stage(3, "done", s.stage_ms[3])

            s.result = result
            _render_overview(result)
            _render_images(s.image_id)
            _render_detections(result)
            _render_assessment(result)
            _render_raw(result)
            tabs.set_value(tab_overview)
            ui.notify("Analysis complete!", type="positive")

        except Exception as exc:
            logger.exception("Pipeline failed at stage %d", s.current_stage)
            if s.current_stage >= 0:
                _mark_stage(s.current_stage, "error")
            stage_name = STAGE_DEFS[s.current_stage][1] if s.current_stage >= 0 else "startup"
            msg = f"{stage_name} error: {type(exc).__name__}"
            err_label.set_text(msg)
            ui.notify(msg, type="negative")

        finally:
            s.current_stage = -1
            s.processing = False
            analyze_btn.enable()

    analyze_btn.on_click(on_analyze)


# ── entry point ────────────────────────────────────────────────────────────────

if not settings.openrouter_api_key:
    raise RuntimeError(
        "FIREEYE_OPENROUTER_API_KEY is not set. "
        "Copy .env.example to .env and fill in your key."
    )

ui.run(
    title="FireEye Dashboard",
    host="0.0.0.0",
    port=8090,
    favicon="🔥",
    show=False,
)
