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
from app.pipeline.spatial import compute_distances, estimate_scale
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
  .compliance-gauge {
    width: 80px; height: 80px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 1.25rem;
  }
  .compliance-good   { background: conic-gradient(#22c55e var(--pct), #e5e7eb var(--pct)); }
  .compliance-warn   { background: conic-gradient(#f97316 var(--pct), #e5e7eb var(--pct)); }
  .compliance-bad    { background: conic-gradient(#ef4444 var(--pct), #e5e7eb var(--pct)); }
  .compliance-inner  {
    width: 60px; height: 60px; border-radius: 50%;
    background: white; display: flex; align-items: center;
    justify-content: center; font-size: 0.9rem; font-weight: 700;
  }
  .dark .compliance-inner { background: #1d1d1d; }
  .flag-present { color: #22c55e; }
  .flag-absent  { color: #ef4444; }
  .flag-unclear { color: #f97316; }
  .proximity-concern { background: rgba(239,68,68,0.08) !important; }
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
            "text-xl font-bold tracking-wide"
        )
        ui.space()
        dark = ui.dark_mode()
        ui.switch("Dark mode").bind_value(dark, "value").props("color=white")

    # ── body: splitter layout ─────────────────────────────────────────────────
    with ui.splitter(value=26).classes("w-full") as splitter:

        # ── LEFT: upload form + pipeline status ───────────────────────────────
        with splitter.before:
            with ui.scroll_area().classes("w-full").style("height: calc(100vh - 56px); min-width: 280px"):
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

                    # LLM backend selector
                    _has_local = bool(settings.local_llm_url)
                    _backend_options = ["openrouter"]
                    if _has_local:
                        _backend_options.append("local")
                    backend_select = ui.select(
                        _backend_options,
                        value=settings.llm_backend if settings.llm_backend in _backend_options else "openrouter",
                        label="LLM Backend",
                    ).classes("w-full").tooltip(
                        f"Local: {settings.local_llm_model or 'N/A'}" if _has_local
                        else "Set FIREEYE_LOCAL_LLM_URL to enable local"
                    )

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
                    tab_overview    = ui.tab("Overview",    icon="dashboard")
                    tab_images      = ui.tab("Images",      icon="image")
                    tab_detections  = ui.tab("Detections",  icon="label")
                    tab_compliance  = ui.tab("Compliance",  icon="verified")
                    tab_assessment  = ui.tab("Assessment",  icon="psychology")
                    tab_audit       = ui.tab("Audit Log",   icon="history")
                    tab_raw         = ui.tab("Raw JSON",    icon="data_object")

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
                        images_row = ui.row().classes("w-full gap-6 flex-wrap p-4")
                        with images_row:
                            ui.label("No images yet.").classes("text-grey-6")

                    # Detections ───────────────────────────────────────────────
                    with ui.tab_panel(tab_detections):
                        det_col = ui.column().classes("w-full p-4")
                        with det_col:
                            ui.label("No detections yet.").classes("text-grey-6")

                    # Compliance ─────────────────────────────────────────────
                    with ui.tab_panel(tab_compliance):
                        compliance_col = ui.column().classes("w-full gap-4 p-4")
                        with compliance_col:
                            ui.label("No compliance data yet.").classes("text-grey-6")

                    # Assessment ───────────────────────────────────────────────
                    with ui.tab_panel(tab_assessment):
                        assess_col = ui.column().classes("w-full gap-4 p-4")
                        with assess_col:
                            ui.label("No assessment yet.").classes("text-grey-6")

                    # Audit Log ────────────────────────────────────────────────
                    with ui.tab_panel(tab_audit):
                        audit_col = ui.column().classes("w-full gap-4 p-4")
                        with audit_col:
                            ui.label("No audit logs yet.").classes("text-grey-6")

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
                ui.label(timing_str).classes("text-xs text-grey-5 ml-auto")

            # ── backend info ─────────────────────────────────────────────
            backend = settings.llm_backend
            model = settings.local_llm_model if backend == "local" else settings.llm_model
            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon("smart_toy", size="xs").classes("text-grey-5")
                ui.label(f"LLM: {model}").classes("text-xs text-grey-5")
                ui.badge(
                    "LOCAL" if backend == "local" else "CLOUD",
                    color="blue" if backend == "local" else "grey",
                ).props("dense").classes("text-xs")

            # ── risk reason ───────────────────────────────────────────────────
            if risk:
                with ui.card().classes("w-full rengoku-accent-card"):
                    ui.label("Risk reason").classes(
                        "text-sm font-semibold text-grey-7"
                    )
                    ui.separator()
                    ui.label(risk.reason).classes("mt-1")

            # ── compliance score (inline) ────────────────────────────────────
            pa = result.present_assessment
            if pa:
                score = pa.compliance_score
                pct = f"{score * 100:.0f}%"
                gauge_cls = (
                    "compliance-good" if score >= 0.8
                    else "compliance-warn" if score >= 0.5
                    else "compliance-bad"
                )
                with ui.row().classes("items-center gap-4 w-full"):
                    with ui.element("div").classes(f"compliance-gauge {gauge_cls}").style(
                        f"--pct: {score * 100:.0f}%"
                    ):
                        with ui.element("div").classes("compliance-inner"):
                            ui.label(pct)
                    with ui.column().classes("gap-1"):
                        ui.label("Compliance Score").classes("font-semibold")
                        issues = pa.compliance_issues
                        if issues:
                            ui.label(f"{len(issues)} issue(s) found").classes(
                                "text-sm text-negative"
                            )
                        else:
                            ui.label("All checks passed").classes(
                                "text-sm text-positive"
                            )

            # ── present assessment ────────────────────────────────────────────
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

    def _render_compliance(result: AnalysisResult) -> None:
        compliance_col.clear()
        with compliance_col:
            pa = result.present_assessment
            if not pa or not pa.compliance_flags:
                ui.label("No compliance flags available.").classes("text-grey-6")
                return

            # Score summary
            score = pa.compliance_score
            score_color = (
                "positive" if score >= 0.8
                else "warning" if score >= 0.5
                else "negative"
            )
            with ui.row().classes("items-center gap-3 w-full"):
                ui.badge(
                    f"Score: {score:.0%}", color=score_color
                ).classes("text-sm px-3 py-1")
                ui.label(
                    f"{len(pa.compliance_flags)} item(s) checked"
                ).classes("text-sm text-grey-7")

            # Compliance flags table
            FLAG_ICON = {
                "present": ("check_circle", "positive"),
                "absent": ("cancel", "negative"),
                "unclear": ("help", "warning"),
            }

            ui.label("Regulatory Compliance Flags").classes(
                "text-base font-semibold mt-2"
            )
            columns = [
                {"name": "status_icon", "label": "", "field": "status_icon", "align": "center"},
                {"name": "item", "label": "Item", "field": "item", "align": "left", "sortable": True},
                {"name": "status", "label": "Status", "field": "status", "align": "center", "sortable": True},
                {"name": "note", "label": "Note", "field": "note", "align": "left"},
            ]
            rows = []
            for i, f in enumerate(pa.compliance_flags):
                icon_name, _ = FLAG_ICON.get(f.status, ("help", "grey"))
                rows.append({
                    "id": i,
                    "status_icon": icon_name,
                    "item": f.item,
                    "status": f.status.upper(),
                    "note": f.note or "—",
                })
            ui.table(columns=columns, rows=rows, row_key="id").classes(
                "w-full"
            ).props("flat bordered dense")

            # Issues summary
            issues = pa.compliance_issues
            if issues:
                ui.label("Issues Requiring Attention").classes(
                    "text-base font-semibold text-negative mt-4"
                )
                for issue in issues:
                    with ui.row().classes("items-start gap-1"):
                        ui.icon("error_outline", size="xs", color="red")
                        ui.label(issue).classes("text-sm")

    def _render_audit_log() -> None:
        audit_col.clear()
        with audit_col:
            audit_dir = settings.base_dir / "audit_logs"
            if not audit_dir.exists():
                ui.label("No audit logs found.").classes("text-grey-6")
                return

            # Find all JSONL files, sorted newest first
            files = sorted(audit_dir.glob("audit_*.jsonl"), reverse=True)
            if not files:
                ui.label("No audit log entries yet.").classes("text-grey-6")
                return

            ui.label(f"{len(files)} audit log file(s)").classes(
                "text-sm text-grey-7"
            )

            # Read and display the most recent entries (up to 50)
            entries: list[dict] = []
            for f in files:
                for line in f.read_text().strip().split("\n"):
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                if len(entries) >= 50:
                    break
            entries.reverse()  # newest first

            if not entries:
                ui.label("Audit files exist but contain no valid entries.").classes(
                    "text-grey-6"
                )
                return

            columns = [
                {"name": "time", "label": "Time", "field": "time", "align": "left", "sortable": True},
                {"name": "image", "label": "Image", "field": "image", "align": "left"},
                {"name": "risk", "label": "Risk", "field": "risk", "align": "center", "sortable": True},
                {"name": "compliance", "label": "Compliance", "field": "compliance", "align": "center"},
                {"name": "detections", "label": "Detections", "field": "detections", "align": "center"},
                {"name": "total_s", "label": "Time (s)", "field": "total_s", "align": "right", "sortable": True},
                {"name": "error", "label": "Error", "field": "error", "align": "left"},
            ]
            rows = []
            for i, e in enumerate(entries[:50]):
                cs = e.get("compliance_score")
                rows.append({
                    "id": i,
                    "time": e.get("started_at", "?")[:19].replace("T", " "),
                    "image": e.get("image_name", "?"),
                    "risk": (e.get("risk_level") or "?").upper(),
                    "compliance": f"{cs:.0%}" if cs is not None else "—",
                    "detections": e.get("detection_count", 0),
                    "total_s": f"{e.get('total_time_s', 0):.1f}",
                    "error": e.get("error") or "—",
                })
            ui.table(columns=columns, rows=rows, row_key="id").classes(
                "w-full"
            ).props("flat bordered dense")

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

        # Apply selected LLM backend
        settings.llm_backend = backend_select.value

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
            _render_compliance(result)
            _render_assessment(result)
            _render_audit_log()
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

if not settings.openrouter_api_key and not settings.local_llm_url:
    raise RuntimeError(
        "No LLM backend configured. Either set FIREEYE_OPENROUTER_API_KEY "
        "or set FIREEYE_LOCAL_LLM_URL (e.g. http://localhost:11434). "
        "See .env.example for details."
    )

ui.run(
    title="FireEye Dashboard",
    host="0.0.0.0",
    port=8090,
    favicon="🔥",
    show=False,
)
