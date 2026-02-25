"""FireEye — Future Agent stress-test on parametric synthetic scenes.

Generates (or reuses) 4 synthetic images that isolate the key variables the
Future Agent reasons about, then runs the full pipeline and prints a side-by-
side comparison of what the Future Agent predicted for each scene.

Scenes (defined in generate_test_scenes.py):
  A — Small contained flame, isolated         (expected: LOW)
  B — Small flame + wood stack 80 px away    (expected: MEDIUM)
  C — Large roaring flame + embers, open     (expected: HIGH)
  D — Large flame + embers + materials + wind (expected: CRITICAL)

Usage:
    python3 test_future_prediction.py [--skip-generate] [--scene A,B,C,D]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from uuid import uuid4

# Make sure app/ is importable when run from the FireEye directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.models.schemas import RiskLevel
from app.pipeline import orchestrator
from app.storage import image_store

SYNTHETIC_DIR = Path(__file__).resolve().parent / "test_data" / "synthetic"

# Map scene letter → (filename, expected_future_risk)
SCENE_MANIFEST: dict[str, tuple[str, RiskLevel]] = {
    "A": ("scene_A_small_contained_clean.png",     RiskLevel.low),
    "B": ("scene_B_small_near_material_clean.png",  RiskLevel.medium),
    "C": ("scene_C_large_roaring_open_clean.png",   RiskLevel.high),
    "D": ("scene_D_critical_cascade_clean.png",     RiskLevel.critical),
}


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

_RISK_COLOUR = {
    "safe":     "\033[92m",   # green
    "low":      "\033[96m",   # cyan
    "medium":   "\033[93m",   # yellow
    "high":     "\033[91m",   # red
    "critical": "\033[95m",   # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _colour_risk(level: str) -> str:
    return f"{_RISK_COLOUR.get(level, '')}{_BOLD}{level.upper()}{_RESET}"


def _wrap(text: str, width: int = 90, indent: str = "      ") -> str:
    return textwrap.fill(text, width=width, subsequent_indent=indent)


def _separator(char: str = "─", n: int = 90) -> str:
    return char * n


# ─────────────────────────────────────────────────────────────────────────────
# Single scene runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scene(letter: str, img_path: Path, expected_future: RiskLevel) -> dict:
    """Run the full FireEye pipeline on one synthetic scene and return results."""
    image_id = uuid4()
    image_store.store_input_image(image_id, img_path.name, img_path.read_bytes())

    t0 = time.time()
    try:
        result = orchestrator.analyze_image(image_id)
    finally:
        image_store.cleanup_image(image_id)
    elapsed = time.time() - t0

    risk  = result.risk_classification
    pres  = result.present_assessment
    fut   = result.future_prediction

    actual_future = fut.overall_risk if fut else RiskLevel.safe
    passed = actual_future in {
        # Accept one level of slack in each direction
        expected_future,
        RiskLevel(expected_future.value),
    }
    # Strict: exact match
    exact = (actual_future == expected_future)

    return {
        "letter":          letter,
        "file":            img_path.name,
        "elapsed":         elapsed,
        "detections":      result.detections,
        "risk_level":      risk.risk_level.value if risk else "?",
        "risk_reason":     risk.reason if risk else "",
        "present_summary": pres.summary if pres else "",
        "present_hazards": pres.hazards if pres else [],
        "present_distances": pres.distances if pres else [],
        "future_overall":  actual_future.value,
        "future_scenarios": fut.scenarios if fut else [],
        "future_rec":      fut.recommendation if fut else "",
        "expected_future": expected_future.value,
        "exact_match":     exact,
        "result":          result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report renderer
# ─────────────────────────────────────────────────────────────────────────────

def print_scene_report(r: dict) -> None:
    """Print a detailed report for one scene."""
    print()
    print(_separator("═"))
    print(f"{_BOLD}SCENE {r['letter']}  —  {r['file']}{_RESET}")
    print(_separator("─"))

    # YOLO
    dets = r["detections"]
    if dets:
        top_labels = ", ".join(f"{d.label}({d.confidence:.0%})" for d in dets[:6])
        print(f"  YOLO detections  : {len(dets)}  [{top_labels}]")
    else:
        print(f"  YOLO detections  : 0  (synthetic image — none expected)")

    # Stage 2 risk
    print(f"  Scene risk level : {_colour_risk(r['risk_level'])}")
    print(f"  Risk reason      : {_wrap(r['risk_reason'], indent='                     ')}")

    # Stage 3 — Present
    print()
    print(f"  {_BOLD}Present Agent{_RESET}")
    print(f"    Summary  : {_wrap(r['present_summary'], indent='               ')}")
    if r["present_hazards"]:
        for h in r["present_hazards"]:
            print(f"    Hazard   · {h}")
    if r["present_distances"]:
        for d in r["present_distances"]:
            print(f"    Distance · {d}")

    # Stage 3 — Future
    print()
    print(f"  {_BOLD}Future Agent{_RESET}  →  overall risk: {_colour_risk(r['future_overall'])}  "
          f"(expected: {_colour_risk(r['expected_future'])})")
    for i, s in enumerate(r["future_scenarios"], 1):
        print(f"    [{i}] {s.scenario}")
        print(f"         likelihood={s.likelihood}  "
              f"severity={s.severity}  "
              f"horizon={s.time_horizon}")
    print(f"  Recommendation : {_wrap(r['future_rec'], indent='                 ')}")

    match_str = f"{_BOLD}\033[92mEXACT MATCH{_RESET}" if r["exact_match"] else f"\033[93mOFF (predicted {r['future_overall'].upper()}){_RESET}"
    print(f"  Match          : {match_str}    ({r['elapsed']:.1f}s)")


def print_comparison_table(results: list[dict]) -> None:
    """Print a compact side-by-side comparison of Future Agent outputs."""
    print()
    print(_separator("═"))
    print(f"{_BOLD}FUTURE AGENT COMPARISON TABLE{_RESET}")
    print(_separator("─"))
    fmt = "  {:<8} {:<22} {:<12} {:<12} {:<10} {:<10}"
    print(fmt.format("Scene", "Key Scenario #1", "Likelihood", "Severity", "Predicted", "Expected"))
    print(_separator("─"))
    for r in results:
        scens = r["future_scenarios"]
        s1_text  = scens[0].scenario[:22] if scens else "(none)"
        s1_like  = scens[0].likelihood[:12] if scens else "—"
        s1_sev   = scens[0].severity[:12] if scens else "—"
        pred     = r["future_overall"].upper()
        exp      = r["expected_future"].upper()
        match_ch = "✓" if r["exact_match"] else "~"
        print(fmt.format(
            f"[{r['letter']}] {match_ch}",
            s1_text,
            s1_like,
            s1_sev,
            pred,
            exp,
        ))
    print(_separator("─"))

    exact   = sum(1 for r in results if r["exact_match"])
    total   = len(results)
    print(f"  Exact matches: {exact}/{total}")

    # Highlight spread: A→D should show escalating risk
    predicted_order = [r["future_overall"] for r in results]
    level_order     = ["safe", "low", "medium", "high", "critical"]
    indices         = [level_order.index(p) for p in predicted_order if p in level_order]
    monotone        = all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
    trend_str = (
        f"{_BOLD}\033[92mMONOTONE ESCALATION{_RESET} — Future Agent correctly senses worsening scenes"
        if monotone else
        f"\033[93mNON-MONOTONE — Future Agent did not escalate risk A→D cleanly{_RESET}"
    )
    print(f"  Risk trend A→D : {trend_str}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test FireEye Future Agent on parametric synthetic scenes"
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip image generation (reuse existing files in test_data/synthetic/)",
    )
    parser.add_argument(
        "--scenes", default="A,B,C,D",
        help="Comma-separated scene letters to run  [default: A,B,C,D]",
    )
    args = parser.parse_args()

    # ── Generate images ──────────────────────────────────────────────────────
    if not args.skip_generate:
        print("Generating synthetic scenes …")
        # Import and run the generator in-process
        import generate_test_scenes as gen
        SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        for filename, (fn, _) in gen.SCENES.items():
            fn(SYNTHETIC_DIR / filename)
        print()

    # ── Sanity check ─────────────────────────────────────────────────────────
    print(_separator("═"))
    print("FireEye — Synthetic Scene Future Prediction Test")
    print(_separator("─"))
    print(f"  LLM model  : {settings.llm_model}")
    print(f"  YOLO model : {settings.yolo_model_name}")
    print(f"  API key    : {'set' if settings.openrouter_api_key else 'MISSING ✗'}")
    print(_separator("─"))

    if not settings.openrouter_api_key:
        print("ERROR: FIREEYE_OPENROUTER_API_KEY not set in .env  — cannot run LLM agents.")
        return 1

    requested = [s.strip().upper() for s in args.scenes.split(",")]
    results: list[dict] = []

    for letter in requested:
        if letter not in SCENE_MANIFEST:
            print(f"WARNING: Unknown scene '{letter}' — skipping.")
            continue

        filename, expected = SCENE_MANIFEST[letter]
        img_path = SYNTHETIC_DIR / filename

        if not img_path.exists():
            print(f"ERROR: {img_path} not found — run without --skip-generate first.")
            continue

        print(f"\nRunning scene {letter}: {filename} …")
        try:
            r = run_scene(letter, img_path, expected)
            results.append(r)
            print_scene_report(r)
        except Exception as exc:
            print(f"  ERROR running scene {letter}: {exc}")

    if results:
        print_comparison_table(results)

    return 0 if all(r["exact_match"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
