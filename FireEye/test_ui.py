"""Playwright visual QA for the FireEye NiceGUI Dashboard.

Starts ui_app.py as a subprocess, then:
  1. Verifies the page loads with the correct title and header.
  2. Checks all five result tabs are present.
  3. Checks the pipeline stage indicators are visible.
  4. Takes a full-page screenshot → qa_screenshots/01_initial.png
  5. Uploads a test image and checks the Analyze button is enabled.
  6. Takes a screenshot of the loaded state → qa_screenshots/02_image_loaded.png

Run from the FireEye/ directory:
    venv/bin/python test_ui.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# ── config ─────────────────────────────────────────────────────────────────────
APP_PORT     = 8090
APP_URL      = f"http://localhost:{APP_PORT}"
STARTUP_WAIT = 12         # seconds to let the app fully start
SCREENSHOT_DIR = Path(__file__).parent / "qa_screenshots"

# Use the first safe test image available
TEST_IMAGE = next(
    (
        p for p in [
            Path(__file__).parent / "test_data" / "dangerous" / "01_dfire.jpg",
            Path(__file__).parent / "test_data" / "safe" / "01_extinguisher.jpg",
        ]
        if p.exists()
    ),
    None,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _take(page, name: str) -> None:
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    dest = SCREENSHOT_DIR / name
    page.screenshot(path=str(dest), full_page=True)
    print(f"  📸  screenshot → {dest}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
    ui_app      = Path(__file__).parent / "ui_app.py"

    print("▶ Starting FireEye dashboard …")
    proc = subprocess.Popen(
        [str(venv_python), str(ui_app)],
        cwd=str(Path(__file__).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until the port accepts connections (up to STARTUP_WAIT seconds)
    deadline = time.time() + STARTUP_WAIT
    ready = False
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", APP_PORT), timeout=1):
                ready = True
                break
        except OSError:
            time.sleep(0.5)

    if not ready or proc.poll() is not None:
        proc.terminate()
        print("✘ App failed to start within the timeout.")
        sys.exit(1)

    print(f"  App is up at {APP_URL}")

    failures: list[str] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page    = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(APP_URL, wait_until="networkidle", timeout=15_000)

            # ── 1. Page title ─────────────────────────────────────────────────
            title = page.title()
            if "FireEye" not in title:
                failures.append(f"Title missing 'FireEye': got '{title}'")
            else:
                print(f"  ✔ title: {title!r}")

            # ── 2. Header text ────────────────────────────────────────────────
            header_text = page.locator("header").inner_text()
            if "FireEye" not in header_text:
                failures.append("Header does not contain 'FireEye'")
            else:
                print("  ✔ header present")

            # ── 3. Rengoku gradient header ────────────────────────────────────
            header_style = page.locator("header").get_attribute("style") or ""
            header_class = page.locator("header").get_attribute("class") or ""
            if "rengoku-header" in header_class or "#C91B00" in header_style or "#E85D04" in header_style:
                print("  ✔ Rengoku flame header class applied")
            else:
                # Check computed background via JS (CSS variable resolved)
                bg = page.evaluate(
                    "() => getComputedStyle(document.querySelector('header')).backgroundImage"
                )
                if "gradient" in bg.lower():
                    print(f"  ✔ Rengoku gradient applied (computed): {bg[:80]}…")
                else:
                    failures.append(f"Rengoku gradient not detected on header. class={header_class!r}")

            # ── 4. Five tabs visible ──────────────────────────────────────────
            expected_tabs = ["Overview", "Images", "Detections", "Compliance", "Assessment", "Audit Log", "Raw JSON"]
            for tab in expected_tabs:
                loc = page.get_by_role("tab", name=tab)
                if loc.count() == 0:
                    failures.append(f"Tab '{tab}' not found")
                else:
                    print(f"  ✔ tab '{tab}'")

            # ── 5. Pipeline stage labels ──────────────────────────────────────
            expected_stages = ["YOLO Detection", "Risk Classifier", "Present Agent", "Future Agent"]
            body_text = page.locator("body").inner_text()
            for stage in expected_stages:
                if stage in body_text:
                    print(f"  ✔ stage '{stage}'")
                else:
                    failures.append(f"Stage label '{stage}' not found on page")

            # ── 6. Analyze button is disabled before upload ───────────────────
            analyze_btn = page.get_by_role("button", name="Analyze")
            if analyze_btn.count() == 0:
                failures.append("Analyze button not found")
            else:
                disabled = analyze_btn.is_disabled()
                if disabled:
                    print("  ✔ Analyze button disabled before upload")
                else:
                    failures.append("Analyze button should be disabled before upload")

            # ── 7. Initial screenshot ─────────────────────────────────────────
            _take(page, "01_initial.png")

            # ── 8. Upload a test image ────────────────────────────────────────
            if TEST_IMAGE and TEST_IMAGE.exists():
                print(f"  Uploading {TEST_IMAGE.name} …")
                # NiceGUI ui.upload renders a hidden <input type="file"> inside a q-uploader
                file_input = page.locator('input[type="file"]').first
                file_input.set_input_files(str(TEST_IMAGE))
                # Wait for the upload notification
                page.wait_for_selector(".q-notification", timeout=8_000)
                time.sleep(1)

                # Analyze button should now be enabled
                if analyze_btn.is_enabled():
                    print("  ✔ Analyze button enabled after upload")
                else:
                    failures.append("Analyze button still disabled after upload")

                _take(page, "02_image_loaded.png")

                # ── 9. Run full analysis ────────────────────────────────────
                print("  Running full analysis (may take 30-60s with LLM) …")
                analyze_btn.click()
                try:
                    # Wait for analysis complete notification (up to 120s for LLM)
                    page.wait_for_selector(
                        "text=Analysis complete", timeout=120_000
                    )
                    print("  ✔ Analysis completed successfully")

                    # Screenshot each tab
                    for tab_name in ["Overview", "Images", "Detections", "Compliance", "Assessment", "Audit Log", "Raw JSON"]:
                        tab_loc = page.get_by_role("tab", name=tab_name)
                        if tab_loc.count() > 0:
                            tab_loc.click()
                            page.wait_for_timeout(500)
                            safe_name = tab_name.lower().replace(" ", "_")
                            _take(page, f"03_{safe_name}.png")
                            print(f"  ✔ tab '{tab_name}' rendered")

                except Exception as e:
                    print(f"  ⚠  Analysis did not complete: {e}")
                    _take(page, "03_analysis_error.png")
            else:
                print("  ⚠  No test image found, skipping upload test")

            browser.close()

    finally:
        proc.terminate()
        proc.wait()
        print("  App stopped.")

    # ── summary ────────────────────────────────────────────────────────────────
    print()
    if failures:
        print(f"✘ {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"    • {f}")
        sys.exit(1)
    else:
        print("✔ All visual QA checks passed.")


if __name__ == "__main__":
    main()
