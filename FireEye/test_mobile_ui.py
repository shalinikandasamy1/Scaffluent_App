"""Playwright mobile viewport screenshots for FireEye NiceGUI Dashboard.

Starts ui_app.py as a subprocess, then takes screenshots at three mobile
viewport sizes to verify the responsive layout.

Run from the FireEye/ directory:
    venv/bin/python test_mobile_ui.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

APP_PORT = 8090
APP_URL = f"http://localhost:{APP_PORT}"
STARTUP_WAIT = 15
SCREENSHOT_DIR = Path(__file__).parent / "qa_screenshots" / "mobile"

VIEWPORTS = [
    ("iphone_se",  375, 812),
    ("iphone_14",  390, 844),
    ("ipad",       768, 1024),
]


def _take(page, name: str) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    dest = SCREENSHOT_DIR / name
    page.screenshot(path=str(dest), full_page=True)
    print(f"  screenshot -> {dest}")


def main() -> None:
    venv_python = Path(__file__).parent / "venv" / "bin" / "python"
    ui_app = Path(__file__).parent / "ui_app.py"

    print("Starting FireEye dashboard ...")
    proc = subprocess.Popen(
        [str(venv_python), str(ui_app)],
        cwd=str(Path(__file__).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

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
        print("App failed to start within the timeout.")
        sys.exit(1)

    print(f"  App is up at {APP_URL}")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)

            for name, w, h in VIEWPORTS:
                print(f"\n--- {name} ({w}x{h}) ---")
                page = browser.new_page(viewport={"width": w, "height": h})
                page.goto(APP_URL, wait_until="networkidle", timeout=15_000)
                time.sleep(1)

                _take(page, f"{name}_initial.png")

                # Scroll down to verify the full layout
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.5)
                _take(page, f"{name}_scrolled.png")

                page.close()

            browser.close()

    finally:
        proc.terminate()
        proc.wait()
        print("\n  App stopped.")

    print("\nDone. Check screenshots in:", SCREENSHOT_DIR)


if __name__ == "__main__":
    main()
