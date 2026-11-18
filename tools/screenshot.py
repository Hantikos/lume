"""Take screenshots of Lume Studio (used for documentation and smoke tests).

    python tools/screenshot.py [url] [output-folder]
"""

from __future__ import annotations

import glob
import os
import sys
import time

from playwright.sync_api import sync_playwright


def _find_chromium() -> str | None:
    candidates = glob.glob("/root/.cache/ms-playwright/chromium-*/chrome-linux/chrome")
    candidates += glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"))
    candidates += glob.glob("/ms-playwright/chromium-*/chrome-linux/chrome")
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777/"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots"
    os.makedirs(out, exist_ok=True)

    with sync_playwright() as play:
        executable = _find_chromium()
        browser = play.chromium.launch(executable_path=executable,
                                       args=["--no-sandbox", "--force-color-profile=srgb"])
        page = browser.new_page(viewport={"width": 1600, "height": 950}, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1200)

        page.screenshot(path=os.path.join(out, "studio-dark.png"))
        print("saved", os.path.join(out, "studio-dark.png"))

        # run the sample program
        page.click("#btnRun")
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(out, "studio-run.png"))
        print("saved", os.path.join(out, "studio-run.png"))

        # light theme
        page.click("#btnTheme")
        page.wait_for_timeout(600)
        page.screenshot(path=os.path.join(out, "studio-light.png"))
        print("saved", os.path.join(out, "studio-light.png"))
        page.click("#btnTheme")
        page.wait_for_timeout(400)

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
