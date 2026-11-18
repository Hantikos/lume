"""Take screenshots of Lume Studio (used for documentation and smoke tests).

    python tools/screenshot.py [url] [output-folder]
"""

from __future__ import annotations

import glob
import os
import sys

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
        browser = play.chromium.launch(executable_path=_find_chromium(),
                                       args=["--no-sandbox", "--force-color-profile=srgb"])
        page = browser.new_page(viewport={"width": 1600, "height": 950}, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1200)

        # open the XOR example so the editor shows real model code
        page.click('.ptab[data-panel="examples"]')
        page.wait_for_timeout(400)
        page.click('.examples li:nth-child(5) button')
        page.wait_for_timeout(900)
        page.screenshot(path=os.path.join(out, "studio-dark.png"))
        print("saved studio-dark.png")

        # run it: the console shows the training log
        page.click("#btnRun")
        page.wait_for_timeout(2500)
        page.screenshot(path=os.path.join(out, "studio-run.png"))
        print("saved studio-run.png")

        # light theme
        page.click("#btnTheme")
        page.wait_for_timeout(700)
        page.screenshot(path=os.path.join(out, "studio-light.png"))
        print("saved studio-light.png")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
