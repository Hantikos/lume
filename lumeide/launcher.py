"""Launches the Lume Studio window.

The studio is a local web application; on a normal desktop we open it in
"app mode" of a Chromium based browser (Edge, Chrome or Brave), which gives a
clean window without browser toolbars.  If no such browser is found we fall
back to the default browser.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser

WINDOWS_EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
LINUX_BROWSERS = [
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "brave-browser",
]
MAC_APPS = [
    "/Applications/Microsoft Edge.app",
    "/Applications/Google Chrome.app",
    "/Applications/Brave Browser.app",
    "/Applications/Chromium.app",
]


def _find_windows_browser():
    for path in WINDOWS_EDGE:
        if os.path.exists(path):
            return path
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _find_linux_browser():
    for name in LINUX_BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def _find_mac_browser():
    for path in MAC_APPS:
        if os.path.exists(path):
            return path
    return None


def open_window(url: str) -> str:
    """Open the studio; returns a short description of what happened."""
    system = sys.platform
    try:
        if system.startswith("win"):
            browser = _find_windows_browser()
            if browser:
                subprocess.Popen([browser, f"--app={url}", "--window-size=1440,900"])
                return f"opened in {os.path.basename(browser)}"
        elif system == "darwin":
            browser = _find_mac_browser()
            if browser:
                subprocess.Popen(["open", "-a", browser, "--args", f"--app={url}",
                                  "--window-size=1440,900"])
                return f"opened in {os.path.basename(browser)}"
        else:
            browser = _find_linux_browser()
            if browser and os.environ.get("DISPLAY"):
                subprocess.Popen([browser, f"--app={url}", "--window-size=1440,900"])
                return f"opened in {os.path.basename(browser)}"
    except Exception:
        pass
    try:
        webbrowser.open(url)
        return "opened in the default browser"
    except Exception:
        return "open this address manually"


def print_banner(url: str, workspace: str) -> None:
    print("\n  Lume Studio is running")
    print(f"  → {url}")
    print(f"  workspace: {os.path.abspath(workspace)}")
    print("  press Ctrl+C to stop\n")
