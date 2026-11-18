"""Upload the project to GitHub through the connected GitHub App.

The App writes one file per call, so this driver walks the project folder and
pushes every text file.  Binary screenshots are stored base64-encoded and are
restored by ``.github/workflows/screenshots.yml``.

    python tools/publish.py [--owner Hantikos] [--repo lume] [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", "dist", ".pyinstaller", "build", ".venv", "node_modules"}
SKIP_FILES = {"ide.log"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".lume", ".json", ".yml", ".yaml", ".toml", ".sh",
                 ".bat", ".css", ".js", ".html", ".svg", ".cfg", ".ini", ".b64", ".pyc"}


def collect_files() -> list[tuple[str, str]]:
    """Return (relative path, local path) for every file that must be uploaded."""
    items: list[tuple[str, str]] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]
        for name in sorted(files):
            if name in SKIP_FILES or name.endswith(".pyc"):
                continue
            suffix = os.path.splitext(name)[1].lower()
            if suffix not in TEXT_SUFFIXES and not name.startswith("."):
                continue
            if name.endswith(".png.b64"):
                continue  # handled separately, below
            rel = os.path.relpath(os.path.join(base, name), ROOT).replace(os.sep, "/")
            items.append((rel, os.path.join(base, name)))
    return items


def encode_binary(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def call_tool(args: dict) -> tuple[bool, str]:
    payload = os.path.join(ROOT, ".publish-args.json")
    with open(payload, "w", encoding="utf-8") as handle:
        json.dump(args, handle, ensure_ascii=False)
    try:
        result = subprocess.run(
            ["mcode-tools", "connector", "call", "connector__github__create_or_update_file",
             "--args-file", payload],
            capture_output=True, text=True, timeout=180,
        )
    finally:
        if os.path.exists(payload):
            os.remove(payload)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="Hantikos")
    parser.add_argument("--repo", default="lume")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = collect_files()
    # binary screenshots are uploaded as base64 text and restored by CI
    extras: list[tuple[str, str]] = []
    screenshots = os.path.join(ROOT, "docs", "screenshots")
    if os.path.isdir(screenshots):
        for name in sorted(os.listdir(screenshots)):
            if name.endswith(".png"):
                target = os.path.join(screenshots, name + ".b64")
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(encode_binary(os.path.join(screenshots, name)))
                extras.append((f"docs/screenshots/{name}.b64", target))

    total = len(files) + len(extras)
    print(f"uploading {total} files to {args.owner}/{args.repo}")
    failed = []
    for index, (rel, local) in enumerate(files + extras, start=1):
        with open(local, "r", encoding="utf-8") as handle:
            content = handle.read()
        if args.dry_run:
            print(f"[{index}/{total}] {rel} ({len(content)} chars)")
            continue
        ok, output = call_tool({
            "owner": args.owner,
            "repo": args.repo,
            "branch": args.branch,
            "path": rel,
            "content": content,
            "message": f"Add {rel}",
        })
        status = "ok" if ok else "FAILED"
        print(f"[{index}/{total}] {status} {rel}")
        if not ok:
            failed.append((rel, output[:300]))
            time.sleep(1)
    if failed:
        print("\nfailures:")
        for rel, output in failed:
            print(" -", rel, "->", output)
        return 1
    print("\nall files uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
