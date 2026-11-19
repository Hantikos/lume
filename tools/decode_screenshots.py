"""Restore the PNG screenshots from their base64 copies.

The GitHub content API used by the publishing tool only accepts text, so the
screenshots are stored as ``docs/screenshots/*.png.b64``.  Run this script (or
let the CI workflow do it) to get the real PNG files back:

    python tools/decode_screenshots.py
"""

from __future__ import annotations

import base64
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(ROOT, "docs", "screenshots")


def main() -> int:
    if not os.path.isdir(FOLDER):
        print("no screenshots folder found")
        return 0
    count = 0
    for name in sorted(os.listdir(FOLDER)):
        if not name.endswith(".b64"):
            continue
        source = os.path.join(FOLDER, name)
        target = os.path.join(FOLDER, name[:-4])
        with open(source, "r", encoding="utf-8") as handle:
            data = base64.b64decode(handle.read())
        with open(target, "wb") as handle:
            handle.write(data)
        print(f"restored {os.path.relpath(target, ROOT)} ({len(data)} bytes)")
        count += 1
    if not count:
        print("nothing to restore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
