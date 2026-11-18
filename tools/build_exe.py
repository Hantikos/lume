"""Build single-file executables for Lume and Lume Studio.

    python tools/build_exe.py            # builds LumeStudio (the IDE)
    python tools/build_exe.py --cli      # builds the lume command line tool
    python tools/build_exe.py --all      # builds both

Requires PyInstaller:   pip install pyinstaller
On Windows the result is a real .exe; on Linux/macOS a single-file binary.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, ".pyinstaller")

CLI_ENTRY = os.path.join(ROOT, "tools", "entry_cli.py")
STUDIO_ENTRY = os.path.join(ROOT, "tools", "entry_studio.py")


def _have_pyinstaller() -> bool:
    try:
        import PyInstaller  # noqa: F401
        return True
    except ImportError:
        return False


def build(target: str, name: str, entry: str, windowed: bool = False) -> int:
    if not _have_pyinstaller():
        print("PyInstaller is not installed.")
        print("Install it first:   pip install pyinstaller")
        return 1

    static_dir = os.path.join(ROOT, "lumeide", "static")
    examples_dir = os.path.join(ROOT, "examples")
    separator = ";" if os.name == "nt" else ":"

    command = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconfirm",
        "--clean",
        "--name", name,
        "--distpath", DIST,
        "--workpath", BUILD,
        "--specpath", BUILD,
        "--paths", ROOT,
        "--hidden-import", "lume",
        "--hidden-import", "lumeide",
        "--collect-submodules", "lume",
        "--collect-submodules", "lumeide",
        "--add-data", f"{static_dir}{separator}lumeide/static",
        "--add-data", f"{examples_dir}{separator}examples",
    ]
    if windowed and os.name == "nt":
        command.append("--windowed")
    command.append(entry)

    print("→", " ".join(command[:6]), "…")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    binary = os.path.join(DIST, name + (".exe" if os.name == "nt" else ""))
    if os.path.exists(binary):
        size = os.path.getsize(binary) / (1024 * 1024)
        print(f"✔ {binary}  ({size:.1f} MB)")
    else:
        print(f"⚠ the binary was not found at {binary}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Lume executables")
    parser.add_argument("--cli", action="store_true", help="build the lume command line tool")
    parser.add_argument("--studio", action="store_true", help="build Lume Studio (default)")
    parser.add_argument("--all", action="store_true", help="build both")
    parser.add_argument("--clean", action="store_true", help="remove the dist folder first")
    args = parser.parse_args()

    if args.clean and os.path.isdir(DIST):
        shutil.rmtree(DIST)

    targets = []
    if args.cli or args.all:
        targets.append(("cli", "lume", CLI_ENTRY, False))
    if args.studio or args.all or not targets:
        targets.append(("studio", "LumeStudio", STUDIO_ENTRY, False))

    for _key, name, entry, windowed in targets:
        code = build(name, name, entry, windowed)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
