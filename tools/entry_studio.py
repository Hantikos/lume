"""Entry point for the packaged Lume Studio application."""

import os
import sys


def _prepare_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    # when frozen by PyInstaller the package lives inside the bundle
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and meipass not in sys.path:
        sys.path.insert(0, meipass)


_prepare_path()

from lumeide.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
