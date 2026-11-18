"""Entry point for the packaged `lume` command line tool."""

import os
import sys

from lume.cli import main


def _run() -> int:
    try:
        return main()
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    code = _run()
    # a closed pipe (e.g. `lume run file.lume | head`) must not break the shutdown
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(code)
