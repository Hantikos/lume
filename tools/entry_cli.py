"""Entry point for the packaged `lume` command line tool."""

import sys

from lume.cli import main

if __name__ == "__main__":
    sys.exit(main())
