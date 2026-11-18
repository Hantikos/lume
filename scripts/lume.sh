#!/usr/bin/env bash
# Run the Lume command line tool from a source checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/lume/__main__.py" "$@"
