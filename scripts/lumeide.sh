#!/usr/bin/env bash
# Start Lume Studio from a source checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 -m lumeide "$@"
