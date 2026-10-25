#!/usr/bin/env bash
# B.S. Detector - double-click to run on macOS (or run ./start.command on Linux).
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 run.py "$@"
fi
echo "Python 3.9 or newer is required. Install it from https://www.python.org/downloads/"
read -r -p "Press Enter to close."
