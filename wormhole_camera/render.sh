#!/usr/bin/env bash
# Both films (Russian and English), 1280x720, 30 fps; the sky cache is built on the first run (about 15 s, 600 MB in .cache/).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-3}"
[ -f ".cache/sky_w6144_L0.npy" ] || "$PYTHON_BIN" wormhole_sky.py --build
for lang in ru en; do
  "$PYTHON_BIN" wormhole_camera.py --lang "$lang" --workers "$WORKERS" "$@"
done
