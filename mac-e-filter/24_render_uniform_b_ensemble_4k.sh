#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=src python renderers/pyqtgraph_uniform_b_ensemble_3d.py \
  --dataset datasets/uniform_b_ensemble.npz \
  --out media/uniform_b_ensemble_4k.mp4 \
  --window-width 3840 \
  --window-height 2160 \
  --camera-distance 15.5 \
  --camera-azimuth -66 \
  --camera-elevation 15 \
  --fps 30 \
  --duration-s 12 \
  --hold-start-s 0.8 \
  --hold-end-s 1.2

