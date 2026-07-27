#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=src python renderers/pyqtgraph_katrin_mac_e_3d.py \
  --dataset datasets/katrin_2013_mac_e.npz \
  --out media/katrin_mac_e_electric_4k.mp4 \
  --window-width 3840 \
  --window-height 2160 \
  --camera-distance 32 \
  --camera-azimuth -48 \
  --camera-elevation 10 \
  --fps 30 \
  --duration-s 12 \
  --hold-start-s 0.8 \
  --hold-end-s 1.2

