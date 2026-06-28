#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."/..
python fields/radiating_charge/render_radiating_charge_pyqtgraph.py \
  fields/radiating_charge/radiating_charge_dataset.npz \
  --mode 3d --show-center --show-fixed-center \
  --lower-key s_rad_mag --transform log --fixed-levels --z-mode fixed \
  --xy-scale 1.45 --surface-faceted \
  --level-max-quantile 0.9992 --height-peak-quantile 1.0 \
  --camera-distance-scale 0.82 --camera-elevation 34 --camera-azimuth -32 \
  --center-size 10 --fixed-center-size 26 \
  --height-scale 0.2 --surface-alpha 0.94 \
  --time-slowdown 1.2 --radiation-slowmo 2.0 --radiation-slowmo-gamma 2.0 --radiation-slowmo-quantile 0.997 \
  --window-width 1920 --window-height 1080 \
  --out fields/radiating_charge/radiating_charge_pyqtgraph_3d.mp4
