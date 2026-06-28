#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."/..
python fields/radiating_charge/render_radiating_charge_pyqtgraph.py \
  fields/radiating_charge/radiating_charge_dataset.npz \
  --mode 2d --show-center --lower-key bz_rad --transform signed_log \
  --out fields/radiating_charge/radiating_charge_pyqtgraph_2d.mp4
