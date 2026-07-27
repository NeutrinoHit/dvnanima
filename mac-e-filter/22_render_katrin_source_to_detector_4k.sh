#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"

PYTHONPATH="${project_dir}/src" \
python "${project_dir}/renderers/pyqtgraph_katrin_source_to_detector_3d.py" \
  --dataset "${project_dir}/datasets/katrin_2013_source_to_detector.npz" \
  --out "${project_dir}/media/katrin_source_to_detector_4k.mp4" \
  --window-width 3840 \
  --window-height 2160 \
  --camera-distance 32 \
  --camera-azimuth -48 \
  --camera-elevation 10 \
  --fps 30 \
  --duration-s 12
