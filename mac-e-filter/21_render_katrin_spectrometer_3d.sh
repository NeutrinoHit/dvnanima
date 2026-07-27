#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"

PYTHONPATH="${project_dir}/src" \
python "${project_dir}/renderers/pyqtgraph_katrin_spectrometer_3d.py" \
  --dataset "${project_dir}/datasets/katrin_2013_collimation_ensemble.npz" \
  --out "${project_dir}/media/katrin_2013_electron_spectrometer_3d.mp4" \
  --fps 30 \
  --duration-s 12
