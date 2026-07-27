#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"

PYTHONPATH="${project_dir}/src" \
python "${project_dir}/renderers/pyqtgraph_katrin_trajectory.py" \
  --dataset "${project_dir}/datasets/katrin_2013_electron_18p6kev.npz" \
  --out "${project_dir}/media/katrin_2013_electron_trajectory_pyqtgraph.mp4" \
  --fps 30 \
  --duration-s 12

