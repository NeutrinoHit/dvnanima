#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname "$0")/.."

python smooth/flatten_hist2d.py \
  --input smooth/mu_positions_hist.npz \
  --output-dir smooth/out \
  --sigma-bins 2.0 \
  --eps-frac 0.05 \
  --min-count-for-roi 1
