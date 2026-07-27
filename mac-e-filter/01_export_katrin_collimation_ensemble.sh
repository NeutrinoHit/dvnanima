#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"

PYTHONPATH="${project_dir}/src" \
python "${project_dir}/export_katrin_collimation_ensemble.py" \
  --config "${project_dir}/configs/katrin_2013_collimation_ensemble.toml" \
  --out "${project_dir}/datasets/katrin_2013_collimation_ensemble.npz"

