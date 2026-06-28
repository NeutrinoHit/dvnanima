#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname "$0")"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
else
  export PYTHONPATH="$(pwd)/.."
fi

export PMT_DATA_PATH="${PMT_DATA_PATH:-pmt_dataset_preview.h5}"
export PMT_MAX_PARTICLES="${PMT_MAX_PARTICLES:-220}"
export PMT_RENDER_SECONDS="${PMT_RENDER_SECONDS:-24}"

manim -pqh pmt_manim.py PMTSchematicFromHDF5 "$@"
