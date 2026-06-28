#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname "$0")"

python export_pmt_dataset.py \
  --config pmt.toml \
  --out pmt_dataset_preview.h5 \
  --preview \
  "$@"
