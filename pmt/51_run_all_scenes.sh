#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-out}"

python main.py \
  --base config/base.toml \
  --scene-dir config/scenes \
  --output-dir "$OUT_DIR"
