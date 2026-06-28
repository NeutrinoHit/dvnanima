#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SCENE="${1:-config/scenes/surface_to_plate.toml}"

python main.py \
  --base config/base.toml \
  --scene "$SCENE" \
  --dry-run
