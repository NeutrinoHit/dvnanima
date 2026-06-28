#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname "$0")"

./01_export_preview_dataset.sh
./10_render_manim_preview.sh
