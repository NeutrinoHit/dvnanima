#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MANIM_BIN="${MANIM_BIN:-manim}"
"$MANIM_BIN" -pqh nucleus_shapes.py NucleusShapeMorphs
