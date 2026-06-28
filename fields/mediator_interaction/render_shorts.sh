#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$0")"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export DVN_PROFILE=shorts
MANIM_BIN="${MANIM_BIN:-manim}"
"$MANIM_BIN" -pqh scene_03_mediator_interaction.py MediatorInteractionScene
