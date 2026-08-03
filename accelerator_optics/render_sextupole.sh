#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export DVN_PROFILE="${DVN_PROFILE:-shorts}"
export DVN_FRAME_RATE="${DVN_FRAME_RATE:-60}"

manim -qh sextupole_optics.py SextupoleMagnetScene
