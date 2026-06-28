#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$0")"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export DVN_PROFILE=widescreen
MANIM_BIN="${MANIM_BIN:-manim}"
"$MANIM_BIN" -pqh scene_01_single_packet.py SingleGaussianWavePacket
"$MANIM_BIN" -pqh scene_02_two_packets.py TwoFreeWavePackets
