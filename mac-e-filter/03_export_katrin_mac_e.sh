#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=src python export_katrin_mac_e.py \
  --config configs/katrin_2013_mac_e.toml \
  --out datasets/katrin_2013_mac_e.npz

