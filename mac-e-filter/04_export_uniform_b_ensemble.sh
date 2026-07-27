#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=src python export_uniform_b_ensemble.py \
  --config configs/uniform_b_ensemble.toml \
  --out datasets/uniform_b_ensemble.npz

