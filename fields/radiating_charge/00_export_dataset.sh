#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."/..
python fields/radiating_charge/export_radiating_charge_dataset.py \
  --config fields/radiating_charge/radiating_charge.toml \
  --out fields/radiating_charge/radiating_charge_dataset.npz
