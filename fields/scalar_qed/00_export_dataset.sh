#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
DEFAULT_CONFIG="$SCRIPT_DIR/scalar_qed.toml"
DATASET_DIR="$SCRIPT_DIR/datasets"

CONFIG="$DEFAULT_CONFIG"
OUT=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --config=*)
      CONFIG="${1#--config=}"
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --out)
      OUT="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --out=*)
      OUT="${1#--out=}"
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OUT" ]]; then
  CONFIG_STEM="$(basename "$CONFIG" .toml)"
  SCENARIO="${CONFIG_STEM#scalar_qed_}"
  if [[ "$SCENARIO" == "scalar_qed" ]]; then
    SCENARIO="$("$PY" - "$CONFIG" <<'PY'
from __future__ import annotations
import sys
import tomllib
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
packets = cfg.get("packets", [])
signs = []
for packet in packets:
    charge = float(packet.get("charge_ratio", 0.0))
    signs.append("p" if charge >= 0.0 else "m")
if not signs:
    signs = ["none"]
print("".join(signs))
PY
)"
  fi
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUT="$DATASET_DIR/scalar_qed_dataset_${SCENARIO}_${TIMESTAMP}.npz"
  FORWARD_ARGS+=(--out "$OUT")
fi

mkdir -p "$DATASET_DIR"
cd "$SCRIPT_DIR"
PYTHONPATH="../.." "$PY" -m fields.scalar_qed.export_scalar_qed_dataset "${FORWARD_ARGS[@]}"

OUT_ABS="$("$PY" - "$OUT" <<'PY'
from __future__ import annotations
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
OUT_REL="$("$PY" - "$SCRIPT_DIR" "$OUT_ABS" <<'PY'
from __future__ import annotations
import os
import sys
from pathlib import Path
script_dir = Path(sys.argv[1]).resolve()
out_abs = Path(sys.argv[2]).resolve()
print(os.path.relpath(out_abs, script_dir))
PY
)"
LATEST_LINK="$DATASET_DIR/scalar_qed_dataset_latest.npz"
OUT_REL_TO_DATASETS="$("$PY" - "$DATASET_DIR" "$OUT_ABS" <<'PY'
from __future__ import annotations
import os
import sys
from pathlib import Path
dataset_dir = Path(sys.argv[1]).resolve()
out_abs = Path(sys.argv[2]).resolve()
print(os.path.relpath(out_abs, dataset_dir))
PY
)"
ln -sfn "$OUT_REL_TO_DATASETS" "$LATEST_LINK"
echo "Saved dataset: $OUT_REL"
echo "Updated latest dataset link: $LATEST_LINK -> $OUT_REL_TO_DATASETS"
