#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
DATASET=""
OUT=""
FORWARD_ARGS=()

source "$SCRIPT_DIR/_dataset_tools.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#--dataset=}"
      shift
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --out=*)
      OUT="${1#--out=}"
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

DATASET="$(resolve_scalar_qed_dataset "$SCRIPT_DIR" "$DATASET")"
require_scalar_qed_dataset "$DATASET"

if [[ -z "$OUT" ]]; then
  OUT="$SCRIPT_DIR/$(scalar_qed_dataset_stem "$DATASET")_pyqtgraph_3d.mp4"
fi

cd "$SCRIPT_DIR"
CMD=(
  "$PY" -m fields.scalar_qed.render_scalar_qed_pyqtgraph "$DATASET"
  --mode 3d
  --show-upper
  --window-width 1312
  --window-height 740
  --camera-elevation 24
  --camera-azimuth -42
  --camera-distance 31
  --upper-camera-elevation 5
  --upper-camera-azimuth 28
  --upper-camera-distance 29
  --height-scale 2.10
  --fps 24
  --out "$OUT"
)
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
  CMD+=("${FORWARD_ARGS[@]}")
fi
PYTHONPATH="../.." "${CMD[@]}"
