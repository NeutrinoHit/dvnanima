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
  OUT="$SCRIPT_DIR/$(scalar_qed_dataset_stem "$DATASET")_pyqtgraph_2d.mp4"
fi

cd "$SCRIPT_DIR"
PYTHONPATH="../.." "$PY" -m fields.scalar_qed.render_scalar_qed_pyqtgraph "$DATASET" \
  --mode 2d \
  --show-upper \
  --show-centers \
  --window-width 1600 \
  --window-height 900 \
  --fps 24 \
  --out "$OUT" \
  "${FORWARD_ARGS[@]}"
