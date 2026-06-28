#!/usr/bin/env bash

resolve_scalar_qed_dataset() {
  local script_dir="$1"
  local requested="${2:-}"

  if [[ -n "$requested" ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  if [[ -n "${SCALAR_QED_DATASET:-}" ]]; then
    printf '%s\n' "$SCALAR_QED_DATASET"
    return 0
  fi

  local latest="$script_dir/scalar_qed_dataset_latest.npz"
  if [[ -f "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi

  local candidates=()
  while IFS= read -r path; do
    candidates+=("$path")
  done < <(find "$script_dir" -maxdepth 1 -type f -name 'scalar_qed_dataset_*.npz' | sort)

  if [[ ${#candidates[@]} -gt 0 ]]; then
    printf '%s\n' "${candidates[$((${#candidates[@]} - 1))]}"
    return 0
  fi

  printf '%s\n' "$latest"
}

require_scalar_qed_dataset() {
  local dataset="$1"

  if [[ -f "$dataset" ]]; then
    return 0
  fi

  cat >&2 <<EOF
Missing dataset: $dataset

Create one first, for example:
  ./00_export_dataset.sh --config scalar_qed_pp.toml
  ./00_export_dataset.sh --config scalar_qed_pm.toml
  ./00_export_dataset.sh --config scalar_qed_single.toml

Or pass an explicit dataset:
  --dataset fields/scalar_qed/scalar_qed_dataset_pp_YYYYMMDD_HHMMSS.npz
EOF
  return 1
}

scalar_qed_dataset_stem() {
  local dataset="$1"
  basename "$dataset" .npz
}
