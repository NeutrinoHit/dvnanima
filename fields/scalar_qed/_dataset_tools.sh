#!/usr/bin/env bash

scalar_qed_dataset_dir() {
  local script_dir="$1"
  printf '%s\n' "$script_dir/datasets"
}

resolve_scalar_qed_dataset() {
  local script_dir="$1"
  local requested="${2:-}"
  local dataset_dir
  dataset_dir="$(scalar_qed_dataset_dir "$script_dir")"

  if [[ -n "$requested" ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  if [[ -n "${SCALAR_QED_DATASET:-}" ]]; then
    printf '%s\n' "$SCALAR_QED_DATASET"
    return 0
  fi

  local latest="$dataset_dir/scalar_qed_dataset_latest.npz"
  if [[ -f "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi

  local candidates=()
  while IFS= read -r path; do
    candidates+=("$path")
  done < <(find "$dataset_dir" -maxdepth 1 -type f -name 'scalar_qed_dataset_*.npz' 2>/dev/null | sort)

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
  --dataset fields/scalar_qed/datasets/scalar_qed_dataset_pp_YYYYMMDD_HHMMSS.npz
EOF
  return 1
}

scalar_qed_dataset_stem() {
  local dataset="$1"
  basename "$dataset" .npz
}

scalar_qed_dataset_label() {
  local dataset="$1"
  local stem
  stem="$(scalar_qed_dataset_stem "$dataset")"
  stem="${stem#scalar_qed_dataset_}"
  if [[ "$stem" =~ ^([[:alnum:]_-]+)_[0-9]{8}_[0-9]{6}$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' "$stem"
  fi
}
