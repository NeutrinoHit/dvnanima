#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname "$0")"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
else
  export PYTHONPATH="$(pwd)/.."
fi

python 40_plot_static_trajectories.py "$@"
