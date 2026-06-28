#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname "$0")"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
else
  export PYTHONPATH="$(pwd)/.."
fi

python -m unittest \
  pmt.tests.test_pmt_numerics \
  pmt.tests.test_scene_pipeline \
  -v "$@"
