#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find "$ROOT" \
  -type d \
  -name partial_movie_files \
  -prune \
  -exec rm -rf {} +

find "$ROOT" \
  -type d \
  -path '*/media/videos/*/480p*' \
  -prune \
  -exec rm -rf {} +

printf 'Render cleanup complete.\n'
