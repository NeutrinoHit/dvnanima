#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

requested_language="${1:-ru}"
if [[ "$requested_language" != "ru" && "$requested_language" != "en" ]]; then
  echo "usage: $0 [ru|en]" >&2
  exit 2
fi

SOLAR_DIFFUSION_LANG="$requested_language" SOLAR_DIFFUSION_PREVIEW=1 \
  manim -ql --disable_caching solar_diffusion.py SolarPhotonDiffusion \
  -o "solar_photon_diffusion_${requested_language}_preview"
