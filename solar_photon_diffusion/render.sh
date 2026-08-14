#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

requested_language="${1:-all}"
if [[ "$requested_language" != "all" && "$requested_language" != "ru" && "$requested_language" != "en" ]]; then
  echo "usage: $0 [ru|en|all]" >&2
  exit 2
fi

languages=(ru en)
if [[ "$requested_language" != "all" ]]; then
  languages=("$requested_language")
fi

for language in "${languages[@]}"; do
  SOLAR_DIFFUSION_LANG="$language" manim -qh --disable_caching \
    solar_diffusion.py SolarPhotonDiffusion \
    -o "solar_photon_diffusion_${language}"
done
