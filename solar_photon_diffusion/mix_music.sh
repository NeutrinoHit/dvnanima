#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

audio_path="${1:-music/suno_solar_diffusion.mp3}"
if [[ ! -f "$audio_path" ]]; then
  echo "Music file not found: $audio_path" >&2
  echo "Download the SUNO track there, or pass its path as the first argument." >&2
  exit 2
fi

video_dir="media/videos/solar_diffusion/1920p30"
for language in ru en; do
  video_path="$video_dir/solar_photon_diffusion_${language}.mp4"
  output_path="$video_dir/solar_photon_diffusion_${language}_music.mp4"
  if [[ ! -f "$video_path" ]]; then
    echo "Rendered video not found: $video_path" >&2
    echo "Run: bash render.sh all" >&2
    exit 2
  fi

  duration="$({ ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$video_path"; })"
  fade_start="$(awk -v duration="$duration" \
    'BEGIN { value = duration - 4.0; if (value < 0) value = 0; printf "%.3f", value }')"

  ffmpeg -y \
    -i "$video_path" \
    -stream_loop -1 -i "$audio_path" \
    -filter_complex \
    "[1:a:0]atrim=duration=${duration},asetpts=PTS-STARTPTS,loudnorm=I=-18:TP=-2:LRA=9,afade=t=in:st=0:d=2.5,afade=t=out:st=${fade_start}:d=4,aresample=48000[music]" \
    -map 0:v:0 -map "[music]" \
    -c:v copy -c:a aac -b:a 192k -shortest \
    -metadata:s:a:0 language="$language" \
    "$output_path"
done

echo "Music versions created in $video_dir"
