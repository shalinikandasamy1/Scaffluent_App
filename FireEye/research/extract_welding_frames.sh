#!/bin/bash
# Extract frames from welding videos at 1 fps.
# Source: Images dataset/welding/welding *.mp4
# Output: FireEye/research/welding_frames/welding{N}_frame_{NNN}.jpg
#
# Usage: bash FireEye/research/extract_welding_frames.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT_DIR="$REPO_ROOT/Images dataset/welding"
OUTPUT_DIR="$REPO_ROOT/FireEye/research/welding_frames"

mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/welding\ *.mp4; do
    name=$(basename "$video" .mp4 | tr ' ' '')  # "welding 1" -> "welding1"
    echo "Extracting $name..."
    ffmpeg -i "$video" -vf fps=1 -q:v 2 "$OUTPUT_DIR/${name}_frame_%03d.jpg" -y -loglevel warning
done

echo "Done. $(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l) frames in $OUTPUT_DIR"
