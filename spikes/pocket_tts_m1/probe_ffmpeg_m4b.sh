#!/usr/bin/env bash
# Reproducible, model-free AAC M4B capability probe for WP1.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

FFMPEG="${FFMPEG:-ffmpeg}"
FFPROBE="${FFPROBE:-ffprobe}"
ARTIFACT_DIR="results/artifacts/ffmpeg-m4b"
RESULT_DIR="results/ffmpeg-m4b"
M4B="$ARTIFACT_DIR/wp1-generated-probe.m4b"
mkdir -p "$ARTIFACT_DIR" "$RESULT_DIR"
rm -f "$M4B"

cat >"$ARTIFACT_DIR/chapters.ffmeta" <<'EOF'
;FFMETADATA1
title=WP1 Generated Probe
author=Kenkui Test Author
artist=Kenkui Test Author
comment=Generated solid-color cover and sine tone; no source media.
[CHAPTER]
TIMEBASE=1/1000
START=0
END=1000
title=Generated Chapter One
[CHAPTER]
TIMEBASE=1/1000
START=1000
END=2000
title=Generated Chapter Two
EOF

"$FFMPEG" -version >"$RESULT_DIR/ffmpeg-version.txt" 2>&1
"$FFPROBE" -version >"$RESULT_DIR/ffprobe-version.txt" 2>&1

(
  printf '%s\n' \
    'Inputs: FFmpeg lavfi 440 Hz sine, 24000 Hz mono, 2 seconds; 64x64 solid color.' \
    'Cover copyright note: a generated unadorned solid-color square has no creative source material.' \
    'Output: AAC-LC mono at 48 kbit/s in an M4B/MP4 container with attached JPEG cover.'
  "$FFMPEG" -hide_banner -nostdin -y \
    -f lavfi -i 'color=c=0x305080:s=64x64:r=1:d=1' \
    -frames:v 1 -c:v mjpeg "$ARTIFACT_DIR/generated-cover.jpg"
  if "$FFMPEG" -hide_banner -nostdin -y \
    -f lavfi -i 'sine=frequency=440:sample_rate=24000:duration=2' \
    -i "$ARTIFACT_DIR/generated-cover.jpg" \
    -f ffmetadata -i "$ARTIFACT_DIR/chapters.ffmeta" \
    -map 0:a:0 -map 1:v:0 -map_metadata 2 -map_chapters 2 \
    -c:a aac -profile:a aac_low -b:a 48k -ac 1 \
    -c:v copy -disposition:v:0 attached_pic \
    -movflags +faststart "$M4B"; then
    status=0
  else
    status=$?
  fi
  printf 'encode_exit_status=%d\n' "$status"
  exit "$status"
) >"$RESULT_DIR/encode.txt" 2>&1

"$FFPROBE" -v error \
  -show_entries 'format=format_name,duration,size:format_tags=title,author,artist,comment:stream=index,codec_name,codec_long_name,profile,codec_type,sample_fmt,sample_rate,channels,channel_layout,width,height:stream_disposition=attached_pic:stream_tags=language,handler_name:chapter=id,time_base,start,end,start_time,end_time:chapter_tags=title' \
  -of json "$M4B" >"$RESULT_DIR/ffprobe.json" 2>"$RESULT_DIR/ffprobe.stderr.txt"

(
  printf '%s\n' 'Command semantics: decode every stream completely and discard decoded output.'
  if "$FFMPEG" -hide_banner -nostdin -v error -i "$M4B" -map 0 -f null -; then
    status=0
  else
    status=$?
  fi
  printf 'full_decode_exit_status=%d\n' "$status"
  exit "$status"
) >"$RESULT_DIR/full-decode.txt" 2>&1

{
  shasum -a 256 "$M4B"
  wc -c "$M4B"
} >"$RESULT_DIR/artifact-sha256-and-size.txt"

printf 'PASS: %s\n' "$M4B"
