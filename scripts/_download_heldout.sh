#!/bin/zsh
# Download spread audio sections for one video id as 16k mono WAV.
# Usage: _download_heldout.sh <video_id> <duration_seconds>
# Videos shorter than 600s are downloaded whole to full.wav.
set -e
cd /Users/priyanshsingh/Developer/Nemukai/koe-benchmark
ID=$1
DUR=$2
OUT="data/audio/heldout_yt/$ID"
mkdir -p "$OUT"
export PATH="/opt/homebrew/bin:$PATH"

if [ "$DUR" -lt 600 ]; then
  [ -f "$OUT/full.wav" ] && exit 0
  uvx yt-dlp -q --no-warnings -f "bestaudio/best" \
    -x --audio-format wav --postprocessor-args "ffmpegextractaudio:-ar 16000 -ac 1" \
    -o "$OUT/full.%(ext)s" "https://www.youtube.com/watch?v=$ID"
  exit 0
fi

S1=$((DUR * 15 / 100)); S2=$((DUR * 45 / 100)); S3=$((DUR * 75 / 100))
LEN=110
[ -f "$OUT/full.wav" ] && exit 0
uvx yt-dlp -q --no-warnings -f "bestaudio/best" \
  --download-sections "*${S1}-$((S1 + LEN))" \
  --download-sections "*${S2}-$((S2 + LEN))" \
  --download-sections "*${S3}-$((S3 + LEN))" \
  -x --audio-format wav --postprocessor-args "ffmpegextractaudio:-ar 16000 -ac 1" \
  -o "$OUT/sec%(section_start)06d.%(ext)s" "https://www.youtube.com/watch?v=$ID"
# Concatenate sections (in time order) into full.wav for reference.
ls "$OUT"/sec*.wav | sort | sed "s/^/file '/; s/$/'/" > "$OUT/concat.txt"
/opt/homebrew/bin/ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 \
  -i "$OUT/concat.txt" -c copy "$OUT/full.wav"
rm "$OUT/concat.txt"
