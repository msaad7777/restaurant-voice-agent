#!/bin/sh
# Download the two Piper voices into voices/ (about 140 MB). Source: huggingface.co/rhasspy/piper-voices
#   en_US-joe-medium: the agent's voice
#   en_US-libritts_r-medium: the eval's synthetic callers (fine-tuned from lessac; see "Data, models and licences")
set -e
cd "$(dirname "$0")/.."
mkdir -p voices
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US
for v in joe/medium/en_US-joe-medium libritts_r/medium/en_US-libritts_r-medium; do
  for ext in onnx onnx.json; do
    f="voices/$(basename "$v").$ext"
    [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$BASE/$v.$ext"
  done
done
ls -l voices
