#!/bin/sh
# Piper TTS as a separate local HTTP server. piper-tts is GPL-3.0, so it is run, not imported.
# Voices (download once into voices/, see README): en_US-joe-medium (agent), en_US-libritts_r-medium (eval callers).
cd "$(dirname "$0")/.."
exec uvx --python 3.12 --from 'piper-tts[http]==1.8.0' python -m piper.http_server \
  -m voices/en_US-joe-medium.onnx --data-dir voices --host 127.0.0.1 --port 5100
