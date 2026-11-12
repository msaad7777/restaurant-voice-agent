#!/bin/sh
# Full eval matrix. Needs Ollama (gpt-oss:20b, qwen3:8b, llama3.2:3b) and, for audio, Piper on :5100.
set -e
cd "$(dirname "$0")/.."
ev() { uv run python eval/run_eval.py "$@" > /dev/null 2>&1 && echo "done: $*"; }
for set in scenarios heldout; do ev text --agent rules --set $set; done
for m in gpt-oss:20b qwen3:8b llama3.2:3b; do
  for set in scenarios heldout; do ev text --agent $m --set $set; done
  [ "$m" = "gpt-oss:20b" ] && ev text --agent $m --no-guards
  [ "$m" = "gpt-oss:20b" ] || ollama stop $m || true
done
for stt in base.en small.en; do ev audio --agent gpt-oss:20b --stt $stt; done
ev audio --agent gpt-oss:20b --stt base.en --set heldout
ev audio --agent rules --stt base.en
