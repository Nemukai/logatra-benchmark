#!/bin/zsh
set -uo pipefail
cd /Users/priyanshsingh/Developer/Nemukai/koe-benchmark
set -a; source ~/Developer/Nemukai/Koe/.env.local; set +a
mkdir -p logs

echo "=== FULL RUN START $(date) ==="

uv run python -m logatra_benchmark run \
  --manifest data/manifests/full_public.jsonl \
  --providers whisper --workers 8 >> logs/whisper.log 2>&1 &
WPID=$!

uv run python -m logatra_benchmark run \
  --manifest data/manifests/full_public.jsonl \
  --providers sarvam --workers 4 >> logs/sarvam.log 2>&1 &
SPID=$!

uv run python scripts/run_openrouter_budget.py \
  --budget-usd 5.0 --providers gemini,voxtral,gpt-audio --workers 8 \
  data/manifests/m1_public.jsonl data/manifests/heldout_yt.jsonl \
  >> logs/openrouter.log 2>&1 &
OPID=$!

wait $WPID && echo "STAGE whisper DONE" || echo "STAGE whisper FAILED"
wait $SPID && echo "STAGE sarvam DONE" || echo "STAGE sarvam FAILED"
wait $OPID && echo "STAGE openrouter DONE" || echo "STAGE openrouter FAILED"

uv run python scripts/score_report.py --out report/data.json \
  data/manifests/full_public.jsonl data/manifests/heldout_yt.silver.jsonl \
  && echo "STAGE scoring DONE" || echo "STAGE scoring FAILED"

echo "=== ALL DONE $(date) ==="
