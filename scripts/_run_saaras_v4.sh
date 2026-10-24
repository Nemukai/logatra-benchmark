#!/bin/zsh
# Runs saaras:v4 over the public split and the held-out split, re-running
# until no sample errors remain (errors are never cached, so each pass only
# re-calls what failed, typically on 429 rate limits).
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source "${BENCHMARK_ENV_FILE:-.env}"; set +a
echo "=== saaras v4 START $(date) ==="
for pass in 1 2 3 4 5 6; do
  echo "--- pass $pass $(date)"
  out=$(./.venv/bin/python -m logatra_benchmark run --manifest data/local/full_public.jsonl \
        --providers sarvam-saaras-v4 --workers 2 --json 2>>logs/saaras-v4.err)
  echo "$out" | ./.venv/bin/python -c "import json,sys;d=json.load(sys.stdin);print('public rows',d['rows'],'errors',d['errors'])"
  errs=$(echo "$out" | ./.venv/bin/python -c "import json,sys;d=json.load(sys.stdin);print(sum(d['errors'].values()))")
  echo "results cached: $(ls results/sarvam-saaras-v4 | wc -l)"
  [ "$errs" = "0" ] && break
  sleep 30
done
for pass in 1 2 3; do
  out=$(./.venv/bin/python -m logatra_benchmark run --manifest data/manifests/heldout_yt.silver.jsonl \
        --providers sarvam-saaras-v4 --workers 2 --json 2>>logs/saaras-v4.err)
  echo "$out" | ./.venv/bin/python -c "import json,sys;d=json.load(sys.stdin);print('heldout rows',d['rows'],'errors',d['errors'])"
  errs=$(echo "$out" | ./.venv/bin/python -c "import json,sys;d=json.load(sys.stdin);print(sum(d['errors'].values()))")
  [ "$errs" = "0" ] && break
  sleep 30
done
echo "=== saaras v4 END $(date) === cached: $(ls results/sarvam-saaras-v4 | wc -l)"
