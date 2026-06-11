#!/bin/zsh
set -uo pipefail
cd /Users/priyanshsingh/Developer/koe-benchmark
set -a; source ~/Developer/Koe/.env.local; set +a
export KOE_GEMINI_REASONING=minimal

spent() { uv run python -c "
import json, glob
print(sum(json.load(open(f))['raw'].get('usage',{}).get('cost',0)
    for p in ('voxtral','gemini','gpt-audio') for f in glob.glob(f'results/{p}/*.json')))"; }

uv run python scripts/run_openrouter_budget.py --budget-usd 0.55 --providers voxtral --workers 8 \
  data/manifests/m1_public.jsonl data/manifests/heldout_yt.jsonl
echo "STAGE voxtral DONE"

uv run python scripts/run_openrouter_budget.py --budget-usd 1.95 --providers gemini --workers 8 \
  data/manifests/m1_public.jsonl data/manifests/heldout_yt.jsonl
echo "STAGE gemini DONE"

S=$(spent)
REMAIN=$(python3 -c "print(max(0, round(5.0 - 0.75 - $S, 3)))")  # 0.75 = pre-pause voxtral spend already counted in caches... recompute properly below
REMAIN=$(python3 -c "print(max(0, round(5.0 - $S, 3)))")
echo "gpt-audio budget: \$$REMAIN (total OR cache spend so far: \$$S)"
uv run python scripts/run_openrouter_budget.py --budget-usd $REMAIN --providers gpt-audio --workers 8 \
  data/manifests/heldout_yt.jsonl data/manifests/m1_public_interleaved.jsonl
echo "STAGE gpt-audio DONE"
echo "OPENROUTER ALL DONE total spend: \$$(spent)"
