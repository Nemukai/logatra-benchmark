# Benchmark status — PAUSED (11 Jun 2026, by Priyansh)

All API runs stopped mid-M1 on cost review. Nothing bills until explicitly resumed.

## Spent so far (benchmark-attributable)
- OpenRouter (Gemini 3.5 Flash, thinking ON): $10.03 / 832 samples — root cause:
  hidden reasoning tokens (86% of cost). Fixed: reasoning now OFF by default.
- Whisper (OpenAI): ~$0.75 · Voxtral/gpt-audio probes: ~$0.01 · Sarvam: ~Rs 45

## Cached (paid once, reusable forever, in results/)
- gemini 832, whisper ~582, sarvam ~275 transcriptions (heldout 132 + partial M1)
- Held-out silver references: 96/132 accepted, 36 in review queue

## To resume
1. Decide key strategy: OpenRouter key is SHARED WITH KOE PROD ($14.49 credit left).
2. `set -a; source ~/Developer/Koe/.env.local; set +a`
3. `uv run python -m koe_benchmark run --manifest data/manifests/m1_public.jsonl \
    --providers sarvam,whisper,gemini,voxtral --workers 8`  (sarvam: use --workers 3)
4. Remaining cost to finish M1: ~$2 + ~Rs 105 (without gpt-audio; +$4.4 with it).
   Full 7,253-row public split: ~$34 + Rs 630 (with fixes; do NOT run gemini thinking-on).
