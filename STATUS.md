# Benchmark status (30 Aug 2026)

Scoring is complete and reproducible offline from the committed results
cache. Nothing bills unless a fresh `run` is started explicitly.

## Done
- Full public split (7,253 samples): sarvam (saarika:v2.5), sarvam-saaras
  (saaras:v3), whisper (whisper-1).
- M1 public subset (800): gemini, voxtral. Partial (336): gpt-audio.
- Held-out split: all 132 rows referenced (96 silver consensus + 36
  adjudicated; see data/manifests/heldout_yt.adjudications.json).
- Report: report/index.html regenerates offline from cache.

## Remaining paid work (estimates at current list prices)
- Complete gemini + voxtral over the full public split: ~$2 + ~$2.
- Complete gpt-audio over the full public split: ~$20 (or drop it from the
  full-split table and keep it intersection-only).
- Unrun adapters over the full split (~17.5 audio-hours): Deepgram ~$5,
  ElevenLabs Scribe ~$7, AssemblyAI ~$3, Google ~$16. Verify current
  pricing before running; use `--workers` conservatively on new providers.

## Next (unpaid)
- Publish: public GitHub repo + Hugging Face dataset (audio + results
  cache) + Space hosting report/.
- Human spot-listen of a sample of held-out references (silver to gold).
