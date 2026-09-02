# Benchmark status (3 Sep 2026, saaras v4 added)

Scoring is complete and reproducible offline from the committed results
cache. Nothing bills unless a fresh `run` is started explicitly.

## Done
- sarvam-saaras-v4 (saaras:v4, mode=transcribe) over the full public
  split (7,230 scored; the same 17 over-30 s clips fail as for v3) and
  the held-out split (132/132). Run 3 Sep 2026 on Koe's Sarvam credits.
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

## Publishing (3 Sep 2026)
- Legal review done: Gramvaani (academic-use-only, no redistribution grant)
  is stripped from the committed manifests and excluded from the dataset;
  FLEURS (CC-BY-4.0), MUCS (CC-BY-SA-4.0) and the held-out YouTube chunks
  (CC-BY) ship with attribution. See README "Data and licensing".
- Code: public at https://github.com/Nemukai/koe-benchmark.
- Report: hosted at https://nemukai.com/labs/koe-benchmark.
- Hugging Face dataset + Space: `scripts/publish_hf.py` stages and uploads;
  needs a write token (`HF_TOKEN`).

## Next (unpaid)
- Human spot-listen of a sample of held-out references (silver to gold).
