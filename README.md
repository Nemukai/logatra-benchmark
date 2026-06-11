# Koe Benchmark

An independent benchmark for speech-to-text (STT) on Indian languages. It compares
commercial and open providers (Sarvam, OpenAI Whisper large-v3, Google, Deepgram,
ElevenLabs Scribe, AssemblyAI) on three axes:

1. **Indic language accuracy**: WER/CER across major Indian languages with an
   Indic-aware text normalization pipeline.
2. **Hinglish / code-mixing**: accuracy on code-switched speech, where script
   choice (Devanagari vs Latin) makes naive WER misleading.
3. **Speaker diarization**: DER on multi-speaker conversational audio.

## Two-layer design

Public Indic speech datasets (AI4Bharat corpora, FLEURS, Common Voice, etc.) are
widely used to train the very models being evaluated, so scores on them are
contamination-prone and flattering. Koe Benchmark therefore splits every manifest
into two layers:

- `split: public`. Samples drawn from established public test sets. Useful as a
  comparable baseline against published numbers, but treated as a
  **contaminated baseline**, not a headline result.
- `split: heldout`. Fresh audio the providers have almost certainly never seen:
  Creative Commons licensed YouTube audio collected after model release dates
  (see `scripts/collect_youtube.py`) plus self-collected recordings, all
  transcribed by humans. **Headline numbers come from this split only.**

Reporting both splits per provider also gives a rough contamination signal: a
large public-vs-heldout gap suggests the public sets leak into training data.

## Layout

```
koe_benchmark/
  manifest.py       # JSONL manifest format + loader
  metrics.py        # WER/CER (jiwer) with Indic normalization; optional DER
  runner.py         # runs providers over a manifest, caches raw responses
  adapters/         # one adapter per STT provider (Sarvam + OpenAI implemented)
scripts/
  collect_youtube.py  # CC-licensed YouTube audio collector (heldout split)
tests/              # pure-python tests for the normalization pipeline
results/            # cached raw provider responses + score artifacts
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # core deps (jiwer, httpx, rich)
uv sync --extra diarization      # + pyannote.metrics for DER
uv sync --extra collect          # + yt-dlp for the YouTube collector
```

Provider API keys are read from environment variables:

| Provider   | Env var              |
|------------|----------------------|
| Sarvam     | `SARVAM_API_KEY`     |
| OpenAI     | `OPENAI_API_KEY`     |
| Deepgram   | `DEEPGRAM_API_KEY`   |
| Google     | `GOOGLE_APPLICATION_CREDENTIALS` (service account JSON path) |
| ElevenLabs | `ELEVENLABS_API_KEY` |
| AssemblyAI | `ASSEMBLYAI_API_KEY` |

## Running

```bash
# Run two providers over a manifest, cache raw responses under results/
uv run python -m koe_benchmark run \
    --manifest data/manifests/hindi-v1.jsonl \
    --providers sarvam,whisper

# Re-score from cache without hitting any API
uv run python -m koe_benchmark run --manifest ... --providers sarvam --score-only
```

Raw provider responses are cached as JSON in `results/<provider>/<sample_id>.json`,
so reruns are idempotent and scoring is reproducible offline.

## Manifest format

One JSON object per line (JSONL):

```json
{
  "id": "hi-heldout-0042",
  "audio_path": "audio/hi/0042.wav",
  "reference_text": "...",
  "language": "hi",
  "code_mixed": false,
  "num_speakers": 1,
  "speaker_segments": null,
  "source": "youtube:VIDEO_ID",
  "license": "CC-BY-3.0",
  "split": "heldout"
}
```

`speaker_segments` is an optional RTTM-style list of
`{"speaker": "spk0", "start": 0.0, "end": 4.2}` used for DER.

## Tests

```bash
uv run pytest
```
