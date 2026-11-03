# Providers

How each entrant in the Logatra Benchmark is called, what it costs, and how to
run it without overspending. Every model name, endpoint, language list and
price here was checked against the provider's own docs on **2026-11-03**
(sources inline). The machine-readable versions live in
`logatra_benchmark/pricing.py` (prices) and
`logatra_benchmark/adapters/languages.py` (language codes).

No adapter has been run against a live API yet. Every request shape is
covered by tests with mocked HTTP (`tests/test_adapters.py`).

## Credits available to the owner

| Provider | Credit (YC deals) | List-price headroom |
|---|---|---|
| Deepgram | $15,000 | about 2.9M min of Nova-3 multilingual |
| Google Cloud | $2,000 | about 125k min of chirp_3 at $0.016/min |
| OpenAI | $1,000 | about 222k min of gpt-transcribe |
| Sarvam | $1,000 | about 170k min at Rs 0.49/min |

ElevenLabs, AssemblyAI, Azure and OpenRouter are paid from a card.

## Summary

All prices are USD per audio minute at pay-as-you-go list rates for
pre-recorded (batch) audio, with no credits or volume tiers applied.

| Entrant (`--providers`) | Model | Price | Env vars |
|---|---|---|---|
| `sarvam-saaras-v4` | `saaras:v4` | Rs 0.49 (~$0.0059) | `SARVAM_API_KEY` |
| `sarvam-saaras` | `saaras:v3` | Rs 0.49 | `SARVAM_API_KEY` |
| `sarvam` | `saarika:v2.5` (deprecated) | Rs 0.49 | `SARVAM_API_KEY` |
| `deepgram` | `nova-3` | $0.0043 mono / $0.0052 multi | `DEEPGRAM_API_KEY` |
| `google` | `chirp_3` (STT v2) | $0.016 | `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT` |
| `elevenlabs` | `scribe_v2` | $0.0037 ($0.22/h) | `ELEVENLABS_API_KEY` |
| `assemblyai` | `universal-3-5-pro` / `universal-2` | $0.0035 / $0.0025 | `ASSEMBLYAI_API_KEY` |
| `gpt-transcribe` | `gpt-transcribe` | $0.0045 | `OPENAI_API_KEY` |
| `gpt-4o-transcribe` | `gpt-4o-transcribe` | $0.006 | `OPENAI_API_KEY` |
| `gpt-4o-mini-transcribe` | `gpt-4o-mini-transcribe` | $0.003 | `OPENAI_API_KEY` |
| `whisper` | `whisper-1` | $0.006 | `OPENAI_API_KEY` |
| `azure` | Fast transcription | $0.006 ($0.36/h) | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` or `AZURE_SPEECH_ENDPOINT` |
| `gemini`, `voxtral`, `gpt-audio` | via OpenRouter | per token (`raw.usage.cost`) | `OPENROUTER_API_KEY` |
| `indicconformer` | `ai4bharat/indic-conformer-600m-multilingual` | $0 (local) | none (`uv sync --extra indicconformer`) |

## Language coverage

Canonical codes are the manifest `language` field. `hi` with
`code_mixed: true` is treated as `hi-en`, which is how the held-out split
labels Hinglish. `en` is treated as `en-IN` and `od` as `or`.

`Y` means an explicit language code is sent. `auto` means the sample runs
without a language hint. `-` means unsupported: the runner skips the sample,
records the reason, does not count it as an error and does not bill it.

| provider | hi | hi-en | en-IN | bn | ta | te | mr | gu | kn | ml | pa | or | as | ur | kok | mai | ne | sd | ks | doi | mni | brx | sat | sa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepgram | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | Y | - | Y | Y | - | - | Y | - | - | - | - | - | - | - |
| google | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | - | - | Y | - | - | - | - | - | - | - |
| elevenlabs | Y | auto | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | - | Y | Y | - | - | - | - | - | - |
| assemblyai | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | Y | Y | - | - | Y | Y | - | - | - | - | - | Y |
| openai (all 4) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | auto | Y | Y | - | - | Y | Y | - | - | - | - | - | Y |
| sarvam saaras v3/v4 | Y | auto* | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| sarvam saarika | Y | auto* | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | auto* | auto* | - | - | - | - | - | - | - | - | - | - |
| azure | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | - | - | Y | - | - | - | - | - | - | - |
| indicconformer | Y | Y | - | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| OpenRouter models | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto | auto |

\* Sarvam sends `language_code=unknown` (auto-detect). That matches how the
cached Sarvam results were produced, so comparisons stay fair.

How each provider handles Hinglish (`hi-en`):

- **Deepgram** sends `language=multi` with `language_hint=hi` and `language_hint=en`. This is billed at the multilingual rate.
- **Google** sends `hi-IN`. You can override this with `LOGATRA_GOOGLE_CODEMIX`, for example `auto`.
- **ElevenLabs** auto-detects.
- **AssemblyAI** uses Universal-3.5 Pro with `hi`. That model handles code-switching natively.
- **gpt-transcribe** sends `languages[]=hi,en`. The other OpenAI models send `language=hi`.
- **Azure** sends the locales `["hi-IN","en-IN"]`, which turns on language ID.

## Per provider

### Sarvam (`sarvam-saaras-v4`, `sarvam-saaras`, `sarvam`)

- **Models:** `saaras:v4` is the default, recommended model. It covers 22 Indic languages, Indian English and Global English, and supports keyterms. `saaras:v3` is still available. `saarika:v2.5` is deprecated by Sarvam and covers 10 Indic languages plus en-IN. Sources: https://docs.sarvam.ai/api/api-guides-tutorials/models/saaras and https://docs.sarvam.ai/api/getting-started/models/saarika.
- **REST endpoint:** `POST https://api.sarvam.ai/speech-to-text`. Send multipart `file`, `model`, `mode=transcribe` and `language_code`, with the header `api-subscription-key`. It accepts **30 s or less per request**.
- **Batch API:** used automatically for clips over 30 s. It is saaras only and takes up to 2 h per file and 20 files per job. The adapter makes these calls:
  1. `POST /speech-to-text/job/v1` with `{job_parameters}`
  2. `POST /speech-to-text/job/v1/upload-files`
  3. `PUT` to the returned SAS URL with the header `x-ms-blob-type: BlockBlob`
  4. `POST /speech-to-text/job/v1/{id}/start`
  5. Poll `GET /speech-to-text/job/v1/{id}/status` every 5 s or more
  6. `POST /speech-to-text/job/v1/download-files`, then `GET` the output JSON

  Source: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api. Saarika clips over 30 s are skipped with a reason.
- **Price:** Rs 0.49/min pay-as-you-go (Pro Rs 0.45, Growth Rs 0.41). It is the same for REST, streaming and batch, billed per second, excluding GST. Diarization (batch) is Rs 0.59/min. Source: https://docs.sarvam.ai/api/getting-started/pricing.
  - The repo's old Rs 0.75/min figure is stale.
  - The third-party figure of "Rs 30/h (Rs 45/h diarized)" is close to the current Rs 29.4/h, but the official sheet is now per minute and lists diarization at Rs 35.4/h.
- **Rate limits (pay-as-you-go):** 60 req/min for REST and 20 req/min for batch, so use `--workers 2` or fewer.
- **Env:** `SARVAM_API_KEY` (or `SARVAM_API_SUBSCRIPTION_KEY`). `LOGATRA_SARVAM_MODEL` overrides the model for every Sarvam entrant, so avoid it.

### Deepgram (`deepgram`)

- **Model:** `nova-3`, Deepgram's best general pre-recorded model. Override with `LOGATRA_DEEPGRAM_MODEL`.
- **Endpoint:** `POST https://api.deepgram.com/v1/listen?model=nova-3&language=..&punctuate=true&smart_format=false`. Send the raw WAV body with the header `Authorization: Token <key>`.
- **Languages:** monolingual Nova-3 covers hi, bn, ta, te, mr, gu, kn, pa, ur, as, ne and en-IN. There is no Malayalam or Odia. Multilingual (`language=multi`) covers hi, bn, ta, te, mr, gu, kn and en among 55 languages. `language_hint` is only valid with `multi`. Sources: https://developers.deepgram.com/docs/models-languages-overview and https://developers.deepgram.com/docs/nova-3-language-prompting.
- **Price:** pre-recorded pay-as-you-go is $0.0043/min monolingual and $0.0052/min multilingual. Growth is $0.0036 and $0.0043. Streaming costs more ($0.0077 and $0.0092). Billed per second. Source: https://deepgram.com/pricing (schema.org offers on the page).
- **Options:** `LOGATRA_DEEPGRAM_DIARIZE=1` turns on diarization. `LOGATRA_DEEPGRAM_CODEMIX=mono` sends `hi` instead of `multi` for Hinglish.

### Google Cloud Speech-to-Text v2 (`google`)

- **Model:** `chirp_3`. Source: https://cloud.google.com/speech-to-text/docs/models/chirp-3.
- **Region:** GA only in the `us` and `eu` multi-regions. Set it with `LOGATRA_GOOGLE_LOCATION` (default `us`). The recognizer is `_` (implicit) unless you set `LOGATRA_GOOGLE_RECOGNIZER`.
- **Endpoints:**
  - `POST https://us-speech.googleapis.com/v2/projects/{p}/locations/us/recognizers/_:recognize` for clips of 60 s / 10 MB or less, with inline base64 audio.
  - `:batchRecognize` for longer clips. This needs `LOGATRA_GOOGLE_GCS_BUCKET`: the adapter uploads to `gs://bucket/logatra-tmp/`, polls the operation and deletes the object afterwards. Without a bucket, long clips are skipped.
- **Auth:** a service account through ADC. Set `GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json` and `GOOGLE_CLOUD_PROJECT` (or `LOGATRA_GOOGLE_PROJECT`), then run `uv sync --extra google` to install google-auth. The service account needs Cloud Speech client access, plus object create/delete on the bucket if you use batch.
- **Languages:** hi-IN and en-IN are GA. bn, ta, te, mr, gu, kn, ml, `pa-Guru-IN`, or, as and ne-NP are Preview. There is **no Urdu**. Diarization exists only for a few locales (en-IN and hi-IN among them) and is not requested.
- **Price:** v2 Standard (chirp included) is $0.016/min for 0 to 500k min/month, then $0.01, $0.008 and $0.004 per min. Dynamic batch is $0.003/min but is not used, because results can take up to 24 h. Source: https://cloud.google.com/speech-to-text/pricing.

### ElevenLabs Scribe (`elevenlabs`)

- **Model:** `scribe_v2`. Override with `LOGATRA_ELEVENLABS_MODEL`.
- **Endpoint:** `POST https://api.elevenlabs.io/v1/speech-to-text`. Send multipart `file`, `model_id`, ISO-639-3 `language_code`, `timestamps_granularity=word` and `tag_audio_events=false`, with the header `xi-api-key`.
- **Limits:** the API reference says files must be under 5 GB, while the FAQ says 3 GB. Files can run up to 10 h.
- **Languages:** 90+ languages, including all core Indic languages plus ne and sd. Source: https://elevenlabs.io/docs/overview/capabilities/speech-to-text.
- **Price:** "$0.22 per hour (Scribe)", with keyterms +$0.05/h. Silence is billed. Source: the FAQ on https://elevenlabs.io/pricing/api. The per-plan table renders client-side and could not be read, so plan-specific rates were **not verified**.
- **Options:** `LOGATRA_ELEVENLABS_DIARIZE=1` turns on diarization. `LOGATRA_ELEVENLABS_ZERO_RETENTION=1` sends `enable_logging=false` (enterprise only).

### AssemblyAI (`assemblyai`)

- **Models:** `universal-3-5-pro` covers 18 languages incl. Hindi, with native code-switching. `universal-2` covers 99 languages. Sources: https://www.assemblyai.com/docs/getting-started/models and https://www.assemblyai.com/docs/pre-recorded-audio/supported-languages.
- **Routing:**
  - hi, hi-en and en-IN send `speech_models=["universal-3-5-pro","universal-2"]`.
  - Other languages send `["universal-2"]`. Universal-2's documented accuracy for most Indic languages is "fair (>50% WER)".
  - Odia is unsupported.
  - Set `LOGATRA_ASSEMBLYAI_DETECT=1` to use `language_detection` instead of an explicit `language_code`.
- **Flow:** `POST /v2/upload`, then `POST /v2/transcript`, then poll `GET /v2/transcript/{id}`. The header is `authorization: <key>`.
- **Price:** Universal-3.5 Pro $0.21/h, Universal-2 $0.15/h, diarization +$0.02/h. Source: https://www.assemblyai.com/pricing.

### OpenAI (`gpt-transcribe`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper`)

- **Endpoint:** `POST https://api.openai.com/v1/audio/transcriptions` (multipart), up to 25 MB per file. Source: https://developers.openai.com/api/docs/guides/speech-to-text.
- **Models:**
  - **`gpt-transcribe`** is OpenAI's current recommended model, at $0.0045/min. It takes `languages[]` hints. Source: https://developers.openai.com/api/docs/models/gpt-transcribe.
  - **`gpt-4o-transcribe`** ($0.006/min) and **`gpt-4o-mini-transcribe`** ($0.003/min) return `json` only.
  - **`whisper-1`** ($0.006/min) uses `verbose_json` with word timestamps.
- **Deprecation:** whisper-1, gpt-4o-transcribe and gpt-4o-mini-transcribe are **deprecated and shut down 2027-02-26**. Benchmark them before then.
- **Price source:** https://developers.openai.com/api/docs/pricing.
- **Languages:** ISO-639-1 hints. Whisper has no Odia, so Odia runs unhinted. When a hint is rejected (`unsupported_language`), the adapter retries without it and marks the record `_logatra_hint_dropped`.
- **Env:** `OPENAI_API_KEY`. `LOGATRA_OPENAI_STT_MODEL` still overrides the `whisper` entrant only. The other three entrants have fixed model ids.

### Azure AI Speech (`azure`)

- **Endpoint:** fast transcription at `POST {endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15`. Send multipart `audio` plus a `definition` JSON (`locales`, `profanityFilterMode=None`). It is synchronous and accepts files under 5 h and 500 MB. Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create.
- **Env:** `AZURE_SPEECH_KEY`, plus `AZURE_SPEECH_REGION` (for example `centralindia`) or `AZURE_SPEECH_ENDPOINT`.
- **Languages:** all 14 core locales are available for fast transcription, incl. or-IN, as-IN and ur-IN, plus ne-NP. Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support.
- **Price:** $0.36/h from the Azure Retail Prices API (meter "Fast Transcription Speech To Text", centralindia and most regions). A "$0.10/h Promo" meter exists in some regions from 2026-09-01. The marketing page renders prices client-side, so it was read through `https://prices.azure.com/api/retail/prices`.

### OpenRouter chat models (`gemini`, `voxtral`, `gpt-audio`)

These are unchanged. They are billed per token, and the actual cost comes back
in `raw.usage.cost`. `pricing.py` holds only a rough `estimate_usd_per_min`
for budget reservations ($0.01, $0.005 and $0.04). These are **guesses, not
list prices**.

### AI4Bharat IndicConformer (`indicconformer`, skeleton)

- **Model:** open weights, MIT license, gated on Hugging Face: `ai4bharat/indic-conformer-600m-multilingual`. It covers 22 scheduled languages and has no English.
- **Setup:** `uv sync --extra indicconformer` installs torch, torchaudio, transformers and onnxruntime. Accept the model terms and run `huggingface-cli login`.
- **Status:** never run. The `model(wav, lang, "ctc"|"rnnt")` call follows the public model card. The card is gated and could not be re-read on 2026-11-03.

## Runbook

```bash
uv sync --extra dev            # + --extra google for chirp_3
set -a; source .env; set +a    # keys; never commit .env

# 1. Estimate (no network): audio minutes, cost, rough wall time, what is
#    cached, what is unsupported, which keys are missing.
uv run python -m logatra_benchmark estimate \
  --manifest data/manifests/m1_public.jsonl \
  --manifest data/manifests/heldout_yt.silver.jsonl \
  --providers deepgram,google,elevenlabs,assemblyai,gpt-transcribe,sarvam-saaras-v4

# 2. Dry run (no network): validates provider names, language support and
#    key presence; exits 3 if a key is missing.
uv run python -m logatra_benchmark run --manifest data/manifests/m1_public.jsonl \
  --providers deepgram --dry-run

# 3. Smoke test: 5 samples, hard cap $0.10.
uv run python -m logatra_benchmark run --manifest data/manifests/m1_public.jsonl \
  --providers deepgram --limit 5 --max-usd 0.10 --yes-spend

# 4. Full run with a budget cap. Re-running is free for cached samples.
uv run python -m logatra_benchmark run --manifest data/manifests/full_public.jsonl \
  --providers deepgram --workers 4 --max-usd 25 --yes-spend --json > logs/deepgram.json

# Private split: results default to data/private/results/ (never results/).
uv run python -m logatra_benchmark estimate --manifest data/private/manifests/air.jsonl \
  --providers sarvam-saaras-v4,deepgram
```

### Safety rules in the runner

- **Explicit opt-in.** `run` refuses to send uncached samples unless you pass `--yes-spend` (or set `LOGATRA_YES_SPEND=1`). It prints the estimate and exits 2. `--score-only`, and runs where every sample is already cached, need no flag.
- **No re-billing.** The cache is `results/<provider>/<sample_id>.json`, written atomically. Cached samples are never re-sent. Errors are never cached, so a re-run retries only the failures.
- **Budget cap.** `--max-usd` reserves each call's list-price estimate before sending and stops before the total would pass the cap, even with `--workers` above 1. The cap is global across providers. The report shows `budget.spent_usd`, whether the run stopped, and how many samples were not sent.
- **Retries.** HTTP 429/5xx and network errors are retried with backoff (2 s, 8 s, 20 s).
- **Concurrency.** `--workers` defaults to 1. Suggested settings:
  - Sarvam pay-as-you-go: 2 (60 REST req/min, 20 batch req/min)
  - Deepgram: 4 to 8
  - OpenAI, ElevenLabs, AssemblyAI, Azure: 2 to 4
- **Skips are not errors.** Unsupported languages and over-limit clips, such as Google above 60 s without a bucket or saarika above 30 s, are reported as `unsupported` with a reason. They are not counted as errors and not billed.
- **New cache fields.** New records also store `language`, `code_mixed`, `audio_s`, `model` and `cost_usd`, so the scorer can price them through `pricing.record_cost_usd`.

## Unverified claims

- ElevenLabs per-plan STT rates: only the FAQ figure of $0.22/h was readable.
- Azure: the list price comes from the retail prices API, not the rendered pricing page. The fast-transcription locale list was read from the language-support table.
- IndicConformer inference API: the model card is gated, so the call follows earlier public docs.
- OpenRouter per-minute figures are rough budgeting guesses.
- Chirp 3 multi-language `languageCodes` for code-mix was not verified. The default is a single `hi-IN`.
- Deepgram diarization price: the pricing page lists no diarization add-on for pre-recorded audio, so it is assumed to be included.
