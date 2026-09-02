# Koe Benchmark

An independent benchmark for speech-to-text on Indian languages and Hinglish,
built to answer one practical question honestly: **which STT engine should you
actually run for Indian speech, at what accuracy, and at what price?**

Seven systems, scored on ~1,000 minutes of public Indic test audio and a
fresh held-out split the models have almost certainly never seen:

| System | Model | Coverage |
|---|---|---|
| `sarvam-saaras-v4` | Sarvam saaras:v4 | full public split + held-out |
| `sarvam-saaras` | Sarvam saaras:v3 | full public split + held-out |
| `sarvam` | Sarvam saarika:v2.5 | full public split + held-out |
| `whisper` | OpenAI whisper-1 | full public split + held-out |
| `gemini` | Gemini 3.5 Flash (via OpenRouter) | 800-sample public subset + held-out |
| `voxtral` | Mistral Voxtral Small 24B (via OpenRouter) | 800-sample public subset + held-out |
| `gpt-audio` | OpenAI gpt-audio (via OpenRouter) | 336-sample public subset + held-out |

Because coverage is uneven, the headline table is computed on the
**comparable intersection**: only samples every system completed
(324 public, 132 held-out). Full-split numbers are reported alongside.
Adapters for Deepgram, ElevenLabs, Google, and AssemblyAI exist but have not
been run yet (see Roadmap).

## Results (comparable intersection)

Strict WER, word-weighted; `sa` = script-agnostic WER (fair to code-mix
script choice); `med` = median per-sample WER (robust to repetition loops).

| System | public WER | public med | held-out WER | held-out med | $/audio-hour |
|---|---|---|---|---|---|
| sarvam-saaras-v4 | 0.261 | 0.217 | 0.258 | 0.134 | ~0.54 |
| gemini | 0.265 | 0.233 | 0.222 * | 0.117 * | ~0.43 |
| sarvam-saaras | 0.323 | 0.286 | 0.228 | 0.106 | ~0.54 |
| sarvam | 0.373 | 0.318 | 0.197 * | 0.000 * | ~0.54 |
| gpt-audio | 0.535 | 0.417 | 0.285 | 0.198 | ~1.39 |
| whisper | 0.783 | 0.793 | 0.476 * | 0.390 * | 0.36 |
| voxtral | 2.938 | 0.833 | 0.720 | 0.266 | ~0.36 |

`*` = this system contributed to the held-out references (see below); treat
its held-out score as an upper bound, not a clean measurement. The per-system
`self_ref_rate` in `report/data.json` quantifies exactly how much
(sarvam 58%, gemini 38%, whisper 5% of held-out rows use that system's own
output as the reference).

Three findings worth calling out:

- **Saaras v4 (added September 2026) is the strongest engine on the public
  split and the fastest of all seven** (0.6 s median per clip), but on the
  fresh held-out audio it trails saaras v3 by about three points of
  script-agnostic WER (23.8% vs 21.0%). Both are fully independent of the
  held-out references, so that gap is a clean measurement: on unseen 2026
  conversational Hindi, v3 still edges v4.

- **Voxtral falls into repetition loops** on Indic audio (46 of 896 outputs
  are runaway repetitions), which is why its mean WER exceeds 100% while its
  median is far lower. The explorer shows the raw cases.
- **Whisper collapses on conversational Hindi**: usable on read speech
  (FLEURS), close to unusable on real interview audio in the held-out split.

## Two-layer design

Public Indic test sets (FLEURS, Gramvaani, MUCS) are widely used to train the
very models being evaluated, so scores on them are contamination-prone and
flattering. The benchmark therefore has two layers:

- `split: public` (7,253 samples): FLEURS test for 12 Indic languages
  (as, bn, gu, hi, kn, ml, mr, or, pa, ta, te, ur), Gramvaani Hindi telephony,
  and MUCS 2021 Hindi-English code-mixed speech. A comparable baseline against
  published numbers, treated as a **contaminated baseline**, never the
  headline.
- `split: heldout` (132 samples, ~48 minutes): Hindi/Hinglish conversational
  audio chunked from 11 Creative Commons (CC-BY) YouTube videos uploaded
  January to June 2026, after every tested model's release. Interviews,
  podcasts, and lecture-style speech, heavy on real code-mixing. Currently
  **Hindi only**; see [ATTRIBUTION.md](ATTRIBUTION.md) for the sources.

### How held-out references were made (read this before citing them)

There is no human gold transcription yet. References are **silver**:

1. Three systems (saarika, gemini, whisper) transcribed every chunk; the
   medoid hypothesis under a consensus distance (min of symmetric WER and
   romanized CER, so script choice does not count as disagreement) became the
   candidate reference. 96/132 rows cleared the automatic agreement threshold.
2. The remaining 36 rows were adjudicated one by one: accept when at least
   two systems agree on content, choose the medoid verbatim, override only
   for demonstrable mishears (checkable against domain facts, the video's own
   metadata, or third-system corroboration). Every decision and rationale is
   logged in
   [heldout_yt.adjudications.json](data/manifests/heldout_yt.adjudications.json).
   Adjudication was performed by an LLM from the hypothesis texts, not from
   audio.

Consequences, stated plainly: held-out scores for contributor systems are
upper bounds (a system cannot disagree with its own reference), the
`self_ref_rate` column quantifies the exposure per system, and the cleanest
held-out comparison is between non-contributors (sarvam-saaras-v4, sarvam-saaras, gpt-audio,
voxtral). Independent human transcription of the held-out audio is the top
roadmap item and would upgrade silver to gold.

## Metrics

- **Strict WER/CER** on Indic-aware normalized text: NFC, lowercase,
  Indic digit mapping, Indic punctuation stripping (danda, Urdu marks).
- **Script-agnostic WER**: Devanagari runs are transliterated to Latin
  (IAST, diacritics stripped, schwa deletion) before comparison, so
  "आई वांट टू स्टार्ट" vs "I want to start" is not an error. Reported as a
  separate column, never silently replacing strict WER.
- **Median per-sample WER**: robustness view; loops and hallucinations move
  the mean, not the median.
- **Latency, RTF, cost**: per-call latency from the runner, real-time factor,
  and cost from provider-reported usage or published per-minute rates.

## Reproducing

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Raw provider
responses are cached as JSON under `results/<system>/<sample_id>.json`, so
scoring and the full report rebuild **offline, without any API key**:

```bash
uv sync
uv run python scripts/score_report.py --out report/data.json \
    data/manifests/full_public.jsonl data/manifests/heldout_yt.silver.jsonl
uv run python scripts/export_explorer.py
uv run python scripts/build_report.py     # writes report/index.html
uv run pytest                             # normalization + silver tests
```

To run systems fresh (needs keys, spends money):

```bash
uv run python -m koe_benchmark run \
    --manifest data/manifests/m1_public.jsonl \
    --providers sarvam-saaras,whisper --workers 8
```

Sarvam's real-time endpoint rejects clips longer than 30 s (HTTP 400), so
17 public clips (11 FLEURS, 4 MUCS, 2 Gramvaani) are absent for every Sarvam
entrant; the comparable-intersection tables are unaffected. The Batch API
would close that gap and is on the roadmap.

Keys via env: `SARVAM_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
(gemini/voxtral/gpt-audio), plus `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`,
`ASSEMBLYAI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS` for the unrun adapters.

Gramvaani is the exception: its license is academic-use-only with no
redistribution grant, so the committed manifests carry Gramvaani rows with an
empty `reference_text` and scoring skips them. To reproduce the Gramvaani
numbers, download OpenSLR 118 yourself and merge the references back in:

```bash
uv run python scripts/prepare_gramvaani.py       # downloads GV_Dev_5h + GV_Eval_3h
uv run python scripts/merge_gramvaani_refs.py    # fills reference_text by sample id
```

The audio (about 1.1 GB after excluding Gramvaani) and the raw results cache
are distributed as a Hugging Face dataset (see "Data and licensing" below);
the repo carries manifests, code, and the generated report.

## Layout

```
koe_benchmark/        harness: manifest, metrics, runner, silver consensus, adapters/
data/manifests/       JSONL manifests (public + held-out), adjudication log
scripts/              dataset prep, collection, scoring, report build, HF publish
report/               generated leaderboard + per-case diff explorer (static HTML)
tests/                pure-python tests for normalization and silver consensus
```

## Disclosure

This benchmark is built by [Nemukai](https://nemukai.com). Our product Koe
runs Sarvam saaras:v3 in production; the benchmark exists because we needed to
choose a provider and did not trust vendor-reported numbers. That is a
conflict of interest you should know about. Mitigations: every raw provider
response is cached and shipped, scoring is reproducible offline, the
methodology penalizes nobody by hand, and the reference pipeline's bias
exposure is quantified per system (`self_ref_rate`).

## Roadmap

- Independent human transcription of the held-out split (silver to gold).
- Complete the public split for gemini, voxtral, gpt-audio.
- Route the 17 over-30 s clips through Sarvam's Batch API.
- Run the written-but-unrun adapters: Deepgram, ElevenLabs Scribe, Google,
  AssemblyAI.
- Multi-language held-out audio beyond Hindi.
- Speaker diarization (DER): metric implemented, no diarization runs yet.

## Data and licensing

Code is MIT ([LICENSE](LICENSE)). The audio and transcripts come from four
sources with four different licenses, and only the ones that permit
redistribution are shipped:

| Subset | Source | License | Shipped in the dataset |
|---|---|---|---|
| `fleurs` (1,200 clips, 12 languages) | [google/fleurs](https://huggingface.co/datasets/google/fleurs), test split | CC-BY-4.0 | yes: audio, references, provider outputs |
| `mucs2021` (3,136 clips, Hindi-English) | [OpenSLR 104](https://www.openslr.org/104/), subtask 2 test | CC-BY-SA-4.0 | yes: audio, references, provider outputs (ShareAlike applies to the derived outputs) |
| `heldout_yt` (132 chunks, ~48 min) | 11 YouTube videos, CC-BY as set by each uploader | CC-BY-3.0 | yes: the 16 kHz chunks only, never full videos; per-video credit in [ATTRIBUTION.md](ATTRIBUTION.md) |
| `gramvaani` (2,917 clips, Hindi telephony) | [OpenSLR 118](https://www.openslr.org/118/), Dev 5h + Eval 3h | academic use only; commercial use by permission of contact@gramvaani.org; no redistribution grant | **no**: neither audio, references, nor provider outputs. Scores are reported; reproduce them locally via `scripts/prepare_gramvaani.py` |

Provider outputs (the results cache) are our own API calls, but they are
transcriptions of the source audio, so they inherit the source license and
are excluded wherever the source is. Raw archives under `data/raw/` are never
redistributed.

If you are a rights holder for any of the held-out videos and want your
material removed, open an issue and it will be pulled promptly.
