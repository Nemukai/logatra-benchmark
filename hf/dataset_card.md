---
license:
  - cc-by-4.0
  - cc-by-sa-4.0
  - cc-by-3.0
language:
  - hi
  - bn
  - ta
  - te
  - mr
  - gu
  - kn
  - ml
  - pa
  - or
  - as
  - ur
  - en
task_categories:
  - automatic-speech-recognition
tags:
  - indic
  - hinglish
  - code-switching
  - benchmark
  - speech-to-text
  - evaluation
pretty_name: Logatra Benchmark
size_categories:
  - 1K<n<10K
---

# Logatra Benchmark

Audio, manifests, and raw provider outputs for the
[Logatra Benchmark](https://github.com/Nemukai/logatra-benchmark), an independent
speech-to-text benchmark for Indian languages and Hinglish built by
[Nemukai](https://nemukai.com). Seven commercial STT systems are scored on a
public split and on a fresh held-out split of 2026 Creative Commons audio that
no model could have trained on. The interactive leaderboard and per-case
explorer live at [nemukai.com/labs/logatra-benchmark](https://nemukai.com/labs/logatra-benchmark).

Scoring and the full report regenerate offline from this data with the code
in the GitHub repository; no API key is needed.

## What is here

| Path | Contents |
|---|---|
| `manifests/*.jsonl` | one JSON object per sample: `id`, relative `audio_path`, `reference_text`, `language`, `code_mixed`, `source`, `license`, `split` |
| `audio/fleurs/` | 1,200 FLEURS test clips, 12 Indic languages, 16 kHz mono WAV |
| `audio/mucs2021/` | 3,136 MUCS 2021 Hindi-English code-switched clips, 16 kHz mono WAV |
| `audio/heldout_yt/` | 132 chunks (about 48 minutes) cut from 11 CC-BY YouTube videos published January to June 2026 |
| `results/<system>/<sample_id>.json` | the raw cached response of every provider call: text, latency, and the provider's own payload |
| `report/data.json`, `report/rows.json` | aggregated and per-sample metrics (WER, CER, script-agnostic WER, latency, cost) |
| `manifests/heldout_yt.adjudications.json` | every adjudication decision for the 36 held-out rows that failed the automatic consensus gate |

Systems: Sarvam saaras:v4, Sarvam saaras:v3, Sarvam saarika:v2.5, OpenAI whisper-1, Gemini 3.5
Flash, Mistral Voxtral Small 24B, OpenAI gpt-audio.

## What is deliberately not here

The Gramvaani Hindi telephony subset (OpenSLR 118, 2,917 clips) is licensed
for academic use only, with commercial use by permission and no
redistribution grant. Its audio, transcripts, and the provider outputs derived
from it are excluded. The manifests keep the Gramvaani rows with an empty
`reference_text` so sample ids stay stable; the GitHub repository documents how
to download OpenSLR 118 yourself and merge the references back in.

Full-length held-out videos are never redistributed, only the evaluated
chunks.

## Held-out references are silver, not gold

There is no human gold transcription of the held-out split yet. References
are the medoid of three systems' hypotheses under a script-agnostic consensus
distance, with 36 of 132 rows adjudicated case by case from the hypothesis
texts. Systems that contributed hypotheses (saarika, Gemini, Whisper) score
against references they partly wrote, so their held-out scores are upper
bounds. The per-system exposure is reported as `self_ref_rate` in
`report/data.json`. Read the methodology section of the repository README
before citing held-out numbers.

## Licensing

Each subset keeps the license of its source, and every manifest row carries
its own `license` field.

| Subset | Source | License |
|---|---|---|
| `fleurs` | [google/fleurs](https://huggingface.co/datasets/google/fleurs), test split (Conneau et al., 2022) | CC-BY-4.0 |
| `mucs2021` | [OpenSLR 104](https://www.openslr.org/104/), MUCS 2021 subtask 2 (Diwan et al., 2021) | CC-BY-SA-4.0 |
| `heldout_yt` | 11 YouTube videos whose uploaders chose the Creative Commons Attribution license; per-video credit in `ATTRIBUTION.md` | CC-BY-3.0 |
| `results/`, `report/` | Nemukai's own API calls and measurements, derived from the audio above | same license as the underlying subset; the MUCS-derived outputs are CC-BY-SA-4.0 |

If you are a rights holder for any of the held-out videos and want your
material removed, open an issue on the GitHub repository and it will be pulled
promptly.

## Citation

```bibtex
@misc{logatra-benchmark-2026,
  title  = {Logatra Benchmark: speech-to-text on Indian languages and Hinglish},
  author = {Singh, Priyansh Kumar},
  year   = {2026},
  url    = {https://github.com/Nemukai/logatra-benchmark}
}
```

Please also cite the source datasets: FLEURS (Conneau et al., 2022), MUCS 2021
(Diwan et al., 2021).

## Disclosure

Nemukai's product Logatra runs Sarvam saaras:v3 in production. The benchmark
exists because we needed to choose a provider and did not trust
vendor-reported numbers. Every raw provider response is shipped here so the
scoring can be checked.
