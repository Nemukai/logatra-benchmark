"""Google Cloud Speech-to-Text adapter (stub).

Docs: https://cloud.google.com/speech-to-text/v2/docs

Auth: `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service-account JSON file
(standard Google ADC). Unlike the other providers this is OAuth2, not a simple
API key, so the clean implementation uses `google-cloud-speech` (add it as an
optional extra) rather than hand-rolled httpx calls.

Implementation notes for when this is filled in:
- Use Speech-to-Text v2 `recognize` with model `chirp_2` (best multilingual
  coverage incl. Indic) or `latest_long` for long-form audio.
- Language codes: hi-IN, ta-IN, bn-IN, te-IN, ... Google also accepts multiple
  candidate languages, useful for code-mixed audio.
- Diarization: enable `diarization_config` (v1) / speaker labels to populate
  SpeakerTurn; chirp models may not support diarization, check per model.
- Audio over 60s must go through batch recognition with a GCS URI; for the
  benchmark either chunk locally or wire up a scratch GCS bucket.
"""

from __future__ import annotations

from pathlib import Path

from .base import Transcription, require_env


class GoogleAdapter:
    name = "google"

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        require_env("GOOGLE_APPLICATION_CREDENTIALS", "Google Cloud STT")
        raise NotImplementedError(
            "Google adapter is a stub. See module docstring in "
            "koe_benchmark/adapters/google.py for the implementation plan."
        )
