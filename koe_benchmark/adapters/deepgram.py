"""Deepgram STT adapter (stub).

Docs: https://developers.deepgram.com/docs/pre-recorded-audio

Auth: `DEEPGRAM_API_KEY` (header `Authorization: Token <key>`).

Implementation notes for when this is filled in:
- Endpoint: POST https://api.deepgram.com/v1/listen with the raw audio body
  (Content-Type: audio/wav) and query params.
- Model: `nova-3` (check current Indic language support; `nova-2` supports
  hi, ta and a few others; hi-Latn exists for romanized Hindi which is
  directly relevant to the Hinglish track).
- Query params: `model`, `language`, `punctuate=true`, `diarize=true`
  (Deepgram returns per-word `speaker` ints, which map to SpeakerTurn),
  `smart_format=false` for benchmark fairness.
- Response path: results.channels[0].alternatives[0].transcript and .words[].
"""

from __future__ import annotations

from pathlib import Path

from .base import Transcription, require_env


class DeepgramAdapter:
    name = "deepgram"

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        require_env("DEEPGRAM_API_KEY", "Deepgram")
        raise NotImplementedError(
            "Deepgram adapter is a stub. See module docstring in "
            "koe_benchmark/adapters/deepgram.py for the implementation plan."
        )
