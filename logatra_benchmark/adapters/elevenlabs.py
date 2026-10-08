"""ElevenLabs Scribe STT adapter (stub).

Docs: https://elevenlabs.io/docs/api-reference/speech-to-text/convert

Auth: `ELEVENLABS_API_KEY` (header `xi-api-key`).

Implementation notes for when this is filled in:
- Endpoint: POST https://api.elevenlabs.io/v1/speech-to-text (multipart:
  `file`, `model_id=scribe_v1`).
- `language_code` accepts ISO codes (hi, ta, bn, ...); omit to auto-detect,
  which is the right mode for code-mixed audio.
- `diarize=true` returns per-word `speaker_id`, which maps to SpeakerTurn.
- Response: {language_code, text, words: [{text, start, end, type,
  speaker_id?}]}. Filter words by type == "word" (it also emits "spacing").
"""

from __future__ import annotations

from pathlib import Path

from .base import Transcription, require_env


class ElevenLabsAdapter:
    name = "elevenlabs"

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        require_env("ELEVENLABS_API_KEY", "ElevenLabs Scribe")
        raise NotImplementedError(
            "ElevenLabs adapter is a stub. See module docstring in "
            "logatra_benchmark/adapters/elevenlabs.py for the implementation plan."
        )
