"""AssemblyAI STT adapter (stub).

Docs: https://www.assemblyai.com/docs/api-reference/transcripts

Auth: `ASSEMBLYAI_API_KEY` (header `authorization`).

Implementation notes for when this is filled in:
- Two-step async flow: (1) POST audio bytes to
  https://api.assemblyai.com/v2/upload -> upload_url, (2) POST
  /v2/transcript with {audio_url, language_code, speaker_labels: true,
  speech_model: "universal"} then poll GET /v2/transcript/{id} until
  status == "completed" | "error".
- Indic coverage: hi is GA; check current language matrix for ta/te/bn etc.
  before claiming coverage in results.
- `speaker_labels` returns `utterances` with speaker letters -> SpeakerTurn.
- Words come back with ms timestamps; convert to seconds for Word.
"""

from __future__ import annotations

from pathlib import Path

from .base import Transcription, require_env


class AssemblyAIAdapter:
    name = "assemblyai"

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        require_env("ASSEMBLYAI_API_KEY", "AssemblyAI")
        raise NotImplementedError(
            "AssemblyAI adapter is a stub. See module docstring in "
            "koe_benchmark/adapters/assemblyai.py for the implementation plan."
        )
