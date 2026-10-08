"""STT adapter protocol and shared plumbing.

Every provider adapter implements the `STTAdapter` protocol: a `name`, and a
`transcribe(audio_path, language)` method returning a `Transcription`.

Adapters read API keys from environment variables and raise `NotConfigured`
with a helpful message when the key is missing, so the runner can skip (rather
than crash on) unconfigured providers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class NotConfigured(Exception):
    """Raised when a provider's credentials are missing from the environment."""


class TranscriptionError(Exception):
    """Raised when a provider call fails (HTTP error, bad response shape, ...)."""


def require_env(var: str, provider: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise NotConfigured(
            f"{provider}: environment variable {var} is not set. "
            f"Export it (e.g. `export {var}=...`) and re-run."
        )
    return value


@dataclass(slots=True)
class Word:
    """One recognized word with optional timing/speaker info."""

    text: str
    start: float | None = None  # seconds
    end: float | None = None    # seconds
    speaker: str | None = None


@dataclass(slots=True)
class SpeakerTurn:
    """One diarized speaker turn, for DER scoring."""

    speaker: str
    start: float  # seconds
    end: float    # seconds


@dataclass(slots=True)
class Transcription:
    """Normalized output of any STT provider."""

    text: str
    words: list[Word] | None = None
    speaker_turns: list[SpeakerTurn] | None = None
    raw: dict = field(default_factory=dict)  # raw provider response, cached as-is


@runtime_checkable
class STTAdapter(Protocol):
    """Protocol all provider adapters implement."""

    name: str

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        """Transcribe one audio file.

        Args:
            audio_path: path to a local audio file (benchmark uses 16 kHz mono WAV).
            language: optional language hint (manifest `language` field); adapters
                map it to the provider's own language-code scheme or ignore it.

        Raises:
            NotConfigured: credentials missing.
            TranscriptionError: provider call failed.
        """
        ...
