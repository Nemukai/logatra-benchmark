"""Provider adapters. `get_adapter(name)` is the registry entry point."""

from __future__ import annotations

from .base import (
    NotConfigured,
    SpeakerTurn,
    STTAdapter,
    Transcription,
    TranscriptionError,
    Word,
)


def get_adapter(name: str) -> STTAdapter:
    """Instantiate an adapter by provider name (lazy imports keep startup fast)."""
    key = name.strip().lower()
    if key == "sarvam":
        from .sarvam import SarvamAdapter

        return SarvamAdapter()
    if key in ("whisper", "openai", "whisper_openai"):
        from .whisper_openai import WhisperOpenAIAdapter

        return WhisperOpenAIAdapter()
    if key == "deepgram":
        from .deepgram import DeepgramAdapter

        return DeepgramAdapter()
    if key == "google":
        from .google import GoogleAdapter

        return GoogleAdapter()
    if key == "elevenlabs":
        from .elevenlabs import ElevenLabsAdapter

        return ElevenLabsAdapter()
    if key == "assemblyai":
        from .assemblyai import AssemblyAIAdapter

        return AssemblyAIAdapter()
    raise ValueError(
        f"unknown provider {name!r}; known: sarvam, whisper, deepgram, google, "
        "elevenlabs, assemblyai"
    )


PROVIDERS = ("sarvam", "whisper", "deepgram", "google", "elevenlabs", "assemblyai")

__all__ = [
    "STTAdapter",
    "Transcription",
    "Word",
    "SpeakerTurn",
    "NotConfigured",
    "TranscriptionError",
    "get_adapter",
    "PROVIDERS",
]
