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
    if key in _OPENROUTER_AUDIO:
        from .openrouter_audio import OpenRouterAudioAdapter

        default_model, model_env = _OPENROUTER_AUDIO[key]
        return OpenRouterAudioAdapter(key, default_model, model_env)
    raise ValueError(
        f"unknown provider {name!r}; known: {', '.join(PROVIDERS)}"
    )


# OpenRouter audio-capable chat models used as STT engines:
# provider name -> (default model id, env var for override)
_OPENROUTER_AUDIO = {
    "gemini": ("google/gemini-3.5-flash", "KOE_GEMINI_MODEL"),
    "voxtral": ("mistralai/voxtral-small-24b-2507", "KOE_VOXTRAL_MODEL"),
    "gpt-audio": ("openai/gpt-audio", "KOE_GPT_AUDIO_MODEL"),
}

PROVIDERS = (
    "sarvam",
    "whisper",
    "gemini",
    "voxtral",
    "gpt-audio",
    "deepgram",
    "google",
    "elevenlabs",
    "assemblyai",
)

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
