"""Provider adapters. `get_adapter(name)` is the registry entry point."""

from __future__ import annotations

from .base import (
    NotConfigured,
    SpeakerTurn,
    STTAdapter,
    Transcription,
    TranscriptionError,
    Unsupported,
    Word,
)


def get_adapter(name: str, **kwargs) -> STTAdapter:
    """Instantiate an adapter by provider name (lazy imports keep startup fast).

    `kwargs` (e.g. `client=httpx.Client(transport=...)`) are forwarded to the
    adapter constructor; tests use this to inject mock transports.
    """
    key = name.strip().lower()
    if key == "sarvam":
        from .sarvam import SarvamAdapter

        return SarvamAdapter(**kwargs)
    if key in ("sarvam-saaras", "saaras"):
        from .sarvam import SarvamAdapter

        return SarvamAdapter(name="sarvam-saaras", default_model="saaras:v3",
                             mode="transcribe", **kwargs)
    if key in ("sarvam-saaras-v4", "saaras-v4"):
        from .sarvam import SarvamAdapter

        return SarvamAdapter(name="sarvam-saaras-v4", default_model="saaras:v4",
                             mode="transcribe", **kwargs)
    if key in ("whisper", "openai", "whisper_openai"):
        from .whisper_openai import WhisperOpenAIAdapter

        return WhisperOpenAIAdapter(**kwargs)
    if key in _OPENAI_FIXED:
        from .whisper_openai import OpenAITranscribeAdapter

        return OpenAITranscribeAdapter(name=key, default_model=_OPENAI_FIXED[key],
                                       model_env=None, **kwargs)
    if key == "deepgram":
        from .deepgram import DeepgramAdapter

        return DeepgramAdapter(**kwargs)
    if key == "google":
        from .google import GoogleAdapter

        return GoogleAdapter(**kwargs)
    if key == "elevenlabs":
        from .elevenlabs import ElevenLabsAdapter

        return ElevenLabsAdapter(**kwargs)
    if key == "assemblyai":
        from .assemblyai import AssemblyAIAdapter

        return AssemblyAIAdapter(**kwargs)
    if key == "azure":
        from .azure import AzureSpeechAdapter

        return AzureSpeechAdapter(**kwargs)
    if key == "indicconformer":
        from .indicconformer import IndicConformerAdapter

        return IndicConformerAdapter()
    if key in _OPENROUTER_AUDIO:
        from .openrouter_audio import OpenRouterAudioAdapter

        default_model, model_env = _OPENROUTER_AUDIO[key]
        return OpenRouterAudioAdapter(key, default_model, model_env, **kwargs)
    raise ValueError(
        f"unknown provider {name!r}; known: {', '.join(PROVIDERS)}"
    )


# OpenRouter audio-capable chat models used as STT engines:
# provider name -> (default model id, env var for override)
_OPENROUTER_AUDIO = {
    "gemini": ("google/gemini-3.5-flash", "LOGATRA_GEMINI_MODEL"),
    "voxtral": ("mistralai/voxtral-small-24b-2507", "LOGATRA_VOXTRAL_MODEL"),
    "gpt-audio": ("openai/gpt-audio", "LOGATRA_GPT_AUDIO_MODEL"),
}

# OpenAI transcription entrants with a fixed model id (no env override, so
# each cache directory maps to exactly one model).
_OPENAI_FIXED = {
    "gpt-4o-transcribe": "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe": "gpt-4o-mini-transcribe",
    "gpt-transcribe": "gpt-transcribe",
}

PROVIDERS = (
    "sarvam",
    "sarvam-saaras",
    "sarvam-saaras-v4",
    "whisper",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-transcribe",
    "gemini",
    "voxtral",
    "gpt-audio",
    "deepgram",
    "google",
    "elevenlabs",
    "assemblyai",
    "azure",
    "indicconformer",
)

__all__ = [
    "STTAdapter",
    "Transcription",
    "Word",
    "SpeakerTurn",
    "NotConfigured",
    "TranscriptionError",
    "Unsupported",
    "get_adapter",
    "PROVIDERS",
]
