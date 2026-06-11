"""Audio-capable chat models via OpenRouter, used as STT engines.

One generic adapter covers every OpenRouter model that accepts audio input
parts (verified live via GET /api/v1/models -> architecture.input_modalities).
Registered provider names and their default models:

    gemini     google/gemini-3.5-flash          (override: KOE_GEMINI_MODEL)
    voxtral    mistralai/voxtral-small-24b-2507 (override: KOE_VOXTRAL_MODEL)
    gpt-audio  openai/gpt-audio                 (override: KOE_GPT_AUDIO_MODEL)

Auth: `OPENROUTER_API_KEY`.

These are chat models, not purpose-built STT endpoints, so the transcript
quality depends on the prompt. The prompt pins down the two failure modes that
matter for this benchmark: (1) translating instead of transcribing, and
(2) "helpfully" normalizing code-mixed speech into a single script. No word
timestamps or diarization are available through this path.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx

from .base import NotConfigured, Transcription, TranscriptionError, require_env

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_PROMPT = (
    "Transcribe this audio recording verbatim.\n"
    "Rules:\n"
    "- Output ONLY the transcript text. No preamble, no labels, no commentary, "
    "no markdown.\n"
    "- Keep the original language(s). Do NOT translate.\n"
    "- For code-mixed speech (e.g. Hindi-English), write each word in the script "
    "it was spoken in: Hindi words in Devanagari, English words in Latin script.\n"
    "- Transcribe numbers as spoken.\n"
    "- If the audio is silent or unintelligible, output an empty string."
)

_SUFFIX_TO_FORMAT = {".wav": "wav", ".mp3": "mp3"}


class OpenRouterAudioAdapter:
    """STT via an audio-capable chat model on OpenRouter."""

    def __init__(self, name: str, default_model: str, model_env: str, timeout: float = 300.0) -> None:
        self.name = name
        self._default_model = default_model
        self._model_env = model_env
        self.timeout = timeout

    @property
    def model(self) -> str:
        return os.environ.get(self._model_env, "").strip() or self._default_model

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        api_key = require_env("OPENROUTER_API_KEY", f"OpenRouter ({self.name})")

        audio_format = _SUFFIX_TO_FORMAT.get(audio_path.suffix.lower())
        if audio_format is None:
            raise TranscriptionError(
                f"{self.name}: unsupported audio format {audio_path.suffix!r} "
                "(OpenRouter input_audio accepts wav/mp3)"
            )
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")

        prompt = _PROMPT
        if language:
            prompt += f"\n- Language hint: {language}."

        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": audio_format},
                        },
                    ],
                }
            ],
        }
        try:
            resp = httpx.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://nemukai.com",
                    "X-Title": "Koe Benchmark",
                },
                json=body,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"{self.name} request failed: {exc}") from exc

        if resp.status_code != 200:
            raise TranscriptionError(
                f"{self.name} HTTP {resp.status_code} for {audio_path.name}: {resp.text[:500]}"
            )
        payload = resp.json()
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranscriptionError(
                f"{self.name} response missing content for {audio_path.name}: "
                f"{str(payload)[:500]}"
            ) from exc
        if text is None:
            raise TranscriptionError(f"{self.name}: null content for {audio_path.name}")

        # Keep the raw payload but drop nothing: usage/cost info is useful later.
        payload["_koe_model"] = self.model
        return Transcription(text=text.strip(), raw=payload)
