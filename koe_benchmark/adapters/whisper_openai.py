"""OpenAI hosted Whisper adapter.

Docs: https://platform.openai.com/docs/api-reference/audio/createTranscription

Auth: `OPENAI_API_KEY`.
Model: override with `KOE_OPENAI_STT_MODEL` (default `whisper-1`).

Caveat: the hosted `whisper-1` endpoint is not guaranteed to be Whisper
large-v3. For a true large-v3 number, run the open weights locally (e.g. via
faster-whisper) behind this same adapter interface, or set
`KOE_OPENAI_STT_MODEL=gpt-4o-transcribe` to benchmark OpenAI's newer STT model.
Record which model id produced each result; it is stored in the cached raw
response under `_koe_model`.

Uses `verbose_json` (supported by whisper-1) to also capture word timestamps.
For models that reject verbose_json, it falls back to plain `json`.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from .base import NotConfigured, Transcription, TranscriptionError, Word, require_env

API_URL = "https://api.openai.com/v1/audio/transcriptions"


class WhisperOpenAIAdapter:
    name = "whisper"

    def __init__(self, timeout: float = 300.0) -> None:
        self.timeout = timeout

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        api_key = require_env("OPENAI_API_KEY", "OpenAI")
        model = os.environ.get("KOE_OPENAI_STT_MODEL", "whisper-1")

        # Whisper takes ISO 639-1 hints ("hi", "ta", ...). Strip any region tag.
        lang_hint = (language or "").split("-")[0] or None

        payload = self._request(api_key, model, audio_path, lang_hint, "verbose_json")
        if payload is None:  # model rejected verbose_json
            payload = self._request(api_key, model, audio_path, lang_hint, "json")
            if payload is None:
                raise TranscriptionError(f"OpenAI: no usable response for {audio_path.name}")

        payload["_koe_model"] = model
        text = payload.get("text")
        if text is None:
            raise TranscriptionError(
                f"OpenAI response missing 'text' for {audio_path.name}: {payload}"
            )

        words = None
        if isinstance(payload.get("words"), list):
            words = [
                Word(text=w.get("word", ""), start=w.get("start"), end=w.get("end"))
                for w in payload["words"]
            ]
        return Transcription(text=text, words=words, raw=payload)

    def _request(
        self,
        api_key: str,
        model: str,
        audio_path: Path,
        lang_hint: str | None,
        response_format: str,
    ) -> dict | None:
        """POST once; return parsed JSON, or None if the format is unsupported."""
        data: dict[str, str] = {"model": model, "response_format": response_format}
        if response_format == "verbose_json":
            data["timestamp_granularities[]"] = "word"
        if lang_hint:
            data["language"] = lang_hint

        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/wav")}
            try:
                resp = httpx.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"OpenAI request failed: {exc}") from exc

        if resp.status_code == 400 and response_format == "verbose_json":
            return None  # caller falls back to plain json
        if resp.status_code != 200:
            raise TranscriptionError(
                f"OpenAI HTTP {resp.status_code} for {audio_path.name}: {resp.text[:500]}"
            )
        return resp.json()
