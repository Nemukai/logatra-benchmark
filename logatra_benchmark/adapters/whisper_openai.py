"""OpenAI hosted transcription adapter (whisper-1, gpt-4o(-mini)-transcribe,
gpt-transcribe).

Docs (checked 2026-11-03):
    https://developers.openai.com/api/docs/guides/speech-to-text
    https://developers.openai.com/api/docs/models/gpt-transcribe
    https://developers.openai.com/api/docs/pricing

Endpoint: POST https://api.openai.com/v1/audio/transcriptions (multipart),
files up to 25 MB. Auth: `OPENAI_API_KEY`.

Registered entrants:
    whisper                 whisper-1 (override: LOGATRA_OPENAI_STT_MODEL, kept
                            for back-compat with existing runs)
    gpt-4o-transcribe       fixed model id
    gpt-4o-mini-transcribe  fixed model id
    gpt-transcribe          OpenAI's current recommended file-transcription
                            model ($0.0045/min); uses `languages[]` hints

NOTE: OpenAI has deprecated whisper-1, gpt-4o-transcribe,
gpt-4o-mini-transcribe and gpt-4o-transcribe-diarize; they shut down on
2027-02-26. Run them before then if they are wanted in the leaderboard.

whisper-1 uses `verbose_json` (word timestamps). The gpt-4o-* models only
support `json`/`text`; their JSON includes `usage` (tokens or seconds). The
model id is recorded in the cached raw response under `_logatra_model`.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from . import languages
from .base import HTTPAdapter, Transcription, TranscriptionError, Word, http_error, require_env

API_URL = "https://api.openai.com/v1/audio/transcriptions"
MAX_BYTES = 25 * 1024 * 1024


class OpenAITranscribeAdapter(HTTPAdapter):
    required_env = ("OPENAI_API_KEY",)

    def __init__(self, name: str = "whisper", default_model: str = "whisper-1",
                 model_env: str | None = "LOGATRA_OPENAI_STT_MODEL",
                 timeout: float = 300.0, client: httpx.Client | None = None) -> None:
        super().__init__(client=client, timeout=timeout)
        self.name = name
        self._default_model = default_model
        self._model_env = model_env

    @property
    def model(self) -> str:
        if self._model_env:
            return os.environ.get(self._model_env, "").strip() or self._default_model
        return self._default_model

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("openai", language)

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        api_key = require_env("OPENAI_API_KEY", "OpenAI")
        model = self.model
        if audio_path.stat().st_size > MAX_BYTES:
            raise TranscriptionError(f"OpenAI: {audio_path.name} exceeds the 25 MB upload limit")

        lang_hint = languages.resolve("openai", language) or None
        multi = None
        canon = languages.canonical(language)
        if model == "gpt-transcribe":
            multi = list(languages.OPENAI_MULTI.get(canon, ())) or ([lang_hint] if lang_hint else None)
            lang_hint = None

        fmt = "verbose_json" if model == "whisper-1" else "json"
        try:
            payload = self._request(api_key, model, audio_path, lang_hint, multi, fmt)
        except TranscriptionError as exc:
            # whisper-1 rejects several Indic hint codes (as, or, ...);
            # fall back to auto-detect rather than failing the sample.
            msg = str(exc)
            is_lang_400 = "HTTP 400" in msg and (
                "unsupported_language" in msg or "language" in msg.lower()
            )
            if not is_lang_400 or not (lang_hint or multi):
                raise
            payload = self._request(api_key, model, audio_path, None, None, fmt)
            payload["_logatra_hint_dropped"] = True
        if payload is None:  # model rejected verbose_json
            payload = self._request(api_key, model, audio_path, lang_hint, multi, "json")
            if payload is None:
                raise TranscriptionError(f"OpenAI: no usable response for {audio_path.name}")

        payload["_logatra_model"] = model
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
        lang_list: list[str] | None,
        response_format: str,
    ) -> dict | None:
        """POST once; return parsed JSON, or None if the format is unsupported."""
        data: dict[str, str | list[str]] = {"model": model, "response_format": response_format}
        if response_format == "verbose_json":
            data["timestamp_granularities[]"] = "word"
        if lang_list:
            data["languages[]"] = lang_list
        elif lang_hint:
            data["language"] = lang_hint

        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/wav")}
            try:
                resp = self.http.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"OpenAI request failed: {exc}") from exc

        if (
            resp.status_code == 400
            and response_format == "verbose_json"
            and "unsupported_language" not in resp.text
            and "language" not in resp.text.lower()
        ):
            return None  # caller falls back to plain json
        if resp.status_code != 200:
            raise http_error("OpenAI", resp, audio_path)
        return resp.json()


class WhisperOpenAIAdapter(OpenAITranscribeAdapter):
    """Back-compat name: whisper-1 with the LOGATRA_OPENAI_STT_MODEL override."""

    def __init__(self, timeout: float = 300.0, client: httpx.Client | None = None) -> None:
        super().__init__(name="whisper", default_model="whisper-1",
                         model_env="LOGATRA_OPENAI_STT_MODEL", timeout=timeout, client=client)
