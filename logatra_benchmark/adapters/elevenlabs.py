"""ElevenLabs Scribe STT adapter (Scribe v2, batch).

Docs (checked 2026-11-03):
    https://elevenlabs.io/docs/api-reference/speech-to-text/convert
    https://elevenlabs.io/docs/overview/capabilities/speech-to-text

Auth: `ELEVENLABS_API_KEY` (header `xi-api-key`).
Model: `scribe_v2` (override: LOGATRA_ELEVENLABS_MODEL).

Endpoint: POST https://api.elevenlabs.io/v1/speech-to-text (multipart: `file`,
`model_id`, `language_code`, `timestamps_granularity=word`,
`tag_audio_events=false`, `diarize`). Limits: < 5 GB upload (convert API
reference; the capability FAQ says 3 GB), up to 10 h per file. Billed per
second of audio including silence.

`language_code` takes ISO-639-1/3; we send ISO-639-3 (hin, tam, ...). For
code-mixed Hinglish we omit it (auto-detect), the documented best mode for
mixed audio. `enable_logging=false` (zero-retention) is opt-in via
LOGATRA_ELEVENLABS_ZERO_RETENTION=1 (enterprise-only on ElevenLabs' side).

Response: {language_code, language_probability, text, words: [{text, start,
end, type, speaker_id?}]}; words of type "spacing"/"audio_event" are dropped.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from . import languages
from .base import (
    HTTPAdapter,
    SpeakerTurn,
    Transcription,
    TranscriptionError,
    Word,
    http_error,
    require_env,
)

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsAdapter(HTTPAdapter):
    name = "elevenlabs"
    required_env = ("ELEVENLABS_API_KEY",)
    wants_code_mixed = True

    def __init__(self, client: httpx.Client | None = None, timeout: float = 600.0) -> None:
        super().__init__(client=client, timeout=timeout)

    @property
    def model(self) -> str:
        return os.environ.get("LOGATRA_ELEVENLABS_MODEL", "").strip() or "scribe_v2"

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("elevenlabs", language, code_mixed)

    def build_form(self, language: str | None, code_mixed: bool = False) -> dict[str, str]:
        code = languages.resolve("elevenlabs", language, code_mixed)
        data = {
            "model_id": self.model,
            "timestamps_granularity": "word",
            "tag_audio_events": "false",
            "diarize": "true" if os.environ.get("LOGATRA_ELEVENLABS_DIARIZE") in ("1", "true") else "false",
        }
        if code:
            data["language_code"] = code
        return data

    def transcribe(self, audio_path: Path, language: str | None = None,
                   code_mixed: bool = False) -> Transcription:
        api_key = require_env("ELEVENLABS_API_KEY", "ElevenLabs Scribe")
        data = self.build_form(language, code_mixed)
        params = {}
        if os.environ.get("LOGATRA_ELEVENLABS_ZERO_RETENTION") in ("1", "true"):
            params["enable_logging"] = "false"
        with audio_path.open("rb") as fh:
            try:
                resp = self.http.post(
                    API_URL,
                    params=params,
                    headers={"xi-api-key": api_key},
                    files={"file": (audio_path.name, fh, "audio/wav")},
                    data=data,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"ElevenLabs request failed: {exc}") from exc
        if resp.status_code != 200:
            raise http_error("ElevenLabs", resp, audio_path)

        payload = resp.json()
        text = payload.get("text")
        if text is None:
            raise TranscriptionError(
                f"ElevenLabs response missing 'text' for {audio_path.name}: {str(payload)[:500]}"
            )
        words: list[Word] = []
        turns: list[SpeakerTurn] = []
        for w in payload.get("words") or []:
            if w.get("type", "word") != "word":
                continue
            spk = w.get("speaker_id")
            words.append(Word(text=w.get("text", ""), start=w.get("start"),
                              end=w.get("end"), speaker=spk))
            if spk is not None and w.get("start") is not None and w.get("end") is not None:
                if turns and turns[-1].speaker == spk:
                    turns[-1].end = w["end"]
                else:
                    turns.append(SpeakerTurn(speaker=spk, start=w["start"], end=w["end"]))
        payload["_logatra_model"] = self.model
        return Transcription(text=text.strip(), words=words or None,
                             speaker_turns=turns or None, raw=payload)
