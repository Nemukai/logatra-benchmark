"""Deepgram pre-recorded STT adapter (Nova-3).

Docs (checked 2026-11-03):
    https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded
    https://developers.deepgram.com/docs/models-languages-overview
    https://developers.deepgram.com/docs/multilingual-code-switching
    https://developers.deepgram.com/docs/nova-3-language-prompting

Auth: `DEEPGRAM_API_KEY` (header `Authorization: Token <key>`).
Model: `nova-3` (override: LOGATRA_DEEPGRAM_MODEL).

Request: POST https://api.deepgram.com/v1/listen?<params> with the raw WAV
body (Content-Type: audio/wav).

Language handling (see languages.DEEPGRAM):
  * monolingual Nova-3 code (hi, bn, ta, te, mr, gu, kn, pa, ur, as, ne, en-IN)
  * Hinglish (hi-en, or hi + code_mixed) -> language=multi with
    language_hint=hi&language_hint=en (Nova-3 multilingual code-switching,
    billed at the multilingual rate). Set LOGATRA_DEEPGRAM_CODEMIX=mono to
    send language=hi instead.
  * Nova-3 has no Malayalam/Odia -> skipped.

`smart_format=false` and `punctuate=true` for benchmark fairness (numbers are
not reformatted to digits). Diarization is off by default (enable with
LOGATRA_DEEPGRAM_DIARIZE=1); per-word `speaker` ints map to SpeakerTurn.
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

API_URL = "https://api.deepgram.com/v1/listen"


def _turns_from_words(words: list[Word]) -> list[SpeakerTurn] | None:
    turns: list[SpeakerTurn] = []
    for w in words:
        if w.speaker is None or w.start is None or w.end is None:
            continue
        if turns and turns[-1].speaker == w.speaker:
            turns[-1].end = w.end
        else:
            turns.append(SpeakerTurn(speaker=w.speaker, start=w.start, end=w.end))
    return turns or None


class DeepgramAdapter(HTTPAdapter):
    name = "deepgram"
    required_env = ("DEEPGRAM_API_KEY",)
    wants_code_mixed = True

    def __init__(self, client: httpx.Client | None = None, timeout: float = 300.0) -> None:
        super().__init__(client=client, timeout=timeout)

    @property
    def model(self) -> str:
        return os.environ.get("LOGATRA_DEEPGRAM_MODEL", "").strip() or "nova-3"

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("deepgram", language, code_mixed)

    def build_params(self, language: str | None, code_mixed: bool = False) -> list[tuple[str, str]]:
        code = languages.resolve("deepgram", language, code_mixed)
        canon = languages.canonical(language, code_mixed)
        params: list[tuple[str, str]] = [
            ("model", self.model),
            ("punctuate", "true"),
            ("smart_format", "false"),
        ]
        if code == "multi" and os.environ.get("LOGATRA_DEEPGRAM_CODEMIX", "").lower() == "mono":
            code = "hi"
        if code:
            params.append(("language", code))
            if code == "multi":
                for hint in languages.DEEPGRAM_MULTI_HINTS.get(canon, ()):
                    params.append(("language_hint", hint))
        else:
            params.append(("detect_language", "true"))
        if os.environ.get("LOGATRA_DEEPGRAM_DIARIZE", "") in ("1", "true"):
            params.append(("diarize", "true"))
        return params

    def transcribe(self, audio_path: Path, language: str | None = None,
                   code_mixed: bool = False) -> Transcription:
        api_key = require_env("DEEPGRAM_API_KEY", "Deepgram")
        params = self.build_params(language, code_mixed)
        try:
            resp = self.http.post(
                API_URL,
                params=params,
                headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
                content=audio_path.read_bytes(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Deepgram request failed: {exc}") from exc
        if resp.status_code != 200:
            raise http_error("Deepgram", resp, audio_path)

        payload = resp.json()
        try:
            alt = payload["results"]["channels"][0]["alternatives"][0]
            text = alt["transcript"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranscriptionError(
                f"Deepgram response missing transcript for {audio_path.name}: {str(payload)[:500]}"
            ) from exc

        words = [
            Word(
                text=w.get("punctuated_word") or w.get("word", ""),
                start=w.get("start"),
                end=w.get("end"),
                speaker=None if w.get("speaker") is None else f"spk{w['speaker']}",
            )
            for w in alt.get("words") or []
        ] or None
        payload["_logatra_model"] = self.model
        payload["_logatra_params"] = params
        return Transcription(
            text=text.strip(),
            words=words,
            speaker_turns=_turns_from_words(words) if words else None,
            raw=payload,
        )
