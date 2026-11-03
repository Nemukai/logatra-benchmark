"""AssemblyAI pre-recorded STT adapter (Universal-3.5 Pro / Universal-2).

Docs (checked 2026-11-03):
    https://www.assemblyai.com/docs/getting-started/models
    https://www.assemblyai.com/docs/pre-recorded-audio/supported-languages
    https://www.assemblyai.com/docs/pre-recorded-audio/api-reference/transcripts/submit
    https://www.assemblyai.com/pricing

Auth: `ASSEMBLYAI_API_KEY` (header `authorization: <key>`).

Flow (async):
    POST https://api.assemblyai.com/v2/upload        raw bytes -> {upload_url}
    POST https://api.assemblyai.com/v2/transcript    {audio_url, speech_models,
                                                      language_code | language_detection}
    GET  https://api.assemblyai.com/v2/transcript/{id}  poll until
         status == "completed" | "error"

Model routing (see languages.ASSEMBLYAI):
    hi, hi-en, en-IN -> speech_models ["universal-3-5-pro", "universal-2"]
                        (U-3.5 Pro: 18 languages incl. Hindi, native code-switching)
    other Indic      -> ["universal-2"] (99 languages; accuracy "fair" for most Indic)
    or (Odia)        -> unsupported, skipped
Language: explicit `language_code` by default. LOGATRA_ASSEMBLYAI_DETECT=1
uses `language_detection: true` instead (useful to compare auto-LID).

Words come back with ms timestamps (converted to seconds); `speaker_labels`
(opt-in, LOGATRA_ASSEMBLYAI_DIARIZE=1, +$0.02/hr) yields utterances.
"""

from __future__ import annotations

import os
import time
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

API_BASE = "https://api.assemblyai.com"
U35 = "universal-3-5-pro"
U2 = "universal-2"
U35_LANGS = {"hi", "hi-en", "en-IN"}


def speech_models_for(language: str | None, code_mixed: bool = False) -> list[str]:
    canon = languages.canonical(language, code_mixed)
    if canon is None or canon in U35_LANGS:
        return [U35, U2]
    return [U2]


class AssemblyAIAdapter(HTTPAdapter):
    name = "assemblyai"
    required_env = ("ASSEMBLYAI_API_KEY",)
    wants_code_mixed = True

    def __init__(self, client: httpx.Client | None = None, timeout: float = 120.0,
                 poll_interval_s: float = 3.0, max_wait_s: float = 1800.0,
                 sleep=time.sleep) -> None:
        super().__init__(client=client, timeout=timeout)
        self.poll_interval_s = poll_interval_s
        self.max_wait_s = max_wait_s
        self._sleep = sleep

    @property
    def model(self) -> str:
        return os.environ.get("LOGATRA_ASSEMBLYAI_MODELS", "").strip() or f"{U35},{U2}"

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("assemblyai", language, code_mixed)

    def build_request(self, audio_url: str, language: str | None,
                      code_mixed: bool = False) -> dict:
        code = languages.resolve("assemblyai", language, code_mixed)
        override = os.environ.get("LOGATRA_ASSEMBLYAI_MODELS", "").strip()
        models = [m.strip() for m in override.split(",") if m.strip()] if override \
            else speech_models_for(language, code_mixed)
        body: dict = {
            "audio_url": audio_url,
            "speech_models": models,
            "punctuate": True,
            "format_text": True,
        }
        if os.environ.get("LOGATRA_ASSEMBLYAI_DETECT") in ("1", "true") or not code:
            body["language_detection"] = True
        else:
            body["language_code"] = code
        if os.environ.get("LOGATRA_ASSEMBLYAI_DIARIZE") in ("1", "true"):
            body["speaker_labels"] = True
        return body

    def _req(self, method: str, url: str, key: str, **kw) -> dict:
        try:
            resp = self.http.request(method, url, headers={"authorization": key},
                                     timeout=self.timeout, **kw)
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"AssemblyAI request failed: {exc}") from exc
        if resp.status_code != 200:
            raise http_error("AssemblyAI", resp)
        return resp.json()

    def transcribe(self, audio_path: Path, language: str | None = None,
                   code_mixed: bool = False) -> Transcription:
        key = require_env("ASSEMBLYAI_API_KEY", "AssemblyAI")
        body_lang = self.build_request("", language, code_mixed)  # validates language first
        up = self._req("POST", f"{API_BASE}/v2/upload", key, content=audio_path.read_bytes())
        upload_url = up.get("upload_url")
        if not upload_url:
            raise TranscriptionError(f"AssemblyAI upload returned no upload_url: {up}")
        body_lang["audio_url"] = upload_url
        job = self._req("POST", f"{API_BASE}/v2/transcript", key, json=body_lang)
        tid = job.get("id")
        if not tid:
            raise TranscriptionError(f"AssemblyAI submit returned no id: {job}")

        deadline = time.monotonic() + self.max_wait_s
        payload = job
        while payload.get("status") not in ("completed", "error"):
            if time.monotonic() > deadline:
                raise TranscriptionError(f"AssemblyAI transcript {tid} timed out")
            self._sleep(self.poll_interval_s)
            payload = self._req("GET", f"{API_BASE}/v2/transcript/{tid}", key)
        if payload["status"] == "error":
            raise TranscriptionError(f"AssemblyAI transcript {tid} error: {payload.get('error')}")

        text = payload.get("text") or ""
        words = [
            Word(text=w.get("text", ""),
                 start=None if w.get("start") is None else w["start"] / 1000.0,
                 end=None if w.get("end") is None else w["end"] / 1000.0,
                 speaker=w.get("speaker"))
            for w in payload.get("words") or []
        ] or None
        turns = [
            SpeakerTurn(speaker=str(u.get("speaker")), start=u["start"] / 1000.0,
                        end=u["end"] / 1000.0)
            for u in payload.get("utterances") or []
            if u.get("start") is not None and u.get("end") is not None
        ] or None
        payload["_logatra_request"] = {k: v for k, v in body_lang.items() if k != "audio_url"}
        return Transcription(text=text.strip(), words=words, speaker_turns=turns, raw=payload)
