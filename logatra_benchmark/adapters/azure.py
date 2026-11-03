"""Azure AI Speech fast-transcription adapter.

Docs (checked 2026-11-03):
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
Price: Azure Retail Prices API, meter "Fast Transcription Speech To Text",
$0.36 per audio hour (centralindia and most regions; a $0.10/h "Promo" meter
exists in some regions since 2026-09).

Auth: `AZURE_SPEECH_KEY` (header `Ocp-Apim-Subscription-Key`) plus either
`AZURE_SPEECH_ENDPOINT` (e.g. https://<resource>.cognitiveservices.azure.com)
or `AZURE_SPEECH_REGION` (e.g. centralindia -> https://centralindia.api.cognitive.microsoft.com).

Endpoint: POST {endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15
multipart: `audio` (file) + `definition` (JSON: {"locales": [...],
"profanityFilterMode": "None"}). Synchronous; < 5 h and < 500 MB per file.
Response: {durationMilliseconds, combinedPhrases:[{text}], phrases:[{text,
offsetMilliseconds, durationMilliseconds, words, locale}]}.

All 14 core Indic locales incl. or-IN/ur-IN/as-IN and ne-NP are listed for fast
transcription; Hinglish passes ["hi-IN", "en-IN"] (multi-locale LID).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from . import languages
from .base import (
    HTTPAdapter,
    NotConfigured,
    Transcription,
    TranscriptionError,
    Word,
    http_error,
    require_env,
)

API_VERSION = "2025-10-15"


class AzureSpeechAdapter(HTTPAdapter):
    name = "azure"
    required_env = ("AZURE_SPEECH_KEY", "AZURE_SPEECH_ENDPOINT|AZURE_SPEECH_REGION")
    wants_code_mixed = True
    model = "fast-transcription"

    def __init__(self, client: httpx.Client | None = None, timeout: float = 600.0) -> None:
        super().__init__(client=client, timeout=timeout)

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("azure", language, code_mixed)

    @staticmethod
    def endpoint() -> str:
        ep = os.environ.get("AZURE_SPEECH_ENDPOINT", "").strip().rstrip("/")
        if ep:
            return ep
        region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
        if not region:
            raise NotConfigured("Azure Speech: set AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION.")
        return f"https://{region}.api.cognitive.microsoft.com"

    def definition(self, language: str | None, code_mixed: bool = False) -> dict:
        code = languages.resolve("azure", language, code_mixed)
        canon = languages.canonical(language, code_mixed)
        locales = list(languages.AZURE_MULTI.get(canon, ())) or ([code] if code else [])
        d: dict = {"profanityFilterMode": "None"}
        if locales:
            d["locales"] = locales
        return d

    def transcribe(self, audio_path: Path, language: str | None = None,
                   code_mixed: bool = False) -> Transcription:
        key = require_env("AZURE_SPEECH_KEY", "Azure Speech")
        definition = self.definition(language, code_mixed)
        url = f"{self.endpoint()}/speechtotext/transcriptions:transcribe"
        with audio_path.open("rb") as fh:
            try:
                resp = self.http.post(
                    url,
                    params={"api-version": API_VERSION},
                    headers={"Ocp-Apim-Subscription-Key": key},
                    files={"audio": (audio_path.name, fh, "audio/wav")},
                    data={"definition": json.dumps(definition)},
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"Azure request failed: {exc}") from exc
        if resp.status_code != 200:
            raise http_error("Azure", resp, audio_path)
        payload = resp.json()
        combined = payload.get("combinedPhrases")
        if combined is None:
            raise TranscriptionError(f"Azure response missing combinedPhrases: {str(payload)[:500]}")
        text = " ".join((c.get("text") or "").strip() for c in combined).strip()
        words = []
        for ph in payload.get("phrases") or []:
            for w in ph.get("words") or []:
                off = w.get("offsetMilliseconds")
                dur = w.get("durationMilliseconds")
                words.append(Word(
                    text=w.get("text", ""),
                    start=None if off is None else off / 1000.0,
                    end=None if off is None or dur is None else (off + dur) / 1000.0,
                ))
        payload["_logatra_definition"] = definition
        return Transcription(text=text, words=words or None, raw=payload)
