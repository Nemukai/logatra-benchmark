"""Sarvam AI speech-to-text adapter (Saarika).

Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe

Auth: `SARVAM_API_KEY` (sent as the `api-subscription-key` header).
Model: override with `KOE_SARVAM_MODEL` (default `saarika:v2.5`).

The real-time endpoint accepts audio up to ~30s per request; for longer
benchmark clips either pre-chunk the audio or switch to Sarvam's Batch API
(left as a follow-up; the adapter raises a clear error on HTTP 4xx).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from .base import NotConfigured, Transcription, TranscriptionError, require_env

API_URL = "https://api.sarvam.ai/speech-to-text"

# Manifest language -> Sarvam language_code. "unknown" lets Saarika auto-detect,
# which is also the right setting for code-mixed (Hinglish) audio.
LANGUAGE_MAP = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "od": "od-IN",
    "or": "od-IN",
    "en": "en-IN",
    "en-IN": "en-IN",
}


class SarvamAdapter:
    name = "sarvam"

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        # Accept both the benchmark's name and Koe's production env var name,
        # so `source Koe/.env.local` works without re-exporting.
        api_key = os.environ.get("SARVAM_API_KEY", "").strip() or require_env(
            "SARVAM_API_SUBSCRIPTION_KEY", "Sarvam"
        )
        model = os.environ.get("KOE_SARVAM_MODEL", "saarika:v2.5")
        language_code = LANGUAGE_MAP.get(language or "", "unknown")

        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/wav")}
            data = {"model": model, "language_code": language_code}
            try:
                resp = httpx.post(
                    API_URL,
                    headers={"api-subscription-key": api_key},
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"Sarvam request failed: {exc}") from exc

        if resp.status_code != 200:
            raise TranscriptionError(
                f"Sarvam HTTP {resp.status_code} for {audio_path.name}: {resp.text[:500]}"
            )

        payload = resp.json()
        text = payload.get("transcript")
        if text is None:
            raise TranscriptionError(
                f"Sarvam response missing 'transcript' for {audio_path.name}: {payload}"
            )
        return Transcription(text=text, raw=payload)
