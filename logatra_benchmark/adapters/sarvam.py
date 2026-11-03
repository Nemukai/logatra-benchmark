"""Sarvam AI speech-to-text adapter (Saaras v4 / v3, legacy Saarika v2.5).

Docs (checked 2026-11-03):
    REST   https://docs.sarvam.ai/api-reference/speech-to-text/transcribe
    Batch  https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/batch-api
           https://docs.sarvam.ai/api-reference/speech-to-text/stt/job/initiate
    Models https://docs.sarvam.ai/api/api-guides-tutorials/models/saaras

Auth: `SARVAM_API_KEY` (or Logatra's `SARVAM_API_SUBSCRIPTION_KEY`), sent as
the `api-subscription-key` header.

Model: override with `LOGATRA_SARVAM_MODEL` (applies to every Sarvam entrant;
prefer the dedicated entrant names instead).

Two paths:
  * REST  POST /speech-to-text, multipart, <= 30 s of audio per request.
  * Batch (saaras only) for clips > 30 s (up to 2 h/file):
        POST /speech-to-text/job/v1                {job_parameters}
        POST /speech-to-text/job/v1/upload-files   {job_id, files} -> SAS URLs
        PUT  <SAS url>  (x-ms-blob-type: BlockBlob)
        POST /speech-to-text/job/v1/{job_id}/start
        GET  /speech-to-text/job/v1/{job_id}/status    (poll, >= 5 s)
        POST /speech-to-text/job/v1/download-files {job_id, files} -> SAS URLs
        GET  <SAS url>  -> output JSON (same `transcript` field as REST)
    Saarika is not available on the Batch API: >30 s saarika samples are
    skipped (Unsupported), not errored.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from . import languages
from .base import (
    HTTPAdapter,
    NotConfigured,
    Transcription,
    TranscriptionError,
    Unsupported,
    audio_duration_s,
    http_error,
    require_env,
)

API_BASE = "https://api.sarvam.ai"
API_URL = f"{API_BASE}/speech-to-text"
JOB_URL = f"{API_BASE}/speech-to-text/job/v1"

REST_MAX_SECONDS = 30.0

# Back-compat export (older code imported this). Prefer languages.SARVAM_*.
LANGUAGE_MAP = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN", "od": "od-IN",
    "or": "od-IN", "en": "en-IN", "en-IN": "en-IN",
}


class SarvamAdapter(HTTPAdapter):
    """Sarvam STT. Registered entrants sharing this adapter:

    sarvam           -> saarika:v2.5 (legacy line, deprecated by Sarvam)
    sarvam-saaras    -> saaras:v3, mode=transcribe (Logatra production's model)
    sarvam-saaras-v4 -> saaras:v4, mode=transcribe (Sarvam's default/latest)
    """

    required_env = ("SARVAM_API_KEY|SARVAM_API_SUBSCRIPTION_KEY",)

    def __init__(self, name: str = "sarvam", default_model: str = "saarika:v2.5",
                 mode: str | None = None, timeout: float = 120.0,
                 client: httpx.Client | None = None,
                 poll_interval_s: float = 5.0, batch_timeout_s: float = 1800.0,
                 sleep=time.sleep) -> None:
        super().__init__(client=client, timeout=timeout)
        self.name = name
        self.default_model = default_model
        self.mode = mode
        self.poll_interval_s = poll_interval_s
        self.batch_timeout_s = batch_timeout_s
        self._sleep = sleep

    @property
    def model(self) -> str:
        return os.environ.get("LOGATRA_SARVAM_MODEL", "") or self.default_model

    @property
    def _table(self) -> str:
        return "sarvam-saarika" if self.model.startswith("saarika") else "sarvam-saaras"

    @property
    def supports_batch(self) -> bool:
        return self.model.startswith("saaras")

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        # Sarvam keeps the manifest language as-is (code_mixed is not used, to
        # stay comparable with cached runs).
        return languages.unsupported_reason(self._table, language)

    def _api_key(self) -> str:
        key = os.environ.get("SARVAM_API_KEY", "").strip()
        if key:
            return key
        try:
            return require_env("SARVAM_API_SUBSCRIPTION_KEY", "Sarvam")
        except NotConfigured:
            raise NotConfigured(
                "Sarvam: environment variable SARVAM_API_KEY (or "
                "SARVAM_API_SUBSCRIPTION_KEY) is not set."
            ) from None

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        api_key = self._api_key()
        language_code = languages.resolve(self._table, language) or "unknown"

        try:
            duration = audio_duration_s(audio_path)
        except FileNotFoundError:
            raise
        if duration > REST_MAX_SECONDS:
            if not self.supports_batch:
                raise Unsupported(
                    f"{self.name}: {duration:.1f}s clip exceeds the {REST_MAX_SECONDS:.0f}s "
                    f"REST limit and {self.model} is not offered on the Batch API"
                )
            return self._transcribe_batch(api_key, audio_path, language_code, duration)
        return self._transcribe_rest(api_key, audio_path, language_code)

    # -- REST -----------------------------------------------------------------
    def _transcribe_rest(self, api_key: str, audio_path: Path, language_code: str) -> Transcription:
        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/wav")}
            data = {"model": self.model, "language_code": language_code}
            if self.mode:
                data["mode"] = self.mode
            try:
                resp = self.http.post(
                    API_URL,
                    headers={"api-subscription-key": api_key},
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                raise TranscriptionError(f"Sarvam request failed: {exc}") from exc

        if resp.status_code != 200:
            raise http_error("Sarvam", resp, audio_path)

        payload = resp.json()
        text = payload.get("transcript")
        if text is None:
            raise TranscriptionError(
                f"Sarvam response missing 'transcript' for {audio_path.name}: {payload}"
            )
        return Transcription(text=text, raw=payload)

    # -- Batch ----------------------------------------------------------------
    def _call(self, method: str, url: str, api_key: str, **kw) -> dict:
        try:
            resp = self.http.request(
                method, url, headers={"api-subscription-key": api_key},
                timeout=self.timeout, **kw,
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Sarvam batch request failed: {exc}") from exc
        if resp.status_code not in (200, 201, 202):
            raise http_error("Sarvam batch", resp)
        return resp.json()

    def _transcribe_batch(self, api_key: str, audio_path: Path, language_code: str,
                          duration: float) -> Transcription:
        params: dict = {"model": self.model, "language_code": language_code,
                        "with_timestamps": False, "with_diarization": False}
        if self.mode:
            params["mode"] = self.mode
        job = self._call("POST", JOB_URL, api_key, json={"job_parameters": params})
        job_id = job.get("job_id")
        if not job_id:
            raise TranscriptionError(f"Sarvam batch: no job_id in {job}")

        fname = audio_path.name
        up = self._call("POST", f"{JOB_URL}/upload-files", api_key,
                        json={"job_id": job_id, "files": [fname]})
        try:
            put_url = up["upload_urls"][fname]["file_url"]
        except (KeyError, TypeError) as exc:
            raise TranscriptionError(f"Sarvam batch: no upload url in {up}") from exc
        try:
            put = self.http.put(
                put_url, content=audio_path.read_bytes(),
                headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "audio/wav"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Sarvam batch upload failed: {exc}") from exc
        if put.status_code not in (200, 201):
            raise http_error("Sarvam batch upload", put, audio_path)

        self._call("POST", f"{JOB_URL}/{job_id}/start", api_key)

        deadline = time.monotonic() + self.batch_timeout_s
        while True:
            status = self._call("GET", f"{JOB_URL}/{job_id}/status", api_key)
            state = status.get("job_state")
            if state == "Completed":
                break
            if state == "Failed":
                raise TranscriptionError(
                    f"Sarvam batch job {job_id} failed: {status.get('error_message')}"
                )
            if time.monotonic() > deadline:
                raise TranscriptionError(f"Sarvam batch job {job_id} timed out ({state})")
            self._sleep(self.poll_interval_s)

        outputs: list[str] = []
        for detail in status.get("job_details") or []:
            if detail.get("state") not in (None, "Success"):
                raise TranscriptionError(
                    f"Sarvam batch file failed: {detail.get('error_message')}"
                )
            outputs += [o["file_name"] for o in detail.get("outputs") or [] if "file_name" in o]
        if not outputs:
            outputs = ["0.json"]
        dl = self._call("POST", f"{JOB_URL}/download-files", api_key,
                        json={"job_id": job_id, "files": outputs[:1]})
        try:
            get_url = dl["download_urls"][outputs[0]]["file_url"]
        except (KeyError, TypeError) as exc:
            raise TranscriptionError(f"Sarvam batch: no download url in {dl}") from exc
        try:
            out = self.http.get(get_url, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Sarvam batch download failed: {exc}") from exc
        if out.status_code != 200:
            raise http_error("Sarvam batch download", out, audio_path)
        payload = out.json()
        text = payload.get("transcript")
        if text is None:
            raise TranscriptionError(f"Sarvam batch output missing 'transcript': {payload}")
        payload["_logatra_batch"] = {"job_id": job_id, "audio_s": round(duration, 2)}
        return Transcription(text=text, raw=payload)
