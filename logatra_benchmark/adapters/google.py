"""Google Cloud Speech-to-Text v2 adapter (Chirp 3).

Docs (checked 2026-11-03):
    https://cloud.google.com/speech-to-text/docs/models/chirp-3
    https://cloud.google.com/speech-to-text/docs/sync-recognize
    https://cloud.google.com/speech-to-text/pricing

Auth: service account via Application Default Credentials:
    GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json   (needs roles/speech.client,
                                                     + storage write for >60 s)
    GOOGLE_CLOUD_PROJECT=<project id>   (or LOGATRA_GOOGLE_PROJECT)
Token minting uses the optional `google-auth` dependency:
    uv sync --extra google

Config env:
    LOGATRA_GOOGLE_LOCATION    default "us" (Chirp 3 is GA in the `us` and `eu`
                               multi-regions only)
    LOGATRA_GOOGLE_RECOGNIZER  default "_" (implicit recognizer; config inline)
    LOGATRA_GOOGLE_MODEL       default "chirp_3"
    LOGATRA_GOOGLE_GCS_BUCKET  bucket for clips > 60 s / > 10 MB (BatchRecognize
                               needs a gs:// URI). Unset -> long clips skipped.
    LOGATRA_GOOGLE_CODEMIX     language list for Hinglish, default "hi-IN"
                               (e.g. "auto" for Chirp 3 language-agnostic mode)

REST:
    POST https://{loc}-speech.googleapis.com/v2/projects/{p}/locations/{loc}/
         recognizers/{r}:recognize        {config, content(base64)}   <= 60 s
    POST .../recognizers/{r}:batchRecognize {config, files:[{uri}],
         recognitionOutputConfig:{inlineResponseConfig:{}}}  -> Operation
    GET  https://{loc}-speech.googleapis.com/v2/{operation.name}  poll `done`

Indic locales other than hi-IN/en-IN are Preview for chirp_3; Urdu is not
offered (skipped). Diarization on chirp_3 is only for a few locales (en-IN,
hi-IN among them) and is not requested here.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

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
)

SYNC_MAX_SECONDS = 60.0
SYNC_MAX_BYTES = 10 * 1024 * 1024
SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _project() -> str:
    for var in ("LOGATRA_GOOGLE_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    raise NotConfigured(
        "Google Cloud STT: set GOOGLE_CLOUD_PROJECT (or LOGATRA_GOOGLE_PROJECT)."
    )


class _ADCTokenProvider:
    """Mints OAuth2 access tokens from ADC (google-auth), refreshing as needed."""

    def __init__(self) -> None:
        self._creds = None
        self._lock = threading.Lock()

    def __call__(self) -> str:
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
            raise NotConfigured(
                "Google Cloud STT: GOOGLE_APPLICATION_CREDENTIALS is not set "
                "(path to a service-account JSON)."
            )
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise NotConfigured(
                "Google Cloud STT needs google-auth: `uv sync --extra google`."
            ) from exc
        with self._lock:
            if self._creds is None:
                self._creds, _ = google.auth.default(scopes=[SCOPE])
            if not self._creds.valid:
                self._creds.refresh(Request())
            return self._creds.token


class GoogleAdapter(HTTPAdapter):
    name = "google"
    required_env = ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT|LOGATRA_GOOGLE_PROJECT")
    wants_code_mixed = True

    def __init__(self, client: httpx.Client | None = None, timeout: float = 300.0,
                 token_provider=None, poll_interval_s: float = 5.0,
                 max_wait_s: float = 3600.0, sleep=time.sleep) -> None:
        super().__init__(client=client, timeout=timeout)
        self._token = token_provider or _ADCTokenProvider()
        self.poll_interval_s = poll_interval_s
        self.max_wait_s = max_wait_s
        self._sleep = sleep

    @property
    def model(self) -> str:
        return os.environ.get("LOGATRA_GOOGLE_MODEL", "").strip() or "chirp_3"

    @property
    def location(self) -> str:
        return os.environ.get("LOGATRA_GOOGLE_LOCATION", "").strip() or "us"

    @property
    def recognizer(self) -> str:
        return os.environ.get("LOGATRA_GOOGLE_RECOGNIZER", "").strip() or "_"

    def endpoint(self) -> str:
        loc = self.location
        return "https://speech.googleapis.com" if loc == "global" else f"https://{loc}-speech.googleapis.com"

    def recognizer_path(self, project: str) -> str:
        return f"projects/{project}/locations/{self.location}/recognizers/{self.recognizer}"

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("google", language, code_mixed)

    def language_codes(self, language: str | None, code_mixed: bool = False) -> list[str]:
        code = languages.resolve("google", language, code_mixed)
        if languages.canonical(language, code_mixed) == "hi-en":
            env = os.environ.get("LOGATRA_GOOGLE_CODEMIX", "").strip()
            if env:
                return [c.strip() for c in env.split(",") if c.strip()]
        return [code] if code else ["auto"]

    def build_config(self, language: str | None, code_mixed: bool = False) -> dict:
        return {
            "autoDecodingConfig": {},
            "model": self.model,
            "languageCodes": self.language_codes(language, code_mixed),
            "features": {"enableAutomaticPunctuation": True},
        }

    def _headers(self, project: str) -> dict:
        return {"Authorization": f"Bearer {self._token()}", "x-goog-user-project": project}

    def _post(self, url: str, headers: dict, **kw) -> httpx.Response:
        try:
            return self.http.post(url, headers=headers, timeout=self.timeout, **kw)
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Google request failed: {exc}") from exc

    def transcribe(self, audio_path: Path, language: str | None = None,
                   code_mixed: bool = False) -> Transcription:
        project = _project()
        config = self.build_config(language, code_mixed)  # raises Unsupported first
        duration = audio_duration_s(audio_path)
        size = audio_path.stat().st_size
        headers = self._headers(project)
        if duration <= SYNC_MAX_SECONDS and size <= SYNC_MAX_BYTES:
            return self._recognize(project, headers, config, audio_path)
        bucket = os.environ.get("LOGATRA_GOOGLE_GCS_BUCKET", "").strip()
        if not bucket:
            raise Unsupported(
                f"google: {duration:.1f}s clip exceeds the 60 s sync limit; set "
                "LOGATRA_GOOGLE_GCS_BUCKET to enable BatchRecognize"
            )
        return self._batch(project, headers, config, audio_path, bucket)

    @staticmethod
    def _join(results: list[dict]) -> str:
        parts = []
        for r in results or []:
            alts = r.get("alternatives") or []
            if alts and alts[0].get("transcript"):
                parts.append(alts[0]["transcript"].strip())
        return " ".join(p for p in parts if p)

    def _recognize(self, project: str, headers: dict, config: dict, audio_path: Path) -> Transcription:
        url = f"{self.endpoint()}/v2/{self.recognizer_path(project)}:recognize"
        body = {"config": config,
                "content": base64.b64encode(audio_path.read_bytes()).decode("ascii")}
        resp = self._post(url, headers, json=body)
        if resp.status_code != 200:
            raise http_error("Google", resp, audio_path)
        payload = resp.json()
        payload["_logatra_model"] = self.model
        payload["_logatra_config"] = config
        return Transcription(text=self._join(payload.get("results")), raw=payload)

    def _batch(self, project: str, headers: dict, config: dict, audio_path: Path,
               bucket: str) -> Transcription:
        obj = f"logatra-tmp/{uuid.uuid4().hex}-{audio_path.name}"
        up_url = (f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
                  f"?uploadType=media&name={quote(obj, safe='')}")
        up = self._post(up_url, {**headers, "Content-Type": "audio/wav"},
                        content=audio_path.read_bytes())
        if up.status_code != 200:
            raise http_error("Google GCS upload", up, audio_path)
        uri = f"gs://{bucket}/{obj}"
        try:
            url = f"{self.endpoint()}/v2/{self.recognizer_path(project)}:batchRecognize"
            body = {"config": config, "files": [{"uri": uri}],
                    "recognitionOutputConfig": {"inlineResponseConfig": {}}}
            resp = self._post(url, headers, json=body)
            if resp.status_code != 200:
                raise http_error("Google", resp, audio_path)
            op = resp.json()
            deadline = time.monotonic() + self.max_wait_s
            while not op.get("done"):
                if time.monotonic() > deadline:
                    raise TranscriptionError(f"Google batch {op.get('name')} timed out")
                self._sleep(self.poll_interval_s)
                try:
                    r = self.http.get(f"{self.endpoint()}/v2/{op['name']}", headers=headers,
                                      timeout=self.timeout)
                except httpx.HTTPError as exc:
                    raise TranscriptionError(f"Google poll failed: {exc}") from exc
                if r.status_code != 200:
                    raise http_error("Google", r, audio_path)
                op = r.json()
            if "error" in op:
                raise TranscriptionError(f"Google batch error: {op['error']}")
            file_res = (op.get("response") or {}).get("results", {}).get(uri) or {}
            if file_res.get("error"):
                raise TranscriptionError(f"Google batch file error: {file_res['error']}")
            inline = file_res.get("inlineResult") or {}
            results = (inline.get("transcript") or file_res.get("transcript") or {}).get("results")
            payload = {"operation": op.get("name"), "results": results,
                       "metadata": file_res.get("metadata"),
                       "_logatra_model": self.model, "_logatra_config": config,
                       "_logatra_batch": True}
            return Transcription(text=self._join(results), raw=payload)
        finally:
            try:
                self.http.delete(
                    f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/{quote(obj, safe='')}",
                    headers=headers, timeout=30.0)
            except httpx.HTTPError:
                pass
