"""STT adapter protocol and shared plumbing.

Every provider adapter implements the `STTAdapter` protocol: a `name`, and a
`transcribe(audio_path, language)` method returning a `Transcription`.

Adapters read API keys from environment variables and raise `NotConfigured`
with a helpful message when the key is missing, so the runner can skip (rather
than crash on) unconfigured providers.

Optional adapter attributes the runner understands (all have safe defaults via
`adapter_required_env`, `adapter_supports`, ...):

    required_env: tuple[str, ...]   env vars that must be set (checked by
                                    `--dry-run` without any network call)
    supports(language, code_mixed) -> str | None
                                    None if the language can be run, else a
                                    human-readable reason (sample is skipped,
                                    not counted as an error)
    wants_code_mixed: bool          if True the runner passes
                                    `code_mixed=<bool>` to `transcribe`
    model: str                      model id recorded in the cache
"""

from __future__ import annotations

import os
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx


class NotConfigured(Exception):
    """Raised when a provider's credentials are missing from the environment."""


class TranscriptionError(Exception):
    """Raised when a provider call fails (HTTP error, bad response shape, ...)."""


class Unsupported(Exception):
    """Raised when a sample cannot be run on this provider by design.

    Examples: language not in the provider's matrix, clip longer than the
    endpoint accepts and no long-audio path configured. The runner records a
    `skipped` reason and does NOT count it as an error (and does not bill).
    """


def require_env(var: str, provider: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise NotConfigured(
            f"{provider}: environment variable {var} is not set. "
            f"Export it (e.g. `export {var}=...`) and re-run."
        )
    return value


@dataclass(slots=True)
class Word:
    """One recognized word with optional timing/speaker info."""

    text: str
    start: float | None = None  # seconds
    end: float | None = None    # seconds
    speaker: str | None = None


@dataclass(slots=True)
class SpeakerTurn:
    """One diarized speaker turn, for DER scoring."""

    speaker: str
    start: float  # seconds
    end: float    # seconds


@dataclass(slots=True)
class Transcription:
    """Normalized output of any STT provider."""

    text: str
    words: list[Word] | None = None
    speaker_turns: list[SpeakerTurn] | None = None
    raw: dict = field(default_factory=dict)  # raw provider response, cached as-is


@runtime_checkable
class STTAdapter(Protocol):
    """Protocol all provider adapters implement."""

    name: str

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcription:
        """Transcribe one audio file.

        Args:
            audio_path: path to a local audio file (benchmark uses 16 kHz mono WAV).
            language: optional language hint (manifest `language` field); adapters
                map it to the provider's own language-code scheme or ignore it.

        Raises:
            NotConfigured: credentials missing.
            Unsupported: the sample cannot be run on this provider (skip).
            TranscriptionError: provider call failed.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers shared by adapters and the runner.
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_shared_client: httpx.Client | None = None


def shared_http_client() -> httpx.Client:
    """One process-wide httpx.Client (thread-safe, keeps connections alive)."""
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = httpx.Client(timeout=300.0)
        return _shared_client


class HTTPAdapter:
    """Base for HTTP adapters: injectable client so tests can mock transport."""

    name: str = "base"
    required_env: tuple[str, ...] = ()
    wants_code_mixed: bool = False

    def __init__(self, client: httpx.Client | None = None, timeout: float = 300.0) -> None:
        self._client = client
        self.timeout = timeout

    @property
    def http(self) -> httpx.Client:
        return self._client if self._client is not None else shared_http_client()

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return None


def http_error(provider: str, resp: httpx.Response, audio_path: Path | None = None) -> TranscriptionError:
    """Uniform error text; the runner retries on 'HTTP 429/5xx' in the message."""
    where = f" for {audio_path.name}" if audio_path is not None else ""
    return TranscriptionError(f"{provider} HTTP {resp.status_code}{where}: {resp.text[:500]}")


def audio_duration_s(path: Path) -> float:
    """Duration in seconds: WAV header if readable, else 16 kHz/16-bit mono size.

    Raises FileNotFoundError if the file is missing.
    """
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            if rate:
                return w.getnframes() / float(rate)
    except (wave.Error, EOFError):
        pass
    except FileNotFoundError:
        raise
    return os.path.getsize(path) / 32000.0


def adapter_required_env(adapter) -> tuple[str, ...]:
    return tuple(getattr(adapter, "required_env", ()) or ())


def adapter_supports(adapter, language: str | None, code_mixed: bool = False) -> str | None:
    fn = getattr(adapter, "supports", None)
    if fn is None:
        return None
    return fn(language, code_mixed)


def missing_env(adapter) -> list[str]:
    """Required env vars that are unset. Entries 'A|B' mean either A or B."""
    missing = []
    for spec in adapter_required_env(adapter):
        options = spec.split("|")
        if not any(os.environ.get(v, "").strip() for v in options):
            missing.append(" or ".join(options))
    return missing
