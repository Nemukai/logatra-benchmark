"""Benchmark runner: providers x manifest -> cached responses -> score table.

Raw provider responses are cached under results/<provider>/<sample_id>.json so
that (a) API calls are never repeated for a sample already transcribed, and
(b) scoring is reproducible offline (`--score-only`).
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adapters import NotConfigured, Transcription, TranscriptionError, get_adapter
from .manifest import Sample, load_manifest
from .metrics import score_asr


# Transient HTTP statuses worth retrying with exponential backoff.
_RETRYABLE = ("429", "500", "502", "503", "504")
_BACKOFF_S = (2.0, 8.0, 20.0)


def _transcribe_with_backoff(adapter, audio: Path, language: str | None) -> Transcription:
    """Retry transient provider failures (rate limits, 5xx) with backoff."""
    last: TranscriptionError | None = None
    for attempt, delay in enumerate((0.0, *_BACKOFF_S)):
        if delay:
            time.sleep(delay)
        try:
            return adapter.transcribe(audio, language=language)
        except TranscriptionError as exc:
            msg = str(exc)
            if not any(f"HTTP {code}" in msg for code in _RETRYABLE):
                raise
            last = exc
    raise last  # type: ignore[misc]


def _cache_path(results_dir: Path, provider: str, sample_id: str) -> Path:
    # Sample ids are file-system friendly by convention; guard the worst cases.
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    return results_dir / provider / f"{safe_id}.json"


def _load_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _store(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe_sample(
    provider: str,
    sample: Sample,
    manifest_dir: Path,
    results_dir: Path,
    score_only: bool = False,
) -> dict | None:
    """Return a cached-or-fresh result record for (provider, sample), or None.

    Record shape:
        {sample_id, provider, text, error?, elapsed_s, raw}
    """
    cache = _cache_path(results_dir, provider, sample.id)
    cached = _load_cached(cache)
    if cached is not None:
        return cached
    if score_only:
        return None

    adapter = get_adapter(provider)
    audio = sample.resolve_audio_path(manifest_dir)
    started = time.monotonic()
    try:
        result: Transcription = _transcribe_with_backoff(adapter, audio, sample.language)
    except NotConfigured:
        raise  # caller decides how to report unconfigured providers
    except (TranscriptionError, NotImplementedError, FileNotFoundError) as exc:
        record = {
            "sample_id": sample.id,
            "provider": provider,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        # Do NOT cache errors: a retry after fixing the issue should re-call.
        return record

    record = {
        "sample_id": sample.id,
        "provider": provider,
        "text": result.text,
        "words": [dataclasses.asdict(w) for w in result.words] if result.words else None,
        "speaker_turns": (
            [dataclasses.asdict(t) for t in result.speaker_turns]
            if result.speaker_turns
            else None
        ),
        "elapsed_s": round(time.monotonic() - started, 3),
        "raw": result.raw,
    }
    _store(cache, record)
    return record


def run_benchmark(
    manifest_path: str | Path,
    providers: list[str],
    results_dir: str | Path = "results",
    score_only: bool = False,
    split: str | None = None,
    workers: int = 1,
) -> dict:
    """Run providers over a manifest and aggregate scores.

    `workers` > 1 transcribes samples concurrently within each provider
    (cache files are per-sample, so concurrent writes never collide).

    Returns {"rows": [...per provider/split aggregate...], "skipped": {...}}.
    """
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    samples = load_manifest(manifest_path)
    if split:
        samples = [s for s in samples if s.split == split]

    # provider -> split -> list of (wer, cer, ref_words)
    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped: dict[str, str] = {}
    errors: dict[str, int] = defaultdict(int)

    for provider in providers:
        # Probe with the first sample serially so a missing key skips the
        # provider before we spin up a pool.
        try:
            records: list[dict | None] = []
            if samples:
                records.append(
                    transcribe_sample(
                        provider, samples[0], manifest_path.parent, results_dir, score_only
                    )
                )
                rest = samples[1:]
                if workers > 1 and rest:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        records.extend(
                            pool.map(
                                lambda s: transcribe_sample(
                                    provider, s, manifest_path.parent, results_dir, score_only
                                ),
                                rest,
                            )
                        )
                else:
                    records.extend(
                        transcribe_sample(
                            provider, s, manifest_path.parent, results_dir, score_only
                        )
                        for s in rest
                    )
        except NotConfigured as exc:
            skipped[provider] = str(exc)
            continue

        for sample, record in zip(samples, records):
            if record is None:
                continue  # score-only and no cache
            if "error" in record:
                errors[provider] += 1
                continue
            if not sample.reference_text:
                continue  # heldout audio awaiting human transcription
            scores = score_asr(sample.reference_text, record["text"])
            agg[provider][sample.split].append(
                (scores.wer, scores.cer, scores.num_reference_words)
            )

    rows = []
    for provider in providers:
        if provider in skipped:
            continue
        for split_name, triples in sorted(agg[provider].items()):
            if not triples:
                continue
            total_words = sum(t[2] for t in triples) or 1
            # Word-weighted aggregate (equivalent to corpus-level WER over concatenation)
            wer = sum(t[0] * t[2] for t in triples) / total_words
            cer = sum(t[1] * t[2] for t in triples) / total_words
            rows.append(
                {
                    "provider": provider,
                    "split": split_name,
                    "samples": len(triples),
                    "wer": round(wer, 4),
                    "cer": round(cer, 4),
                    "errors": errors.get(provider, 0),
                }
            )
    return {"rows": rows, "skipped": skipped, "errors": dict(errors)}


def render_table(report: dict) -> str:
    """Render the score table with rich if available, else plain text."""
    rows = report["rows"]
    try:
        from io import StringIO

        from rich.console import Console
        from rich.table import Table

        table = Table(title="Koe Benchmark")
        for col in ("Provider", "Split", "Samples", "WER", "CER", "Errors"):
            table.add_column(col)
        for r in rows:
            table.add_row(
                r["provider"],
                r["split"],
                str(r["samples"]),
                f"{r['wer']:.2%}",
                f"{r['cer']:.2%}",
                str(r["errors"]),
            )
        buf = StringIO()
        Console(file=buf, width=100).print(table)
        out = buf.getvalue()
    except ImportError:
        header = f"{'provider':<12} {'split':<8} {'n':>5} {'WER':>8} {'CER':>8} {'err':>4}"
        lines = [header, "-" * len(header)]
        for r in rows:
            lines.append(
                f"{r['provider']:<12} {r['split']:<8} {r['samples']:>5} "
                f"{r['wer']:>8.2%} {r['cer']:>8.2%} {r['errors']:>4}"
            )
        out = "\n".join(lines) + "\n"

    for provider, reason in report["skipped"].items():
        out += f"\nskipped {provider}: {reason}\n"
    return out
