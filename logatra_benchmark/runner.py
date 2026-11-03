"""Benchmark runner: providers x manifest -> cached responses -> score table.

Raw provider responses are cached under results/<provider>/<sample_id>.json so
that (a) API calls are never repeated for a sample already transcribed, and
(b) scoring is reproducible offline (`--score-only`).

Cache record shape (success):
    {sample_id, provider, text, words, speaker_turns, elapsed_s, raw,
     language, code_mixed, audio_s, cost_usd, model}
Errors are returned as {sample_id, provider, error, elapsed_s} and never
cached (a re-run re-calls). Samples a provider cannot run by design
(unsupported language, clip too long for the endpoint) come back as
{sample_id, provider, skipped: <reason>}: not cached, not errors, not billed.

Spend controls (see `plan`, `Budget`):
    * cached samples are never re-sent;
    * `max_usd` reserves each call's list-price estimate before sending and
      stops the run before the estimate would exceed the cap;
    * `limit` caps the number of manifest samples considered.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import pricing
from .adapters import NotConfigured, Transcription, TranscriptionError, Unsupported, get_adapter
from .adapters.base import adapter_supports, audio_duration_s, missing_env
from .manifest import Sample, load_manifest
from .metrics import score_asr

# Transient HTTP statuses worth retrying with exponential backoff.
_RETRYABLE = ("429", "500", "502", "503", "504")
_BACKOFF_S = (2.0, 8.0, 20.0)

# Per-provider single-request duration limits that change the call path
# (used by `plan` to flag long clips; adapters enforce them).
SYNC_LIMIT_S = {"sarvam": 30.0, "sarvam-saaras": 30.0, "sarvam-saaras-v4": 30.0,
                "google": 60.0}


def _is_retryable(msg: str) -> bool:
    return any(f"HTTP {code}" in msg for code in _RETRYABLE) or "request failed" in msg


def _call_adapter(adapter, audio: Path, sample: Sample) -> Transcription:
    if getattr(adapter, "wants_code_mixed", False):
        return adapter.transcribe(audio, language=sample.language,
                                  code_mixed=bool(sample.code_mixed))
    return adapter.transcribe(audio, language=sample.language)


def _transcribe_with_backoff(adapter, audio: Path, sample_or_language,
                             backoff: tuple[float, ...] = _BACKOFF_S,
                             sleep=time.sleep) -> Transcription:
    """Retry transient provider failures (rate limits, 5xx, network) with backoff."""
    sample = sample_or_language
    if not isinstance(sample, Sample):  # back-compat: (adapter, audio, language)
        sample = Sample(id="_", audio_path=str(audio), reference_text="",
                        language=sample_or_language or "")
    last: TranscriptionError | None = None
    for delay in (0.0, *backoff):
        if delay:
            sleep(delay)
        try:
            return _call_adapter(adapter, audio, sample)
        except TranscriptionError as exc:
            if not _is_retryable(str(exc)):
                raise
            last = exc
    raise last  # type: ignore[misc]


def _cache_path(results_dir: Path, provider: str, sample_id: str) -> Path:
    # Sample ids are file-system friendly by convention; guard the worst cases.
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    return Path(results_dir) / provider / f"{safe_id}.json"


def _load_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _store(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class Budget:
    """Thread-safe spend cap. Reserve an estimate before a call, settle after.

    Once a reservation is refused the budget is closed: no further calls.
    """

    def __init__(self, max_usd: float | None) -> None:
        self.max_usd = max_usd
        self.spent = 0.0
        self.reserved = 0.0
        self.closed = False
        self._lock = threading.Lock()

    def try_reserve(self, amount: float) -> bool:
        with self._lock:
            if self.closed:
                return False
            if self.max_usd is not None and self.spent + self.reserved + amount > self.max_usd + 1e-12:
                self.closed = True
                return False
            self.reserved += amount
            return True

    def release(self, amount: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - amount)

    def settle(self, reserved: float, actual: float | None) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += reserved if actual is None else actual


def _skip(provider: str, sample: Sample, reason: str, **extra) -> dict:
    return {"sample_id": sample.id, "provider": provider, "skipped": reason, **extra}


def transcribe_sample(
    provider: str,
    sample: Sample,
    manifest_dir: Path,
    results_dir: Path,
    score_only: bool = False,
    *,
    adapter=None,
    budget: Budget | None = None,
    backoff: tuple[float, ...] = _BACKOFF_S,
    sleep=time.sleep,
) -> dict | None:
    """Return a cached-or-fresh result record for (provider, sample), or None.

    Record shape:
        {sample_id, provider, text, error?, skipped?, elapsed_s, raw, ...}
    """
    cache = _cache_path(results_dir, provider, sample.id)
    cached = _load_cached(cache)
    if cached is not None:
        return cached
    if score_only:
        return None

    adapter = adapter if adapter is not None else get_adapter(provider)
    reason = adapter_supports(adapter, sample.language, bool(sample.code_mixed))
    if reason:
        return _skip(provider, sample, reason)

    audio = sample.resolve_audio_path(Path(manifest_dir))
    try:
        audio_s = audio_duration_s(audio)
    except FileNotFoundError as exc:
        return {"sample_id": sample.id, "provider": provider,
                "error": f"FileNotFoundError: {exc}", "elapsed_s": 0.0}

    estimate = pricing.estimate_cost_usd(provider, audio_s, sample.language,
                                         bool(sample.code_mixed), for_budget=True) or 0.0
    if budget is not None and not budget.try_reserve(estimate):
        return _skip(provider, sample, f"budget cap ${budget.max_usd:.2f} reached", budget_stop=True)

    started = time.monotonic()
    try:
        result: Transcription = _transcribe_with_backoff(adapter, audio, sample, backoff, sleep)
    except NotConfigured:
        if budget is not None:
            budget.release(estimate)
        raise  # caller decides how to report unconfigured providers
    except Unsupported as exc:
        if budget is not None:
            budget.release(estimate)
        return _skip(provider, sample, str(exc))
    except (TranscriptionError, NotImplementedError, FileNotFoundError) as exc:
        if budget is not None:
            budget.release(estimate)
        # Do NOT cache errors: a retry after fixing the issue should re-call.
        return {
            "sample_id": sample.id,
            "provider": provider,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }

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
        "language": sample.language,
        "code_mixed": bool(sample.code_mixed),
        "audio_s": round(audio_s, 3),
        "model": getattr(adapter, "model", None),
    }
    cost = pricing.record_cost_usd(provider, record, audio_s)
    record["cost_usd"] = None if cost is None else round(cost, 6)
    if budget is not None:
        budget.settle(estimate, cost)
    _store(cache, record)
    return record


PRIVATE_RESULTS_DIR_NAME = "results"


def _private_dir() -> Path | None:
    try:
        from .manifest import PRIVATE_DIR  # added with the "private" split
    except ImportError:  # pragma: no cover - older manifest.py
        return None
    return Path(PRIVATE_DIR)


def _is_private_path(path: str | Path) -> bool:
    priv = _private_dir()
    if priv is None:
        return False
    try:
        Path(path).resolve().relative_to(priv.resolve())
    except ValueError:
        return False
    return True


def default_results_dir(manifest_paths: list[str | Path], split: str | None = None) -> Path:
    """results/ normally; data/private/results/ for private manifests/splits.

    Private-split audio must never leave data/private/ (gitignored, never
    published), and that includes the provider transcripts we cache.
    """
    priv = _private_dir()
    if priv is not None and (split == "private" or any(_is_private_path(m) for m in manifest_paths)):
        return priv / PRIVATE_RESULTS_DIR_NAME
    return Path("results")


def _guard_private(samples: list[Sample], results_dir: Path) -> None:
    if any(s.split == "private" for s in samples) and not _is_private_path(results_dir):
        raise ValueError(
            f"refusing to cache private-split transcripts under {results_dir}: use a results "
            "dir inside data/private/ (the CLI defaults to data/private/results for private "
            "manifests)"
        )


def select_samples(manifest_path: Path, split: str | None = None,
                   limit: int | None = None) -> list[Sample]:
    samples = load_manifest(manifest_path)
    if split:
        samples = [s for s in samples if s.split == split]
    if limit is not None:
        samples = samples[: max(limit, 0)]
    return samples


def run_benchmark(
    manifest_path: str | Path,
    providers: list[str],
    results_dir: str | Path = "results",
    score_only: bool = False,
    split: str | None = None,
    workers: int = 1,
    *,
    limit: int | None = None,
    max_usd: float | None = None,
    adapters: dict | None = None,
    backoff: tuple[float, ...] = _BACKOFF_S,
    sleep=time.sleep,
    log=None,
) -> dict:
    """Run providers over a manifest and aggregate scores.

    `workers` > 1 transcribes samples concurrently within each provider
    (cache files are per-sample, so concurrent writes never collide).
    `max_usd` is a global cap across all providers in this run.

    Returns {"rows", "skipped", "errors", "unsupported", "budget"}.
    """
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    samples = select_samples(manifest_path, split, limit)
    if not score_only:
        _guard_private(samples, results_dir)
    budget = Budget(max_usd)

    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    skipped: dict[str, str] = {}
    errors: dict[str, int] = defaultdict(int)
    unsupported: dict[str, Counter] = defaultdict(Counter)
    budget_stopped: dict[str, int] = defaultdict(int)

    for provider in providers:
        adapter = (adapters or {}).get(provider)
        if adapter is None and not score_only:
            adapter = get_adapter(provider)
        if adapter is not None and not score_only:
            missing = missing_env(adapter)
            if missing:
                skipped[provider] = f"missing env: {', '.join(missing)}"
                continue

        def one(s: Sample, _p=provider, _a=adapter):
            return transcribe_sample(_p, s, manifest_path.parent, results_dir, score_only,
                                     adapter=_a, budget=budget, backoff=backoff, sleep=sleep)

        try:
            if workers > 1 and len(samples) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    records = list(pool.map(one, samples))
            else:
                records = [one(s) for s in samples]
        except NotConfigured as exc:
            skipped[provider] = str(exc)
            continue

        for sample, record in zip(samples, records):
            if record is None:
                continue  # score-only and no cache
            if "skipped" in record:
                if record.get("budget_stop"):
                    budget_stopped[provider] += 1
                else:
                    unsupported[provider][record["skipped"]] += 1
                continue
            if "error" in record:
                errors[provider] += 1
                if log:
                    log(f"[{provider}] {sample.id}: {record['error'][:200]}")
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
    return {
        "rows": rows,
        "skipped": skipped,
        "errors": dict(errors),
        "unsupported": {p: dict(c) for p, c in unsupported.items()},
        "budget": {
            "max_usd": max_usd,
            "spent_usd": round(budget.spent, 4),
            "stopped": budget.closed,
            "samples_not_sent": dict(budget_stopped),
        },
    }


# ---------------------------------------------------------------------------
# Offline planning: `estimate` and `run --dry-run`. No network calls.
# ---------------------------------------------------------------------------

def plan(
    manifest_paths: list[str | Path],
    providers: list[str],
    results_dir: str | Path = "results",
    split: str | None = None,
    limit: int | None = None,
    workers: int = 1,
    assume_seconds: float = 15.0,
) -> dict:
    """Per provider: what a run would send, cost and take. Pure local work."""
    results_dir = Path(results_dir)
    samples: list[tuple[Sample, Path]] = []
    for mp in manifest_paths:
        mp = Path(mp)
        samples += [(s, mp.parent) for s in select_samples(mp, split, None)]
    if limit is not None:
        samples = samples[: max(limit, 0)]

    durations: dict[str, float] = {}
    missing_audio = 0
    for s, mdir in samples:
        try:
            durations[s.id] = audio_duration_s(s.resolve_audio_path(mdir))
        except FileNotFoundError:
            durations[s.id] = assume_seconds
            missing_audio += 1

    out = {
        "samples": len(samples),
        "audio_minutes": round(sum(durations.values()) / 60.0, 2),
        "missing_audio_files": missing_audio,
        "assumed_seconds_for_missing": assume_seconds if missing_audio else None,
        "providers": {},
    }
    for provider in providers:
        adapter = get_adapter(provider)
        unsupported: Counter = Counter()
        cached = 0
        to_call: list[Sample] = []
        for s, _ in samples:
            if _cache_path(results_dir, provider, s.id).exists():
                cached += 1
                continue
            reason = adapter_supports(adapter, s.language, bool(s.code_mixed))
            if reason:
                unsupported[s.language + ("+cm" if s.code_mixed else "")] += 1
                continue
            to_call.append(s)
        secs = [durations[s.id] for s in to_call]
        costs = [pricing.estimate_cost_usd(provider, durations[s.id], s.language,
                                           bool(s.code_mixed), for_budget=False)
                 for s in to_call]
        budget_costs = [pricing.estimate_cost_usd(provider, durations[s.id], s.language,
                                                  bool(s.code_mixed), for_budget=True)
                        for s in to_call]
        list_priced = all(c is not None for c in costs)
        limit_s = SYNC_LIMIT_S.get(provider)
        long_clips = sum(1 for d in secs if limit_s is not None and d > limit_s)
        price = pricing.get_price(provider)
        out["providers"][provider] = {
            "model": getattr(adapter, "model", None),
            "cached": cached,
            "unsupported": dict(unsupported),
            "to_call": len(to_call),
            "audio_minutes_to_call": round(sum(secs) / 60.0, 2),
            "est_usd": round(sum(c for c in costs if c is not None), 4) if list_priced else None,
            "est_usd_rough": None if list_priced else round(sum(c or 0.0 for c in budget_costs), 4),
            "usd_per_min": None if price is None else price.rate(),
            "price_source": None if price is None else price.source,
            "price_checked": None if price is None else price.checked,
            "est_wall_s": round(pricing.estimate_wall_s(provider, secs, workers, limit_s), 1),
            "long_clips_over_sync_limit": long_clips,
            "missing_env": missing_env(adapter),
        }
    return out


def render_plan(p: dict) -> str:
    lines = [
        f"samples: {p['samples']}   audio: {p['audio_minutes']:.1f} min"
        + (f"   (audio missing for {p['missing_audio_files']}; assumed "
           f"{p['assumed_seconds_for_missing']}s each)" if p["missing_audio_files"] else ""),
        "",
        f"{'provider':<24}{'cached':>7}{'skip':>6}{'call':>7}{'min':>9}{'est $':>10}"
        f"{'wall':>9}  keys",
    ]
    total = 0.0
    for name, r in p["providers"].items():
        est = r["est_usd"]
        if est is None:
            est_s = f"~{r['est_usd_rough']:.2f}*"
            total += r["est_usd_rough"] or 0.0
        else:
            est_s = f"{est:.2f}"
            total += est
        wall = r["est_wall_s"]
        wall_s = f"{wall / 3600:.1f}h" if wall >= 3600 else f"{wall / 60:.0f}m"
        keys = "ok" if not r["missing_env"] else "MISSING " + ", ".join(r["missing_env"])
        lines.append(
            f"{name:<24}{r['cached']:>7}{sum(r['unsupported'].values()):>6}{r['to_call']:>7}"
            f"{r['audio_minutes_to_call']:>9.1f}{est_s:>10}{wall_s:>9}  {keys}"
        )
        if r["unsupported"]:
            lines.append(f"{'':<24}unsupported (skipped): {r['unsupported']}")
        if r["long_clips_over_sync_limit"]:
            lines.append(f"{'':<24}{r['long_clips_over_sync_limit']} clips exceed the sync "
                         "limit (batch path / skip, see docs/PROVIDERS.md)")
    lines.append("")
    lines.append(f"total estimated list-price spend: ${total:.2f}")
    lines.append("* token-billed provider: rough guess, actual cost comes from raw.usage.cost")
    lines.append(f"prices: logatra_benchmark/pricing.py (checked {pricing.CHECKED}); "
                 "wall time is a rough model, not a measurement")
    return "\n".join(lines) + "\n"


def render_table(report: dict) -> str:
    """Render the score table with rich if available, else plain text."""
    rows = report["rows"]
    try:
        from io import StringIO

        from rich.console import Console
        from rich.table import Table

        table = Table(title="Logatra Benchmark")
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
    for provider, reasons in (report.get("unsupported") or {}).items():
        for reason, n in reasons.items():
            out += f"unsupported {provider}: {n} samples ({reason})\n"
    b = report.get("budget") or {}
    if b.get("max_usd") is not None or b.get("spent_usd"):
        out += (f"\nspend this run (list-price est.): ${b.get('spent_usd', 0):.4f}"
                + (f" / cap ${b['max_usd']:.2f}" if b.get("max_usd") is not None else "")
                + (" -- BUDGET CAP REACHED, run stopped" if b.get("stopped") else "") + "\n")
    return out
