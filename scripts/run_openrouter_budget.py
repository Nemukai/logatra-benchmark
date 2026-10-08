"""Run OpenRouter providers under a hard dollar budget.

Processes providers in priority order over one or more manifests, transcribing
concurrently in small batches. After every batch it sums the actual cost of
NEW responses (OpenRouter returns `usage.cost` per call) and stops everything
the moment the budget is reached. Cached samples are skipped and never
re-billed.

Usage:
    uv run python scripts/run_openrouter_budget.py \
        --budget-usd 5.0 --providers gemini,voxtral,gpt-audio \
        --results-dir results \
        data/manifests/m1_public.jsonl data/manifests/heldout_yt.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logatra_benchmark.adapters import NotConfigured  # noqa: E402
from logatra_benchmark.manifest import load_manifest  # noqa: E402
from logatra_benchmark.runner import _cache_path, transcribe_sample  # noqa: E402

BATCH = 24


def record_cost(record: dict | None) -> float:
    if not record or "error" in record:
        return 0.0
    return float(((record.get("raw") or {}).get("usage") or {}).get("cost", 0.0))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("manifests", nargs="+")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--providers", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    spent = 0.0
    done_counts: dict[str, int] = {}
    err_counts: dict[str, int] = {}
    stopped = False

    for provider in providers:
        if stopped:
            break
        done = errs = 0
        for mpath in args.manifests:
            if stopped:
                break
            mpath = Path(mpath)
            samples = [
                s for s in load_manifest(mpath)
                if not _cache_path(results_dir, provider, s.id).exists()
            ]
            print(f"[{provider}] {mpath.name}: {len(samples)} uncached", flush=True)
            for i in range(0, len(samples), BATCH):
                batch = samples[i : i + BATCH]
                try:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        records = list(
                            pool.map(
                                lambda s: transcribe_sample(
                                    provider, s, mpath.parent, results_dir
                                ),
                                batch,
                            )
                        )
                except NotConfigured as exc:
                    print(f"[{provider}] not configured: {exc}", flush=True)
                    records = []
                    break
                batch_cost = sum(record_cost(r) for r in records)
                spent += batch_cost
                done += sum(1 for r in records if r and "error" not in r)
                errs += sum(1 for r in records if r and "error" in r)
                print(
                    f"[{provider}] +{len(batch)} samples, batch ${batch_cost:.3f}, "
                    f"total ${spent:.3f} / ${args.budget_usd:.2f}",
                    flush=True,
                )
                if spent >= args.budget_usd:
                    print(f"BUDGET REACHED at ${spent:.3f}; stopping.", flush=True)
                    stopped = True
                    break
        done_counts[provider] = done
        err_counts[provider] = errs

    print(json.dumps({
        "spent_usd": round(spent, 3),
        "budget_usd": args.budget_usd,
        "stopped_early": stopped,
        "completed": done_counts,
        "errors": err_counts,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
