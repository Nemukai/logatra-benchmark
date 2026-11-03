"""CLI entry point.

    python -m logatra_benchmark estimate --manifest M [--manifest M2] --providers a,b
    python -m logatra_benchmark run --manifest M --providers a,b --dry-run
    python -m logatra_benchmark run --manifest M --providers a,b --yes-spend --max-usd 20
    python -m logatra_benchmark run --manifest M --providers a,b --score-only
    python -m logatra_benchmark prices
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters import PROVIDERS
from .manifest import VALID_SPLITS
from .runner import default_results_dir, plan, render_plan, render_table, run_benchmark


def _providers(arg: str) -> list[str]:
    names = [p.strip() for p in arg.split(",") if p.strip()]
    unknown = [n for n in names if n not in PROVIDERS]
    if unknown:
        raise SystemExit(f"unknown provider(s): {', '.join(unknown)}; known: {', '.join(PROVIDERS)}")
    return names


def _add_common(p: argparse.ArgumentParser, multi_manifest: bool) -> None:
    if multi_manifest:
        p.add_argument("--manifest", required=True, action="append",
                       help="path to JSONL manifest (repeatable), e.g. "
                            "data/manifests/m1_public.jsonl or data/private/manifests/*.jsonl")
    else:
        p.add_argument("--manifest", required=True, help="path to JSONL manifest")
    p.add_argument("--providers", required=True,
                   help=f"comma-separated provider names ({', '.join(PROVIDERS)})")
    p.add_argument("--results-dir", default=None,
                   help="directory for cached raw responses (default: results/, or "
                        "data/private/results/ for manifests under data/private/ or "
                        "--split private)")
    p.add_argument("--split", choices=VALID_SPLITS, default=None,
                   help="restrict to one split (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="only consider the first N samples of the manifest")
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent requests per provider (default: 1; see docs/PROVIDERS.md "
                        "for per-provider rate limits)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logatra_benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run providers over a manifest and print scores")
    _add_common(run, multi_manifest=False)
    run.add_argument("--score-only", action="store_true",
                     help="never call provider APIs; score from cached responses only")
    run.add_argument("--dry-run", action="store_true",
                     help="validate config and key presence, print the plan; no API calls")
    run.add_argument("--yes-spend", action="store_true",
                     help="required to send uncached samples to paid APIs "
                          "(or set LOGATRA_YES_SPEND=1)")
    run.add_argument("--max-usd", type=float, default=None,
                     help="stop before the list-price estimate of this run would exceed "
                          "this many USD (across all providers)")

    est = sub.add_parser("estimate",
                         help="audio minutes, estimated cost and wall time per provider "
                              "(no network calls)")
    _add_common(est, multi_manifest=True)
    est.add_argument("--assume-seconds", type=float, default=15.0,
                     help="duration to assume for audio files not present locally (default 15)")

    sub.add_parser("prices", help="print the dated, sourced price table")

    silver = sub.add_parser(
        "silver",
        help="generate silver references for untranscribed audio via "
        "multi-model consensus",
    )
    silver.add_argument("--manifest", required=True, help="path to JSONL manifest")
    silver.add_argument(
        "--providers",
        default="sarvam,gemini,whisper",
        help="comma-separated providers to form the consensus "
        "(default: sarvam,gemini,whisper)",
    )
    silver.add_argument(
        "--results-dir",
        default="results",
        help="shared response cache directory (default: results/)",
    )
    silver.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="max mean pairwise WER for a consensus to count as silver "
        "(default: 0.20)",
    )
    silver.add_argument(
        "--min-hypotheses",
        type=int,
        default=2,
        help="minimum usable hypotheses required per sample (default: 2)",
    )
    silver.add_argument(
        "--min-agree",
        type=int,
        default=2,
        help="hypotheses that must agree with the medoid (default: 2)",
    )
    return parser


def _print_plan(p: dict, as_json: bool) -> None:
    print(json.dumps(p, indent=2, ensure_ascii=False) if as_json else render_plan(p))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prices":
        from .pricing import price_table

        print(json.dumps(price_table(), indent=2, ensure_ascii=False))
        return 0

    if args.command in ("estimate", "run") and args.results_dir is None:
        manifests = args.manifest if isinstance(args.manifest, list) else [args.manifest]
        args.results_dir = str(default_results_dir(manifests, args.split))

    if args.command == "estimate":
        p = plan(args.manifest, _providers(args.providers), args.results_dir,
                 split=args.split, limit=args.limit, workers=args.workers,
                 assume_seconds=args.assume_seconds)
        _print_plan(p, args.json)
        return 0

    if args.command == "run":
        providers = _providers(args.providers)
        if args.dry_run:
            p = plan([args.manifest], providers, args.results_dir, split=args.split,
                     limit=args.limit, workers=args.workers)
            _print_plan(p, args.json)
            missing = {k: v["missing_env"] for k, v in p["providers"].items() if v["missing_env"]}
            if missing:
                print("dry-run: missing credentials -> " + json.dumps(missing), file=sys.stderr)
                return 3
            print("dry-run: config OK, no API calls made.", file=sys.stderr)
            return 0
        yes = args.yes_spend or os.environ.get("LOGATRA_YES_SPEND", "") == "1"
        if not args.score_only and not yes:
            p = plan([args.manifest], providers, args.results_dir, split=args.split,
                     limit=args.limit, workers=args.workers)
            to_call = sum(v["to_call"] for v in p["providers"].values())
            if to_call:
                print(render_plan(p), file=sys.stderr)
                print(f"refusing to call paid APIs for {to_call} uncached samples without "
                      "--yes-spend (or LOGATRA_YES_SPEND=1). Use --dry-run to check config "
                      "or --score-only to score the cache.", file=sys.stderr)
                return 2
        report = run_benchmark(
            manifest_path=Path(args.manifest),
            providers=providers,
            results_dir=Path(args.results_dir),
            score_only=args.score_only,
            split=args.split,
            workers=args.workers,
            limit=args.limit,
            max_usd=args.max_usd,
            log=lambda m: print(m, file=sys.stderr),
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(render_table(report))
        return 0

    if args.command == "silver":
        from .silver import generate_silver

        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        summary = generate_silver(
            manifest_path=Path(args.manifest),
            providers=providers,
            results_dir=Path(args.results_dir),
            threshold=args.threshold,
            min_hypotheses=args.min_hypotheses,
            min_agree=args.min_agree,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
