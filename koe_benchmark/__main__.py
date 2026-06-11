"""CLI entry point: python -m koe_benchmark run --manifest ... --providers sarvam,whisper"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import PROVIDERS
from .runner import render_table, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="koe_benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run providers over a manifest and print scores")
    run.add_argument("--manifest", required=True, help="path to JSONL manifest")
    run.add_argument(
        "--providers",
        required=True,
        help=f"comma-separated provider names ({', '.join(PROVIDERS)})",
    )
    run.add_argument(
        "--results-dir",
        default="results",
        help="directory for cached raw responses (default: results/)",
    )
    run.add_argument(
        "--split",
        choices=("public", "heldout"),
        default=None,
        help="restrict to one split (default: both)",
    )
    run.add_argument(
        "--score-only",
        action="store_true",
        help="never call provider APIs; score from cached responses only",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="emit the aggregate report as JSON instead of a table",
    )

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        report = run_benchmark(
            manifest_path=Path(args.manifest),
            providers=providers,
            results_dir=Path(args.results_dir),
            score_only=args.score_only,
            split=args.split,
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
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
