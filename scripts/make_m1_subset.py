"""Build the M1 public-split subset manifest.

Deterministically samples N rows per source manifest (seeded by sample id
hash, so re-runs pick the same rows even if manifests grow) and merges them
into one manifest with audio paths rewritten relative to the output location.

Usage:
    uv run python scripts/make_m1_subset.py \
        --out data/manifests/m1_public.jsonl \
        --per-source 75 \
        data/manifests/fleurs_*.jsonl data/manifests/mucs2021_hi_en.jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koe_benchmark.manifest import Sample, load_manifest, write_manifest  # noqa: E402


def stable_key(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:12], 16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("manifests", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--per-source", type=int, default=75)
    p.add_argument(
        "--max-dur-hint",
        type=float,
        default=30.0,
        help="(unused placeholder for future duration filtering)",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    merged: list[Sample] = []
    for mpath in args.manifests:
        mpath = Path(mpath)
        samples = load_manifest(mpath)
        picked = sorted(samples, key=lambda s: stable_key(s.id))[: args.per_source]
        for s in picked:
            audio_abs = s.resolve_audio_path(mpath.parent).resolve()
            try:
                rel = audio_abs.relative_to(out_path.parent.resolve())
                s.audio_path = str(rel)
            except ValueError:
                s.audio_path = str(audio_abs)
            merged.append(s)
        print(f"{mpath.name}: picked {len(picked)} of {len(samples)}", file=sys.stderr)

    # Guard against id collisions across source manifests.
    seen: set[str] = set()
    for s in merged:
        if s.id in seen:
            raise SystemExit(f"duplicate id across sources: {s.id}")
        seen.add(s.id)

    write_manifest(merged, out_path)
    print(f"wrote {len(merged)} samples -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
