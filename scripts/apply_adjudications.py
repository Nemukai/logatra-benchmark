"""Apply adjudication decisions to the held-out silver manifest.

Reads the review queue (hypotheses per disputed sample) and the adjudication
decisions file, fills in the chosen hypothesis as the reference for every
accepted row, and rewrites the silver manifest in place. Rows adjudicated here
are tagged `:adjudicated(<provider>)` in `source` so they can never be
confused with threshold-accepted silver rows (`:silver(...)`) or human gold.

Excluded rows (decision != "accept") keep an empty reference and simply never
score. The decisions file carries a per-row rationale; see its `_method` key
for the adjudication protocol.

Usage:
    uv run python scripts/apply_adjudications.py \
        [--manifest data/manifests/heldout_yt.silver.jsonl] \
        [--review data/manifests/heldout_yt.review.jsonl] \
        [--decisions data/manifests/heldout_yt.adjudications.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koe_benchmark.manifest import load_manifest, write_manifest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--manifest", default="data/manifests/heldout_yt.silver.jsonl")
    p.add_argument("--review", default="data/manifests/heldout_yt.review.jsonl")
    p.add_argument("--decisions", default="data/manifests/heldout_yt.adjudications.json")
    args = p.parse_args()

    hypotheses = {
        row["sample_id"]: row["hypotheses"]
        for row in map(json.loads, Path(args.review).read_text().splitlines())
    }
    decisions = json.loads(Path(args.decisions).read_text())["decisions"]

    unknown = set(decisions) - set(hypotheses)
    if unknown:
        raise SystemExit(f"decisions for samples not in review queue: {sorted(unknown)}")

    samples = load_manifest(args.manifest)
    applied = excluded = 0
    out = []
    for s in samples:
        d = decisions.get(s.id)
        if d is None or ":adjudicated(" in s.source:
            out.append(s)
            continue
        if s.reference_text:
            raise SystemExit(f"{s.id}: has a reference already but is in the decisions file")
        if d["decision"] != "accept":
            excluded += 1
            out.append(s)
            continue
        text = hypotheses[s.id].get(d["provider"])
        if not text:
            raise SystemExit(f"{s.id}: no {d['provider']} hypothesis in review queue")
        out.append(
            replace(
                s,
                reference_text=text.strip(),
                source=f"{s.source}:adjudicated({d['provider']})",
            )
        )
        applied += 1

    write_manifest(out, args.manifest)
    with_ref = sum(1 for s in out if s.reference_text)
    print(
        f"applied {applied}, excluded {excluded}; "
        f"{with_ref}/{len(out)} rows now have references -> {args.manifest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
