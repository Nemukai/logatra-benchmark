"""Build the self-contained HTML showcase from report/data.json + rows.json.

Embeds all data inline (no fetch, works from file://). Side-by-side transcript
comparisons are read from the results/ cache for the most provider-divergent
samples plus hand-picked dataset representatives.

Usage: uv run python scripts/build_report.py [--out report/index.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koe_benchmark.manifest import load_manifest  # noqa: E402
from koe_benchmark.runner import _cache_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("sarvam", "whisper", "gemini", "voxtral", "gpt-audio")


def pick_showcase(rows: list[dict], results_dir: Path, manifests: list[Path], k: int = 6) -> list[dict]:
    """Pick samples where providers diverge most, with dataset diversity."""
    by_sample: dict[str, list[dict]] = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], []).append(r)

    refs: dict[str, dict] = {}
    for mp in manifests:
        for s in load_manifest(mp):
            if s.reference_text:
                refs[s.id] = {"ref": s.reference_text, "lang": s.language,
                              "dataset": None, "source": s.source}

    scored = []
    for sid, rs in by_sample.items():
        if len(rs) < 3 or sid not in refs:
            continue
        wers = [r["wer_sa"] for r in rs]
        spread = max(wers) - min(wers)
        scored.append((spread, sid, rs[0]["dataset"]))
    scored.sort(reverse=True)

    picked, seen_ds = [], set()
    for spread, sid, ds in scored:
        if ds in seen_ds and len(seen_ds) < 4:
            continue
        seen_ds.add(ds)
        picked.append(sid)
        if len(picked) >= k:
            break
    for spread, sid, ds in scored:  # fill remainder
        if sid not in picked:
            picked.append(sid)
        if len(picked) >= k:
            break

    cases = []
    for sid in picked[:k]:
        hyps = {}
        for prov in PROVIDERS:
            c = _cache_path(results_dir, prov, sid)
            if c.exists():
                rec = json.loads(c.read_text())
                if rec.get("text"):
                    row = next((r for r in by_sample[sid] if r["provider"] == prov), None)
                    hyps[prov] = {"text": rec["text"],
                                  "wer": row["wer"] if row else None,
                                  "wer_sa": row["wer_sa"] if row else None}
        cases.append({"sample_id": sid, "reference": refs[sid]["ref"],
                      "language": refs[sid]["lang"],
                      "dataset": by_sample[sid][0]["dataset"],
                      "source": refs[sid]["source"], "hypotheses": hyps})
    return cases


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/index.html")
    args = p.parse_args()

    data = json.loads((ROOT / "report" / "data.json").read_text())
    rows = json.loads((ROOT / "report" / "rows.json").read_text())
    manifests = [ROOT / "data/manifests/full_public.jsonl",
                 ROOT / "data/manifests/heldout_yt.silver.jsonl"]
    cases = pick_showcase(rows, ROOT / "results", manifests)

    html = (ROOT / "scripts" / "report_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    html = html.replace("/*__CASES__*/null", json.dumps(cases, ensure_ascii=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} kB, "
          f"{len(cases)} showcase cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
