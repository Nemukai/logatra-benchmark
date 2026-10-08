"""Aggregate cached benchmark results into report/data.json.

For every provider x sample with a cached response and a non-empty reference:
  - strict WER / CER (Indic-normalized)
  - script-agnostic WER (romanized comparison; fair to code-mix script choice)
  - latency (elapsed_s from the cache record) and RTF (latency / audio seconds)
  - cost: OpenRouter providers report usage.cost per call; Sarvam priced at
    Rs 0.75/min (Rs 83.5/USD), Whisper at $0.006/min.

Aggregates: provider overall, provider x dataset, provider x language.
Held-out rows score against silver references (consensus of sarvam/gemini/
whisper) and are flagged: contributors' held-out scores are upper bounds.

Usage:
    uv run python scripts/score_report.py --out report/data.json \
        data/manifests/full_public.jsonl data/manifests/heldout_yt.silver.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer  # noqa: E402

from logatra_benchmark.manifest import load_manifest  # noqa: E402
from logatra_benchmark.metrics import normalize_text, romanize, score_asr  # noqa: E402
from logatra_benchmark.runner import _cache_path  # noqa: E402

PROVIDERS = ("sarvam", "sarvam-saaras", "sarvam-saaras-v4", "whisper", "gemini", "voxtral", "gpt-audio")
INR_PER_USD = 83.5
FLAT_RATES_USD_PER_MIN = {"sarvam": 0.75 / INR_PER_USD,
                          "sarvam-saaras": 0.75 / INR_PER_USD,
                          "sarvam-saaras-v4": 0.75 / INR_PER_USD,
                          "whisper": 0.006}
SILVER_CONTRIBUTORS = {"sarvam", "gemini", "whisper"}
# Which provider's verbatim output became the reference, for silver and
# adjudicated rows: that provider scores ~0 on the row by construction.
_REF_PROVIDER_RE = re.compile(r":(?:silver|adjudicated)\((\w+)")


def wer_script_agnostic(ref: str, hyp: str) -> float:
    r, h = romanize(ref), romanize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(r, h)


def dataset_of(source: str) -> str:
    s = source.split(":")[0].split("/")[0]
    return {"gramvaani_telephony": "gramvaani", "mucs2021": "mucs_hinglish",
            "youtube": "heldout_yt"}.get(s, s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("manifests", nargs="+")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="report/data.json")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    rows: list[dict] = []
    samples_seen: set[str] = set()

    for mpath in args.manifests:
        mpath = Path(mpath)
        for s in load_manifest(mpath):
            if s.id in samples_seen or not s.reference_text:
                continue
            samples_seen.add(s.id)
            audio = s.resolve_audio_path(mpath.parent)
            try:
                audio_s = os.path.getsize(audio) / 32000
            except OSError:
                continue
            is_silver = ":silver(" in s.source or ":adjudicated(" in s.source
            ref_match = _REF_PROVIDER_RE.search(s.source)
            ref_provider = ref_match.group(1) if ref_match else None
            for prov in PROVIDERS:
                cache = _cache_path(results_dir, prov, s.id)
                if not cache.exists():
                    continue
                rec = json.loads(cache.read_text())
                if "error" in rec or not rec.get("text"):
                    continue
                sc = score_asr(s.reference_text, rec["text"])
                latency = rec.get("elapsed_s")
                usage = (rec.get("raw") or {}).get("usage") or {}
                cost = usage.get("cost")
                if cost is None and prov in FLAT_RATES_USD_PER_MIN:
                    cost = FLAT_RATES_USD_PER_MIN[prov] * audio_s / 60
                rows.append({
                    "sample_id": s.id,
                    "provider": prov,
                    "dataset": dataset_of(s.source),
                    "language": s.language,
                    "code_mixed": s.code_mixed,
                    "split": s.split,
                    "silver_ref": is_silver,
                    "self_ref": prov == ref_provider,
                    "wer": sc.wer,
                    "cer": sc.cer,
                    "wer_sa": wer_script_agnostic(s.reference_text, rec["text"]),
                    "ref_words": sc.num_reference_words,
                    "latency_s": latency,
                    "audio_s": round(audio_s, 2),
                    "rtf": round(latency / audio_s, 4) if latency and audio_s else None,
                    "cost_usd": cost,
                })

    def aggregate(keyfn):
        agg = defaultdict(lambda: {"wer_n": 0.0, "cer_n": 0.0, "sa_n": 0.0,
                                   "w": 0, "n": 0, "wers": [], "lat": [], "rtf": [],
                                   "cost": 0.0, "audio_s": 0.0})
        for r in rows:
            a = agg[keyfn(r)]
            w = max(r["ref_words"], 1)
            a["wer_n"] += r["wer"] * w
            a["cer_n"] += r["cer"] * w
            a["sa_n"] += r["wer_sa"] * w
            a["w"] += w
            a["n"] += 1
            a["wers"].append(r["wer"])
            a["self"] = a.get("self", 0) + (1 if r.get("self_ref") else 0)
            if r["latency_s"] is not None:
                a["lat"].append(r["latency_s"])
            if r["rtf"] is not None:
                a["rtf"].append(r["rtf"])
            if r["cost_usd"]:
                a["cost"] += r["cost_usd"]
            a["audio_s"] += r["audio_s"]
        out = {}
        for k, a in agg.items():
            lat = sorted(a["lat"])
            out["|".join(k) if isinstance(k, tuple) else k] = {
                "samples": a["n"],
                "wer": round(a["wer_n"] / a["w"], 4),
                "cer": round(a["cer_n"] / a["w"], 4),
                "wer_script_agnostic": round(a["sa_n"] / a["w"], 4),
                # Median per-sample WER: robust to repetition-loop outliers,
                # so a provider that loops occasionally is visible as
                # mean >> median rather than just an absurd mean.
                "wer_median": round(sorted(a["wers"])[a["n"] // 2], 4),
                # Share of rows whose reference is this provider's own output
                # (silver/adjudicated rows only): those rows score ~0 by
                # construction, so a high rate means the aggregate flatters.
                "self_ref_rate": round(a.get("self", 0) / a["n"], 4),
                "latency_median_s": round(lat[len(lat) // 2], 3) if lat else None,
                "latency_p90_s": round(lat[int(len(lat) * 0.9)], 3) if lat else None,
                "rtf_median": round(sorted(a["rtf"])[len(a["rtf"]) // 2], 4) if a["rtf"] else None,
                "cost_usd_total": round(a["cost"], 4),
                "cost_usd_per_hour": round(a["cost"] / (a["audio_s"] / 3600), 4) if a["audio_s"] else None,
                "audio_minutes": round(a["audio_s"] / 60, 1),
            }
        return out

    # Comparable subset: per split, only samples that EVERY present provider
    # completed, so the headline table never compares apples to oranges.
    present = sorted({r["provider"] for r in rows})
    comparable_ids: dict[str, set] = {}
    for split in ("public", "heldout"):
        per_prov = [
            {r["sample_id"] for r in rows if r["provider"] == p and r["split"] == split}
            for p in present
        ]
        per_prov = [s for s in per_prov if s]
        comparable_ids[split] = set.intersection(*per_prov) if len(per_prov) == len(present) else set()
    rows_all = rows
    rows = [r for r in rows_all if r["sample_id"] in comparable_ids[r["split"]]]
    by_provider_comparable = aggregate(lambda r: (r["provider"], r["split"]))
    rows = rows_all

    report = {
        "by_provider": aggregate(lambda r: (r["provider"], r["split"])),
        "by_provider_comparable": by_provider_comparable,
        "comparable_counts": {k: len(v) for k, v in comparable_ids.items()},
        "by_provider_dataset": aggregate(lambda r: (r["provider"], r["dataset"])),
        "by_provider_language": aggregate(lambda r: (r["provider"], r["language"])),
        "silver_caveat_providers": sorted(SILVER_CONTRIBUTORS),
        "row_count": len(rows),
        "sample_count": len(samples_seen),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    # Per-sample rows for the showcase page (side-by-side comparisons).
    (out_path.parent / "rows.json").write_text(json.dumps(rows))
    print(f"wrote {out_path} ({len(rows)} scored rows, "
          f"{len(samples_seen)} samples)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
