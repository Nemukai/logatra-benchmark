"""Aggregate cached benchmark results into report/data.json.

For every provider x sample with a cached response and a non-empty reference:
  - strict WER / CER (Indic-normalized)
  - script-agnostic WER (romanized comparison; fair to code-mix script choice)
  - latency (elapsed_s from the cache record) and RTF (latency / audio seconds)
  - cost: provider-reported usage.cost when present (OpenRouter), else the
    dated list price in logatra_benchmark/pricing.py (pricing.record_cost_usd).

Aggregates: provider overall, provider x dataset, provider x language,
provider x matrix cell (FLEURS language / dataset). Held-out rows score
against silver references (consensus of sarvam/gemini/whisper) and are
flagged: contributors' held-out scores are upper bounds, so contributors are
also scored on the rows where they are not the reference source.

Uncertainty: percentile bootstrap 95% CIs (see ``bootstrap``) on every
comparable-set provider score, every matrix cell, and every pairwise gap on
the comparable sets. Held-out resamples whole source videos (chunks of one
video are not independent); public resamples samples.

Private split: rows with ``split == "private"`` are dropped unless
``--include-private`` is given, and then output may only go to
data/private/reports/ (gitignored, never published).

Gramvaani: provider outputs for OpenSLR 118 are not redistributed, so a fresh
checkout cannot rescore them. ``--carry-forward report/rows.json`` keeps the
previously published Gramvaani rows (metrics only) for samples that have no
local reference, so the Gramvaani numbers stay in the report. Rows scored
locally always win over carried rows.

Usage:
    uv run python scripts/score_report.py --out report/data.json \
        --carry-forward report/rows.json \
        data/manifests/full_public.jsonl data/manifests/heldout_yt.silver.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logatra_benchmark.manifest import (  # noqa: E402
    PRIVATE_REPORT_DIR,
    load_manifest,
    require_private_output,
)
from logatra_benchmark.metrics import score_asr, script_agnostic_wer  # noqa: E402
from logatra_benchmark.pricing import record_cost_usd  # noqa: E402
from logatra_benchmark.runner import _cache_path  # noqa: E402

PROVIDERS = ("sarvam", "sarvam-saaras", "sarvam-saaras-v4", "whisper", "gemini", "voxtral", "gpt-audio")
SILVER_CONTRIBUTORS = {"sarvam", "gemini", "whisper"}
# Which provider's verbatim output became the reference, for silver and
# adjudicated rows: that provider scores ~0 on the row by construction.
_REF_PROVIDER_RE = re.compile(r":(?:silver|adjudicated)\((\w+)")


_HELDOUT_ID_RE = re.compile(r"^heldout_yt_(.+)_\d+$")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261103
SPLITS = ("public", "heldout")


def wer_script_agnostic(ref: str, hyp: str) -> float:
    return script_agnostic_wer(ref, hyp)


def cluster_of(sample_id: str, split: str) -> str:
    """Bootstrap resampling unit: the source video for held-out chunks
    (heldout_yt_<videoid>_NNN), the sample itself everywhere else."""
    if split == "heldout":
        m = _HELDOUT_ID_RE.match(sample_id)
        if m:
            return m.group(1)
    return sample_id


def _ci(values: list[float]) -> list[float]:
    v = sorted(values)
    n = len(v)
    return [round(v[int(0.025 * (n - 1))], 4), round(v[int(round(0.975 * (n - 1)))], 4)]


def bootstrap(rows: list[dict], providers: list[str], *, b: int = BOOTSTRAP_B,
              seed: int = BOOTSTRAP_SEED, pairwise: bool = False) -> dict:
    """Paired cluster bootstrap of word-weighted strict and script-agnostic WER.

    ``rows`` all share one resampling design (one split). Clusters come from
    ``row["cluster"]``. Each replicate draws clusters with replacement and
    recomputes every provider's ratio estimator sum(wer*w)/sum(w) from the
    same draw, so gaps between providers on a shared sample set are paired.
    Providers that did not score a drawn cluster simply contribute nothing
    for it (only matters outside comparable sets). Returns, per provider,
    95% percentile CIs, and with ``pairwise`` the gap a-b for every pair
    (positive = a has the higher error) with its CI.
    """
    clusters: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    for r in rows:
        w = max(r["ref_words"], 1)
        acc = clusters[r["cluster"]][r["provider"]]
        acc[0] += r["wer"] * w
        acc[1] += r["wer_sa"] * w
        acc[2] += w
    keys = sorted(clusters)
    if not keys:
        return {"providers": {}, "pairwise": {}}
    rng = random.Random(seed)
    reps: dict[str, dict[str, list[float]]] = {p: {"wer": [], "sa": []} for p in providers}
    pair_reps: dict[str, dict[str, list[float]]] = {}
    for _ in range(b):
        draw = rng.choices(keys, k=len(keys))
        tot = {p: [0.0, 0.0, 0.0] for p in providers}
        for c in draw:
            for p, acc in clusters[c].items():
                if p in tot:
                    t = tot[p]
                    t[0] += acc[0]
                    t[1] += acc[1]
                    t[2] += acc[2]
        est = {}
        for p, t in tot.items():
            if t[2]:
                est[p] = (t[0] / t[2], t[1] / t[2])
                reps[p]["wer"].append(est[p][0])
                reps[p]["sa"].append(est[p][1])
        if pairwise:
            for a in providers:
                for b_ in providers:
                    if a != b_ and a in est and b_ in est:
                        d = pair_reps.setdefault(f"{a}|{b_}", {"wer": [], "sa": []})
                        d["wer"].append(est[a][0] - est[b_][0])
                        d["sa"].append(est[a][1] - est[b_][1])
    out = {"providers": {}, "pairwise": {}}
    for p, d in reps.items():
        if d["wer"]:
            out["providers"][p] = {"wer_ci": _ci(d["wer"]), "wer_sa_ci": _ci(d["sa"]),
                                   "n_clusters": sum(1 for c in keys if p in clusters[c])}
    for k, d in pair_reps.items():
        out["pairwise"][k] = {"wer_ci": _ci(d["wer"]), "wer_sa_ci": _ci(d["sa"])}
    return out


def dataset_of(source: str) -> str:
    s = source.split(":")[0].split("/")[0]
    return {"gramvaani_telephony": "gramvaani", "mucs2021": "mucs_hinglish",
            "youtube": "heldout_yt"}.get(s, s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("manifests", nargs="+")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default=None,
                   help="default report/data.json, or data/private/reports/data.json "
                        "with --include-private")
    p.add_argument("--carry-forward", default=None, metavar="ROWS_JSON",
                   help="previous rows.json; its Gramvaani rows are kept for samples "
                        "with no local reference (provider outputs are not redistributed)")
    p.add_argument("--include-private", action="store_true",
                   help="also score split=private rows; output must go under "
                        "data/private/reports/")
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B,
                   help="bootstrap replicates (default %(default)s)")
    args = p.parse_args()
    if args.out is None:
        args.out = str(PRIVATE_REPORT_DIR / "data.json") if args.include_private else "report/data.json"
    if args.include_private:
        require_private_output(args.out)
    splits = SPLITS + (("private",) if args.include_private else ())
    carried_rows: list[dict] = []
    if args.carry_forward and Path(args.carry_forward).exists():
        carried_rows = [r for r in json.loads(Path(args.carry_forward).read_text())
                        if r.get("dataset") == "gramvaani" and r.get("split") in SPLITS]
        # Re-price carried rows at today's list rates so costs stay comparable.
        for r in carried_rows:
            if r.get("audio_s"):
                r["cost_usd"] = record_cost_usd(r["provider"], {"language": r.get("language"),
                                                "code_mixed": r.get("code_mixed")}, r["audio_s"])

    results_dir = Path(args.results_dir)
    rows: list[dict] = []
    samples_seen: set[str] = set()
    unreferenced: set[str] = set()
    n_private_skipped = 0

    for mpath in args.manifests:
        mpath = Path(mpath)
        for s in load_manifest(mpath):
            if s.split not in splits:
                n_private_skipped += 1
                continue
            if s.id in samples_seen:
                continue
            if not s.reference_text:
                unreferenced.add(s.id)
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
                cost = record_cost_usd(prov, {**rec, "language": s.language,
                                              "code_mixed": s.code_mixed}, audio_s)
                rows.append({
                    "sample_id": s.id,
                    "provider": prov,
                    "dataset": dataset_of(s.source),
                    "language": s.language,
                    "code_mixed": s.code_mixed,
                    "split": s.split,
                    "cluster": cluster_of(s.id, s.split),
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

    # Carry forward rows (Gramvaani) for samples we could not score locally.
    n_carried = 0
    for r in carried_rows:
        if r["sample_id"] in unreferenced and r["sample_id"] not in samples_seen:
            r = dict(r, cluster=r.get("cluster") or cluster_of(r["sample_id"], r["split"]),
                     carried=True)
            rows.append(r)
            n_carried += 1
    samples_seen |= {r["sample_id"] for r in rows if r.get("carried")}

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
    for split in splits:
        per_prov = [
            {r["sample_id"] for r in rows if r["provider"] == p and r["split"] == split}
            for p in present
        ]
        per_prov = [s for s in per_prov if s]
        comparable_ids[split] = set.intersection(*per_prov) if len(per_prov) == len(present) else set()
    rows_all = rows
    rows = [r for r in rows_all if r["sample_id"] in comparable_ids[r["split"]]]
    by_provider_comparable = aggregate(lambda r: (r["provider"], r["split"]))
    comparable_rows = rows

    # Contributors on held-out: also score them on the rows where their own
    # output is NOT the reference. Still not independent (their hypothesis
    # shaped the consensus), but no longer scored against itself.
    rows = [r for r in comparable_rows if r["split"] != "public" and not r["self_ref"]
            and r["provider"] in SILVER_CONTRIBUTORS]
    by_provider_nonself = aggregate(lambda r: (r["provider"], r["split"]))
    rows = rows_all

    # Bootstrap CIs + pairwise gaps on the comparable sets.
    stats: dict[str, dict] = {}
    for split in splits:
        sr = [r for r in comparable_rows if r["split"] == split]
        if not sr:
            continue
        bs = bootstrap(sr, present, b=args.bootstrap, pairwise=True)
        for p, ci in bs["providers"].items():
            by_provider_comparable[f"{p}|{split}"].update(ci)
        stats[split] = {
            "resampling_unit": "source video" if split == "heldout" else "sample",
            "n_clusters": len({r["cluster"] for r in sr}),
            "pairwise": bs["pairwise"],
        }
        nonself = [r for r in sr if not r["self_ref"] and r["provider"] in SILVER_CONTRIBUTORS]
        if split != "public" and nonself:
            for p, ci in bootstrap(nonself, present, b=args.bootstrap)["providers"].items():
                by_provider_nonself[f"{p}|{split}"].update(ci)
        print(f"bootstrap {split}: {stats[split]['n_clusters']} clusters "
              f"({stats[split]['resampling_unit']}), B={args.bootstrap}", flush=True)
    # Gap of every system against the top system on each comparable split,
    # per metric. On held-out the top is chosen among non-contributors only:
    # contributors are not ranked there (they score against themselves).
    gaps: dict[str, dict] = {}
    for split, st in stats.items():
        gaps[split] = {}
        for metric, field, cif in (("wer", "wer", "wer_ci"),
                                   ("wer_script_agnostic", "wer_script_agnostic", "wer_sa_ci")):
            rankable = [p for p in present if f"{p}|{split}" in by_provider_comparable
                        and (split == "public" or p not in SILVER_CONTRIBUTORS)]
            if not rankable:
                continue
            top = min(rankable, key=lambda p: by_provider_comparable[f"{p}|{split}"][field])
            tv = by_provider_comparable[f"{top}|{split}"][field]
            g = {}
            for p in present:
                k = f"{p}|{split}"
                if p == top or k not in by_provider_comparable:
                    continue
                ci = st["pairwise"][f"{p}|{top}"][cif]
                g[p] = {"gap": round(by_provider_comparable[k][field] - tv, 4), "ci": ci,
                        "tie": ci[0] <= 0 <= ci[1]}
            gaps[split][metric] = {"top": top, "gaps": g}

    # Matrix cells: FLEURS per language, every other dataset as one cell.
    def cell_of(r):
        return f"fleurs:{r['language']}" if r["dataset"] == "fleurs" else r["dataset"]

    by_provider_cell = aggregate(lambda r: (r["provider"], cell_of(r)))
    cell_rows: dict[str, list] = defaultdict(list)
    for r in rows_all:
        cell_rows[cell_of(r)].append(r)
    for cell, cr in sorted(cell_rows.items()):
        for p, ci in bootstrap(cr, present, b=max(200, args.bootstrap // 2))["providers"].items():
            by_provider_cell[f"{p}|{cell}"].update(ci)

    report = {
        "by_provider": aggregate(lambda r: (r["provider"], r["split"])),
        "by_provider_comparable": by_provider_comparable,
        "by_provider_comparable_nonself": by_provider_nonself,
        "comparable_counts": {k: len(v) for k, v in comparable_ids.items()},
        "comparable_stats": {k: {kk: vv for kk, vv in v.items() if kk != "pairwise"}
                             for k, v in stats.items()},
        "pairwise_comparable": {k: v["pairwise"] for k, v in stats.items()},
        "gaps_vs_top": gaps,
        "by_provider_dataset": aggregate(lambda r: (r["provider"], r["dataset"])),
        "by_provider_language": aggregate(lambda r: (r["provider"], r["language"])),
        "by_provider_cell": by_provider_cell,
        "silver_caveat_providers": sorted(SILVER_CONTRIBUTORS),
        "methodology": {
            "wer": "word-weighted: sum(per-sample WER x reference words) / sum(reference words)",
            "accuracy": "max(0, 1 - word-weighted script-agnostic WER)",
            "ci": f"95% percentile bootstrap, B={args.bootstrap} (cells B={max(200, args.bootstrap // 2)}), "
                  "seed fixed; held-out resamples source videos, public resamples samples",
            "tie": "gap CI includes 0",
            "script_agnostic": "metrics.script_agnostic_pair (v2: all Indic scripts + Urdu skeleton)",
            "carried_rows": n_carried,
        },
        "row_count": len(rows),
        "sample_count": len(samples_seen),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    # Per-sample rows for the showcase page (side-by-side comparisons).
    (out_path.parent / "rows.json").write_text(json.dumps(rows))
    print(f"wrote {out_path} ({len(rows)} scored rows, "
          f"{len(samples_seen)} samples, {n_carried} carried-forward rows, "
          f"{n_private_skipped} private-split samples skipped)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
