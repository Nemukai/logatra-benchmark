"""Export per-case explorer data: report/cases.js (window.LOGATRA_CASES).

One entry per scored sample: ground truth, language, dataset, and every
provider's transcript with accuracy numbers. Texts are truncated at 600 chars
(repetition-loop outputs would otherwise dominate the file); a `cut` flag
marks truncation. Accuracy = max(0, 1 - WER), so insertion-heavy garbage
floors at 0% instead of printing nonsense like 34125%.

Private split: samples with split == "private" are never exported to
report/. ``--include-private`` (with the private manifest and rows passed
explicitly) writes to data/private/reports/cases.js instead, which is
gitignored and never published.

Usage: uv run python scripts/export_explorer.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logatra_benchmark.manifest import (  # noqa: E402
    PRIVATE_REPORT_DIR,
    PUBLISHABLE_SPLITS,
    load_manifest,
    require_private_output,
)
from logatra_benchmark.runner import _cache_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("sarvam-saaras-v4", "sarvam-saaras", "sarvam", "whisper", "gemini", "voxtral", "gpt-audio")
CUT = 600

LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    "hi-en": "Hinglish",
}


def acc(wer: float | None) -> float | None:
    return None if wer is None else round(max(0.0, 1.0 - wer) * 100, 1)


_REF_PROVIDER_RE = re.compile(r":(?:silver|adjudicated)\((\w+)")
DEFAULT_MANIFESTS = (ROOT / "data/manifests/full_public.jsonl",
                     ROOT / "data/manifests/heldout_yt.silver.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("manifests", nargs="*", type=Path, default=list(DEFAULT_MANIFESTS))
    ap.add_argument("--rows", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--include-private", action="store_true")
    args = ap.parse_args()
    splits = PUBLISHABLE_SPLITS + (("private",) if args.include_private else ())
    if args.include_private:
        args.rows = args.rows or PRIVATE_REPORT_DIR / "rows.json"
        args.out = args.out or PRIVATE_REPORT_DIR / "cases.js"
        require_private_output(args.out)
    args.rows = args.rows or ROOT / "report" / "rows.json"
    args.out = args.out or ROOT / "report" / "cases.js"

    rows = [r for r in json.loads(args.rows.read_text()) if r.get("split") in splits]
    by_sample: dict[str, dict] = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], {"m": {}})["m"][r["provider"]] = r

    out = []
    for mp in args.manifests:
        for s in load_manifest(mp):
            if s.split not in splits:
                continue
            if not s.reference_text or s.id not in by_sample:
                continue
            entry_rows = by_sample[s.id]["m"]
            provs = {}
            for p in PROVIDERS:
                cache = _cache_path(args.results_dir, p, s.id)
                if not cache.exists() or p not in entry_rows:
                    continue
                rec = json.loads(cache.read_text())
                text = (rec.get("text") or "").strip()
                if not text:
                    continue
                r = entry_rows[p]
                provs[p] = {
                    "t": text[:CUT],
                    "cut": len(text) > CUT,
                    "a": acc(r["wer"]),          # strict accuracy
                    "as_": acc(r["wer_sa"]),     # script-agnostic accuracy
                    "lat": r["latency_s"],
                }
            if not provs:
                continue
            ds = entry_rows[next(iter(entry_rows))]["dataset"]
            lang = s.language
            out.append({
                "id": s.id,
                "ref": s.reference_text[:CUT],
                "lang": lang,
                "langName": LANG_NAMES.get(lang, lang) + (" (code-mixed)" if s.code_mixed and lang != "hi-en" else ""),
                "ds": ds,
                "real": s.split == "heldout",
                "silver": ":silver(" in s.source or ":adjudicated(" in s.source,
                "refBy": (m.group(1) if (m := _REF_PROVIDER_RE.search(s.source)) else None),
                "p": provs,
            })

    js = "window.LOGATRA_CASES=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(js)
    print(f"wrote {args.out}: {len(out)} cases, {len(js)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
