"""Export per-case explorer data: report/cases.js (window.KOE_CASES).

One entry per scored sample: ground truth, language, dataset, and every
provider's transcript with accuracy numbers. Texts are truncated at 600 chars
(repetition-loop outputs would otherwise dominate the file); a `cut` flag
marks truncation. Accuracy = max(0, 1 - WER), so insertion-heavy garbage
floors at 0% instead of printing nonsense like 34125%.

Usage: uv run python scripts/export_explorer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koe_benchmark.manifest import load_manifest  # noqa: E402
from koe_benchmark.runner import _cache_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("sarvam-saaras", "sarvam", "whisper", "gemini", "voxtral", "gpt-audio")
CUT = 600

LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia",
    "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    "hi-en": "Hinglish",
}


def acc(wer: float | None) -> float | None:
    return None if wer is None else round(max(0.0, 1.0 - wer) * 100, 1)


def main() -> int:
    rows = json.loads((ROOT / "report" / "rows.json").read_text())
    by_sample: dict[str, dict] = {}
    for r in rows:
        by_sample.setdefault(r["sample_id"], {"m": {}})["m"][r["provider"]] = r

    out = []
    for mp in (ROOT / "data/manifests/full_public.jsonl",
               ROOT / "data/manifests/heldout_yt.silver.jsonl"):
        for s in load_manifest(mp):
            if not s.reference_text or s.id not in by_sample:
                continue
            entry_rows = by_sample[s.id]["m"]
            provs = {}
            for p in PROVIDERS:
                cache = _cache_path(ROOT / "results", p, s.id)
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
                "silver": ":silver(" in s.source,
                "p": provs,
            })

    js = "window.KOE_CASES=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";"
    (ROOT / "report" / "cases.js").write_text(js)
    print(f"wrote report/cases.js: {len(out)} cases, {len(js)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
