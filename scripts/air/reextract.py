"""Re-run text extraction/classification on already-fetched bulletins.

Use after improving airtext.py (or adding a legacy-font converter): updates
script.txt and the text fields of every kept items/<id>.json without any
network access. Pairing evidence that depends on the text (header date) is
refreshed too.

Usage:
    uv run --no-project --with pymupdf python scripts/air/reextract.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import airtext  # noqa: E402
from airlib import DATA, ITEMS, LANGS  # noqa: E402


def main() -> None:
    stats = Counter()
    for p in sorted(ITEMS.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if not r.get("kept"):
            continue
        pdf = DATA / r["local"]["pdf"]
        ext = airtext.extract_best(pdf, LANGS[r["language_key"]]["scripts"])
        txt = pdf.with_name("script.txt")
        txt.write_text(ext["text"], encoding="utf-8")
        hdr = airtext.header_dates(ext["text"])
        r["pairing_evidence"]["pdf_header_dates"] = hdr
        r["pairing_evidence"]["pdf_header_date_match"] = airtext.date_matches(hdr, r["bulletin_date"])
        r.update(text_status=ext["status"], text_extractor=ext["extractor"],
                 text_quality=ext["quality"], char_count=len(ext["text"]))
        r["local"]["text"] = str(txt.relative_to(DATA))
        if ext["status"] == "wrong_script":   # PDF is in another language: unpair, free disk
            r["kept"] = False
            r["reject_reasons"] = r.get("reject_reasons", []) + ["script_text_in_unexpected_script"]
            shutil.rmtree(pdf.parent, ignore_errors=True)
            r["local"] = {}
        p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        stats[(r["language_key"], ext["status"])] += 1
    for (lang, st), n in sorted(stats.items()):
        print(f"{lang:14s} {st:26s} {n}")


if __name__ == "__main__":
    main()
