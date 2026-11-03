"""Build the AIR index (JSONL + SQLite) and the benchmark manifest from items/*.json.

Outputs
  data/private/air/index.jsonl          one row per kept bulletin
  data/private/air/air.sqlite           table `bulletins`, same columns (nested fields as JSON text)
  data/private/manifests/air.jsonl      repo manifest schema; reference_text is filled
                                        only for text_status == "ok" (else "")
  data/private/air/manifest_with_suspect_text.jsonl
                                        same rows, but reference_text also filled for
                                        unicode_glyphmap_suspect scripts (noisy refs)
  data/private/air/rejected.jsonl       candidates dropped, with reasons

Usage:
    python scripts/air/build.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from airlib import DATA, INDEX, ITEMS, MANIFEST, REPO, SQLITE, dir_size, write_jsonl  # noqa: E402

LICENSE = "Prasar Bharati, private evaluation only, not redistributable"
COLUMNS = ["id", "language", "language_key", "rnu", "source_listing", "post_id", "bulletin_date",
           "bulletin_time", "audio_url", "pdf_url", "audio_path", "pdf_path", "wav_path", "text_path",
           "duration_s", "wav_duration_s", "text_status", "text_extractor", "char_count",
           "pairing_confidence", "pairing_evidence", "text_quality", "pdf_fonts", "fetched_at"]
JSON_COLS = {"pairing_evidence", "text_quality", "pdf_fonts"}


def repo_rel(p: str | None) -> str | None:
    return str((DATA / p).relative_to(REPO)) if p else None


def main() -> None:
    items = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ITEMS.glob("*.json"))]
    for r in items:  # re-extraction may reveal a wrong-language script: drop the pair
        if r.get("kept") and r.get("text_status") == "wrong_script":
            r["kept"] = False
            r["reject_reasons"] = r.get("reject_reasons", []) + ["script_text_in_unexpected_script"]
    kept = [r for r in items if r.get("kept")]
    rows = []
    for r in sorted(kept, key=lambda r: (r["language_key"], r["bulletin_date"], r["bulletin_time"], r["id"])):
        loc = r["local"]
        rows.append({
            "id": r["id"], "language": r["language"], "language_key": r["language_key"], "rnu": r["rnu"],
            "source_listing": r["source_listing"], "post_id": r["post_id"],
            "bulletin_date": r["bulletin_date"], "bulletin_time": r["bulletin_time"],
            "audio_url": r["audio_url"], "pdf_url": r["pdf_url"],
            "audio_path": repo_rel(loc.get("audio")), "pdf_path": repo_rel(loc.get("pdf")),
            "wav_path": repo_rel(loc.get("wav")), "text_path": repo_rel(loc.get("text")),
            "duration_s": r["duration_s"], "wav_duration_s": r["wav_duration_s"],
            "text_status": r["text_status"], "text_extractor": r["text_extractor"],
            "char_count": r["char_count"], "pairing_confidence": r["pairing_evidence"]["confidence"],
            "pairing_evidence": r["pairing_evidence"], "text_quality": r["text_quality"],
            "pdf_fonts": r.get("pdf_meta", {}).get("fonts"), "fetched_at": r["fetched_at"],
        })
    write_jsonl(INDEX, rows)
    write_jsonl(DATA / "rejected.jsonl", [
        {k: r.get(k) for k in ("id", "audio_url", "pdf_url", "reject_reasons", "pairing_evidence", "duration_s")}
        for r in items if not r.get("kept")])

    tmp = SQLITE.with_suffix(".sqlite.tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute(f"CREATE TABLE bulletins ({', '.join(c + (' TEXT PRIMARY KEY' if c == 'id' else '') for c in COLUMNS)})")
    con.executemany(f"INSERT INTO bulletins VALUES ({','.join('?' * len(COLUMNS))})",
                    [[json.dumps(row[c], ensure_ascii=False) if c in JSON_COLS else row[c] for c in COLUMNS]
                     for row in rows])
    con.commit()
    con.close()
    tmp.replace(SQLITE)

    mdir = MANIFEST.parent
    strict, loose = [], []
    for row in rows:
        text = (REPO / row["text_path"]).read_text(encoding="utf-8") if row["text_path"] else ""
        hhmm = row["pairing_evidence"]["listed_time_hhmm"]
        base = {
            "id": row["id"],
            "audio_path": str(Path("..") / (REPO / row["wav_path"]).relative_to(mdir.parent)),
            "reference_text": "",
            "language": row["language"], "code_mixed": False, "num_speakers": 1,
            "speaker_segments": None,
            "source": f"air:{row['rnu']}:{row['bulletin_date']}:{hhmm}:script",
            "license": LICENSE, "split": "private",
        }
        strict.append({**base, "reference_text": text if row["text_status"] == "ok" else ""})
        loose.append({**base, "reference_text": text if row["text_status"] in ("ok", "unicode_glyphmap_suspect") else ""})
    write_jsonl(MANIFEST, strict)
    write_jsonl(DATA / "manifest_with_suspect_text.jsonl", loose)

    # summary
    agg = defaultdict(lambda: defaultdict(float))
    for row in rows:
        a = agg[row["language_key"]]
        a["n"] += 1
        a["hours"] += (row["duration_s"] or 0) / 3600
        a[row["text_status"]] += 1
    print(f"{'language':14s} {'n':>3s} {'hours':>6s}  ok/suspect/legacy/other")
    tot = defaultdict(float)
    for k in sorted(agg):
        a = agg[k]
        other = a["n"] - a["ok"] - a["unicode_glyphmap_suspect"] - a["legacy_font_unconverted"]
        print(f"{k:14s} {int(a['n']):3d} {a['hours']:6.2f}  {int(a['ok'])}/{int(a['unicode_glyphmap_suspect'])}/"
              f"{int(a['legacy_font_unconverted'])}/{int(other)}")
        for kk, v in a.items():
            tot[kk] += v
    print(f"TOTAL {int(tot['n'])} bulletins, {tot['hours']:.2f} h, ok={int(tot['ok'])} "
          f"suspect={int(tot['unicode_glyphmap_suspect'])} legacy={int(tot['legacy_font_unconverted'])}; "
          f"rejected {len(items) - len(kept)}; disk {dir_size(DATA) / 1e9:.2f} GB")
    print(f"wrote {INDEX}, {SQLITE}, {MANIFEST}")


if __name__ == "__main__":
    main()
