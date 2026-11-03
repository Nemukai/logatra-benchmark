"""Select, download and process paired AIR bulletins from the discovery catalog.

For each language: pick recent cards (date >= --since) that carry both an
audio URL and a PDF URL, round-robin over distinct (RNU, time-slot) so one
station does not dominate, then for each pick:

  1. download original audio + PDF (resumable, Range-retry),
  2. convert audio to 16 kHz mono s16 WAV (ffmpeg),
  3. extract + classify the script text (airtext.py),
  4. check pairing evidence and keep only confident pairs.

Output per bulletin under data/private/air/<lang>/<id>/ (audio.<ext>,
script.pdf, audio_16k.wav, script.txt) plus a sidecar JSON in
data/private/air/items/<id>.json (the source of truth for build.py).
Rejected candidates are recorded in items/<id>.json with "kept": false and
their media deleted, so they are not retried. Idempotent and resumable.

Usage (needs pymupdf):
    uv run --no-project --with pymupdf python scripts/air/fetch.py \
        [--per-lang 6] [--since 2026-09-01] [--langs hindi,tamil] [--budget-gb 7.5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import airtext  # noqa: E402
from airlib import (CATALOG, DATA, ITEMS, LANGS, download, dir_size, norm_lang,  # noqa: E402
                    now_iso, read_jsonl, slug)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".mp4"}
MIN_DUR, MAX_DUR = 120.0, 2400.0


def resolve(card: dict) -> tuple[str | None, str]:
    """(language key, RNU/station) for a card; fixes swapped city/lang labels."""
    lang = norm_lang(card["lang_label"]) or norm_lang(card["city"])
    city = card["city"]
    if norm_lang(city):                      # NSD cards use the language as "city"
        other = card["lang_label"]
        city = other if other and not norm_lang(other) else "Delhi-NSD" if card["source"] == "nsd-audio" else city
    return lang, slug(city)


def make_id(lang: str, rnu: str, card: dict) -> str:
    hhmm = re.sub(r"\D", "", card["time"])[:4].zfill(4) or "0000"
    return f"air_{lang}_{rnu}_{card['date'].replace('-', '')}_{hhmm}_{card['post_id']}"


def probe_duration(path: Path) -> float | None:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def to_wav(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 44:
        return
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "pcm_s16le", str(tmp)],
                   check=True)
    tmp.replace(dst)


def pdf_date(d: str | None) -> str | None:
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", d or "")
    return f"{m[1]}-{m[2]}-{m[3]}" if m else None


def time_tokens(url: str) -> list[str]:
    """4-digit HHMM-looking tokens in a file name (informational evidence)."""
    name = url.rsplit("/", 1)[-1]
    return [t for t in re.findall(r"(?<!\d)(\d{4})(?!\d)", name) if int(t[:2]) < 24 and int(t[2:]) < 60]


def process(card: dict, lang: str, rnu: str, item_id: str, dup_urls: Counter) -> dict:
    d = DATA / lang / item_id
    a_ext = Path(card["audio_url"]).suffix.lower()
    audio, pdf, wav, txt = d / f"audio{a_ext}", d / "script.pdf", d / "audio_16k.wav", d / "script.txt"
    rec = {"id": item_id, "language_key": lang, "language": LANGS[lang]["iso"], "rnu": rnu,
           "source_listing": card["source"], "post_id": card["post_id"], "city_label": card["city"],
           "lang_label": card["lang_label"], "bulletin_date": card["date"], "bulletin_time": card["time"],
           "audio_url": card["audio_url"], "pdf_url": card["pdf_url"], "listing_url": card["listing_url"]}
    reasons = []
    download(card["audio_url"], audio)
    download(card["pdf_url"], pdf)
    if pdf.read_bytes()[:5] != b"%PDF-":
        reasons.append("pdf_url_not_a_pdf")
    dur = probe_duration(audio)
    if dur is None or not (MIN_DUR <= dur <= MAX_DUR):
        reasons.append(f"audio_duration_out_of_range:{dur}")
    meta = {}
    ext = {"text": "", "status": "no_text_layer", "extractor": None, "quality": {}}
    if not reasons:
        meta = airtext.pdf_meta(pdf)
        ext = airtext.extract_best(pdf, LANGS[lang]["scripts"])
        txt.write_text(ext["text"], encoding="utf-8")
        to_wav(audio, wav)
    hdr = airtext.header_dates(ext["text"]) if ext["text"] else []
    hdr_match = airtext.date_matches(hdr, card["date"])
    cdate = pdf_date(meta.get("creation_date")) or pdf_date(meta.get("mod_date"))
    cdiff = None
    if cdate:
        cdiff = (dt.date.fromisoformat(cdate) - dt.date.fromisoformat(card["date"])).days
    listed_hhmm = re.sub(r"\D", "", card["time"])[:4].zfill(4)
    evidence = {
        "cms_same_post": True,   # audio and pdf linked from one bulletin post (share-bulletin?id=post_id)
        "audio_url_shared_by_other_posts": dup_urls[card["audio_url"]] > 1,
        "pdf_url_shared_by_other_posts": dup_urls[card["pdf_url"]] > 1,
        "pdf_header_dates": hdr,
        "pdf_header_date_match": hdr_match,
        "pdf_creation_date": cdate,
        "pdf_creation_minus_bulletin_days": cdiff,
        "listed_time_hhmm": listed_hhmm,
        "audio_name_time_tokens": time_tokens(card["audio_url"]),
        "pdf_name_time_tokens": time_tokens(card["pdf_url"]),
    }
    if evidence["audio_url_shared_by_other_posts"] or evidence["pdf_url_shared_by_other_posts"]:
        reasons.append("media_url_reused_across_posts")
    if hdr_match is False:
        reasons.append("pdf_header_date_mismatch")
    if hdr_match is None and cdiff is not None and not (-1 <= cdiff <= 2):
        reasons.append("pdf_creation_date_far_from_bulletin")
    for key in ("audio_name_time_tokens", "pdf_name_time_tokens"):
        toks = evidence[key]
        if toks and listed_hhmm not in toks:
            evidence[f"{key.split('_')[0]}_name_time_differs"] = True
    if hdr_match:
        conf = "high"
    elif cdiff is not None and -1 <= cdiff <= 1:
        conf = "medium"
    else:
        conf = "low"
    evidence["confidence"] = conf
    if conf == "low" and not reasons:
        reasons.append("insufficient_pairing_evidence")

    rec.update({
        "kept": not reasons, "reject_reasons": reasons, "pairing_evidence": evidence,
        "duration_s": round(dur, 2) if dur else None,
        "wav_duration_s": round((wav.stat().st_size - 44) / 32000, 2) if wav.exists() else None,
        "text_status": ext["status"], "text_extractor": ext["extractor"], "text_quality": ext["quality"],
        "char_count": len(ext["text"]), "pdf_meta": meta,
        "local": {k: str(p.relative_to(DATA)) for k, p in
                  (("audio", audio), ("pdf", pdf), ("wav", wav), ("text", txt)) if p.exists()},
        "fetched_at": now_iso(),
    })
    if reasons:  # free disk; keep the decision so we never retry this post
        shutil.rmtree(d, ignore_errors=True)
        rec["local"] = {}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-lang", type=int, default=6, help="target kept bulletins per language")
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--langs", default="", help="comma list of language keys (default: all)")
    ap.add_argument("--budget-gb", type=float, default=7.5, help="stop when data/private/air exceeds this")
    ap.add_argument("--max-tries-factor", type=int, default=3, help="candidates tried per language = factor*per-lang")
    args = ap.parse_args()

    ITEMS.mkdir(parents=True, exist_ok=True)
    catalog = read_jsonl(CATALOG)
    dup_urls = Counter(u for c in catalog for u in {c["audio_url"], c["pdf_url"]} if u)
    done = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in ITEMS.glob("*.json")}

    by_lang: dict[str, list] = defaultdict(list)
    for c in catalog:
        lang, rnu = resolve(c)
        if (not lang or not c.get("pdf_url") or not c.get("date") or c["date"] < args.since
                or Path(c["audio_url"]).suffix.lower() not in AUDIO_EXT):
            continue
        by_lang[lang].append((c, rnu))
    wanted = [l for l in args.langs.split(",") if l] or sorted(by_lang)
    budget = args.budget_gb * 1e9

    for lang in wanted:
        cands = sorted(by_lang.get(lang, []), key=lambda x: (x[0]["date"], x[0]["time"], x[0]["post_id"]), reverse=True)
        # round-robin over (rnu, slot) so the sample spans stations and time slots
        groups: dict[tuple, list] = defaultdict(list)
        for c, rnu in cands:
            groups[(rnu, c["time"])].append((c, rnu))
        order, gl = [], list(groups.values())
        while any(gl):
            for g in gl:
                if g:
                    order.append(g.pop(0))
        kept = sum(1 for r in done.values() if r.get("kept") and r["language_key"] == lang)
        tries = 0
        print(f"[{lang}] {len(order)} candidates, {kept} already kept", flush=True)
        for c, rnu in order:
            if kept >= args.per_lang or tries >= args.max_tries_factor * args.per_lang:
                break
            iid = make_id(lang, rnu, c)
            if iid in done:
                continue
            if dir_size(DATA) > budget:
                print("disk budget reached; stopping", flush=True)
                return
            tries += 1
            try:
                rec = process(c, lang, rnu, iid, dup_urls)
            except Exception as e:  # noqa: BLE001 - network/ffmpeg errors: skip, retry next run
                print(f"  ! {iid}: {e}", flush=True)
                continue
            (ITEMS / f"{iid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            done[iid] = rec
            kept += rec["kept"]
            print(f"  {'KEEP' if rec['kept'] else 'drop'} {iid} {rec['duration_s']}s {rec['text_status']} "
                  f"conf={rec['pairing_evidence']['confidence']} {rec['reject_reasons'] or ''}", flush=True)


if __name__ == "__main__":
    main()
