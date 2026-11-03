"""Collect CC-BY YouTube audio with manual (uploader) captions for a private eval split.

Subcommands (each is resumable and idempotent; state lives in yt.sqlite):

    discover   YouTube search (CC + subtitles filters) per language -> candidates
    probe      full metadata per candidate: license, date, manual subs, caption
               quality checks -> videos.status accepted / rejected / blocked
    fetch      download audio of accepted videos, 16 kHz mono s16 WAV, cut
               cue-aligned chunks (<= 30 s), store chunk rows
    export     index.jsonl, sources.json, sqlite chunks, manifest JSONL
    status     per-language summary
    all        discover -> probe -> fetch -> export

Run from the repo root:

    uv run --with yt-dlp --with yt-dlp-ejs python scripts/youtube_captions/ytcap.py all
    uv run --with yt-dlp python scripts/youtube_captions/ytcap.py probe --langs ta te

No paid ASR is used anywhere. No cookies, accounts or proxies: when YouTube
answers with a bot check the video is marked `blocked` and the run stops after
a few consecutive blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import wave
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions as CAP  # noqa: E402
import config as C  # noqa: E402
import db as DB  # noqa: E402

BLOCK_MARKERS = ("confirm you", "not a bot", "sign in to confirm", "http error 429",
                 "too many requests", "page needs to be reloaded")
MAX_CONSECUTIVE_BLOCKS = 4


class Blocked(Exception):
    pass


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def polite_sleep(rng: tuple[float, float]) -> None:
    time.sleep(random.uniform(*rng))


def yt_dlp():
    try:
        import yt_dlp  # type: ignore
        return yt_dlp
    except ImportError:
        sys.exit("yt-dlp missing: run with `uv run --with yt-dlp python ...`")


def ydl_opts(player_clients: str | None = None, **extra) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "sleep_interval_requests": C.YTDLP_SLEEP_REQUESTS,
        "retries": 3,
        "extractor_retries": 2,
    }
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    if player_clients:
        opts["extractor_args"] = {"youtube": {"player_client": player_clients.split(",")}}
    opts.update(extra)
    return opts


def is_block_error(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in BLOCK_MARKERS)


# ---------------------------------------------------------------- discover --

def cmd_discover(args, con) -> None:
    ytd = yt_dlp()
    langs = args.langs or list(C.LANGS)
    # flat listings: yt-dlp would apply sleep_interval_requests per listed item,
    # so it is disabled here; polite_sleep() spaces the searches instead
    opts = ydl_opts(extract_flat="in_playlist", playlistend=args.per_query)
    opts.pop("sleep_interval_requests")
    with ytd.YoutubeDL(opts) as ydl:
        for code in langs:
            lang = C.LANGS[code]
            for query in lang.queries:
                for fname in args.filters:
                    done = con.execute("SELECT 1 FROM searches WHERE lang=? AND query=? AND filter=?",
                                       (code, query, fname)).fetchone()
                    if done and not args.refresh:
                        continue
                    url = ("https://www.youtube.com/results?search_query="
                           + urllib.parse.quote_plus(query) + "&sp=" + C.SEARCH_FILTERS[fname])
                    try:
                        info = ydl.extract_info(url, download=False)
                    except Exception as exc:  # noqa: BLE001
                        log(f"  ! search failed [{code}] {query!r}: {exc}")
                        if is_block_error(str(exc)):
                            log("search is blocked; stopping discovery")
                            return
                        continue
                    entries = [e for e in (info or {}).get("entries") or [] if e and e.get("id")]
                    for e in entries:
                        if e.get("ie_key") not in (None, "Youtube") or len(e["id"]) != 11:
                            continue
                        con.execute(
                            "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?)",
                            (e["id"], code, query, fname, e.get("title"),
                             e.get("channel") or e.get("uploader"),
                             e.get("channel_id"), e.get("duration"), DB.now()))
                    DB.upsert(con, "searches", {"lang": code, "query": query, "filter": fname,
                                                "n_results": len(entries), "done_at": DB.now()})
                    log(f"[{code}] {fname:15s} {query!r}: {len(entries)} results")
                    polite_sleep(C.SLEEP_BETWEEN_SEARCHES)
        if not args.no_seeds:
            discover_seeds(ydl, con, langs, args)
    for row in con.execute("SELECT lang, COUNT(*) n FROM candidates GROUP BY lang"):
        log(f"candidates {row['lang']:6s} {row['n']}")


def discover_seeds(ydl, con, langs, args) -> None:
    seeds = json.loads((Path(__file__).with_name("seeds.json")).read_text())
    for code in langs:
        for url in seeds.get(code, []):
            done = con.execute("SELECT 1 FROM searches WHERE lang=? AND query=? AND filter='seed'",
                               (code, url)).fetchone()
            if done and not args.refresh:
                continue
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as exc:  # noqa: BLE001
                log(f"  ! seed failed [{code}] {url}: {str(exc)[:120]}")
                continue
            entries = [e for e in (info or {}).get("entries") or [] if e and e.get("id")]
            for e in entries:
                con.execute("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?)",
                            (e["id"], code, url, "seed", e.get("title"),
                             e.get("channel") or info.get("channel") or info.get("uploader"),
                             e.get("channel_id") or info.get("channel_id"),
                             e.get("duration"), DB.now()))
            DB.upsert(con, "searches", {"lang": code, "query": url, "filter": "seed",
                                        "n_results": len(entries), "done_at": DB.now()})
            log(f"[{code}] seed {url}: {len(entries)} videos")
            polite_sleep(C.SLEEP_BETWEEN_SEARCHES)


# ------------------------------------------------------------------- probe --

def freshness(upload_date: str | None) -> str:
    if not upload_date:
        return "unknown"
    if upload_date >= C.FRESH_AFTER:
        return "fresh"
    if upload_date >= C.OLDEST_OK:
        return "older"
    return "stale"


def spoken_language(info: dict) -> str | None:
    """Best signal YouTube gives for the spoken language, without any ASR call."""
    for key in (info.get("automatic_captions") or {}):
        if key.endswith("-orig"):
            return key[:-5].split("-")[0]
    lang = info.get("language")
    return lang.split("-")[0] if lang else None


def manual_tracks(info: dict, prefixes: tuple) -> list[str]:
    subs = info.get("subtitles") or {}
    keys = [k for k in subs if k != "live_chat" and k.split("-")[0].lower() in prefixes]
    # exact code first, then regional / named variants
    return sorted(keys, key=lambda k: (k.split("-")[0] not in prefixes or "-" in k, k))


def fetch_track(ydl, info: dict, key: str, dest_dir: Path) -> Path:
    fmts = info["subtitles"][key]
    pref = {"vtt": 0, "json3": 1, "srv3": 2, "srt": 3}
    fmt = sorted(fmts, key=lambda f: pref.get(f.get("ext"), 9))[0]
    ext = fmt.get("ext") if fmt.get("ext") in ("vtt", "json3", "srt") else "vtt"
    if fmt.get("data"):
        data = fmt["data"].encode()
    else:
        data = ydl.urlopen(fmt["url"]).read()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"captions.{key}.{ext}"
    out.write_bytes(data)
    return out


def guess_speakers(info: dict, quality: dict) -> int:
    t = (info.get("title") or "").lower()
    if quality.get("speaker_markers", 0) >= 4 or re.search(
            r"podcast|interview|conversation|debate|discussion|panel| ft\.| feat|with |\bx\b|"
            r"बातचीत|चर्चा|इंटरव्यू|நேர்காணல்|ఇంటర్వ్యూ|ಸಂದರ್ಶನ|അഭിമുഖം|ਗੱਲਬਾਤ|সাক্ষাৎকার|مکالمہ|انٹرویو",
            t):
        return 2
    return 1


def trimmed_info(info: dict) -> dict:
    keep = ("id", "title", "channel", "channel_id", "channel_url", "uploader", "upload_date",
            "license", "duration", "categories", "tags", "language", "description",
            "webpage_url", "view_count", "live_status", "was_live")
    d = {k: info.get(k) for k in keep}
    d["manual_subtitle_langs"] = sorted((info.get("subtitles") or {}).keys())
    d["auto_caption_orig"] = [k for k in (info.get("automatic_captions") or {}) if k.endswith("-orig")]
    return d


def probe_one(ydl, con, vid: str, cand_langs: list[str]) -> dict:
    """Return a videos-row dict. Raises Blocked on bot checks."""
    url = f"https://www.youtube.com/watch?v={vid}"
    row = {"video_id": vid, "url": url, "probed_at": DB.now()}
    try:
        info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if is_block_error(msg):
            raise Blocked(msg[:200]) from exc
        return {**row, "status": "error", "reason": msg[:200]}
    row.update({
        "title": info.get("title"), "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"), "upload_date": info.get("upload_date"),
        "license": info.get("license"), "duration": info.get("duration"),
        "category": ",".join(info.get("categories") or []),
    })
    row["freshness"] = freshness(row["upload_date"])

    def rej(reason, lang=None):
        return {**row, "status": "rejected", "reason": reason, "lang": lang or cand_langs[0]}

    if "creative commons" not in (row["license"] or "").lower():
        return rej(f"license:{row['license']}")
    if row["freshness"] == "stale":
        return rej(f"too_old:{row['upload_date']}")
    if row["freshness"] == "unknown":
        return rej("no_upload_date")
    if info.get("live_status") in ("is_live", "is_upcoming"):
        return rej("live")
    has_audio = any(f.get("acodec") not in (None, "none") for f in info.get("formats") or [])
    player_blocked = not info.get("automatic_captions") and not has_audio
    if not row["duration"]:  # missing when the player response is blocked
        r = con.execute("SELECT MAX(duration) FROM candidates WHERE video_id=?", (vid,)).fetchone()
        row["duration"] = r[0] if r and r[0] else None
    dur = row["duration"] or 0
    if row["duration"] and not (C.MIN_DURATION_S <= dur <= C.MAX_DURATION_S):
        return rej(f"duration:{dur}")
    title_l = (row["title"] or "").lower()
    if any(w in title_l for w in C.REJECT_TITLE_WORDS):
        return rej("music_or_film_title")
    if any(c in (info.get("categories") or []) for c in C.REJECT_CATEGORIES):
        return rej("music_category")

    # Bot-checked player response: metadata (license/date) came from the watch
    # page, but captions and audio formats are missing. Keep it as a pre-screened
    # candidate to retry later instead of wrongly rejecting it for "no subs".
    if player_blocked:
        return {**row, "status": "blocked", "reason": "player_blocked(prescreen_ok)",
                "lang": cand_langs[0]}

    spoken = spoken_language(info)
    if not [k for k in (info.get("subtitles") or {}) if k != "live_chat"]:
        return rej("no_manual_subs")

    # Without a spoken-language signal, a video carrying manual tracks in many
    # languages is most likely a translated set; we cannot tell which is original.
    n_manual_langs = len({k.split("-")[0] for k in info.get("subtitles") or {} if k != "live_chat"})
    if spoken is None and n_manual_langs >= 3:
        return rej(f"translation_set_spoken_unknown({n_manual_langs}_langs)")

    vdir_tmp = C.DATA_DIR / "_probe" / vid
    last_reason = "no_manual_subs_in_language"
    for code in cand_langs:
        lang = C.LANGS[code]
        if spoken and spoken not in lang.spoken:
            last_reason = f"spoken_language_mismatch(spoken={spoken})"
            continue
        for key in manual_tracks(info, lang.sub_langs):
            try:
                path = fetch_track(ydl, info, key, vdir_tmp)
            except Exception as exc:  # noqa: BLE001
                if is_block_error(str(exc)):
                    raise Blocked(str(exc)[:200]) from exc
                last_reason = f"caption_download_error:{str(exc)[:80]}"
                continue
            cues = CAP.build_cues(CAP.parse_caption_file(path))
            if not cues:
                # timedtext can answer 200 with an empty body when YouTube wants a
                # PO token; that is a block, not a property of the captions
                shutil.rmtree(vdir_tmp, ignore_errors=True)
                return {**row, "status": "blocked", "lang": code,
                        "reason": f"caption_fetch_empty({key},{path.stat().st_size}B)"}
            q = CAP.caption_quality(cues, dur, lang)
            final = code
            if code in ("hi", "hi-en"):
                variant = CAP.classify_hindi_variant(q)
                if variant == "en":
                    last_reason = "english_track_on_hindi_query"
                    continue
                final = variant
                if final == "hi-en" and code == "hi":
                    q = CAP.caption_quality(cues, dur, C.LANGS["hi-en"])
            if code == "en-IN" and q["roman_hindi_share"] >= 0.08:
                final = "hi-en"
                q = CAP.caption_quality(cues, dur, C.LANGS["hi-en"])
            reason = CAP.quality_reject_reason(q, C.LANGS[final])
            if reason:
                last_reason = f"{key}:{reason}"
                continue
            n_same = con.execute(
                "SELECT COUNT(*) FROM videos WHERE status='accepted' AND lang=? AND channel_id=?",
                (final, row["channel_id"])).fetchone()[0]
            if n_same >= C.MAX_VIDEOS_PER_CHANNEL:
                return rej("channel_cap", final)
            vdir = C.DATA_DIR / final / vid
            vdir.mkdir(parents=True, exist_ok=True)
            dest = vdir / path.name
            shutil.move(str(path), dest)
            shutil.rmtree(vdir_tmp, ignore_errors=True)
            q["num_speakers_guess"] = guess_speakers(info, q)
            q["spoken_language_signal"] = spoken
            q["manual_track_languages"] = n_manual_langs
            (vdir / "info.json").write_text(json.dumps(trimmed_info(info), ensure_ascii=False, indent=1))
            return {**row, "status": "accepted", "reason": None, "lang": final,
                    "sub_lang": key, "sub_path": str(dest.relative_to(C.DATA_DIR)), "quality": q}
    shutil.rmtree(vdir_tmp, ignore_errors=True)
    return rej(last_reason)


def lang_progress(con, code: str) -> tuple[int, float]:
    r = con.execute("SELECT COUNT(*), COALESCE(SUM(json_extract(quality,'$.cue_seconds')),0) "
                    "FROM videos WHERE status='accepted' AND lang=?", (code,)).fetchone()
    return r[0], r[1]


def ordered_candidates(con, code: str, retry_blocked: bool = False) -> list[str]:
    """Tiered by language evidence, then filter freshness, round-robin over channels."""
    rows = con.execute(
        "SELECT c.video_id, c.channel_id, c.filter FROM candidates c "
        "LEFT JOIN videos v ON v.video_id = c.video_id "
        "WHERE c.lang=? AND (v.video_id IS NULL OR v.status='error' "
        "  OR (v.status='blocked' AND (? OR v.reason NOT LIKE 'player_blocked%')))",
        (code, int(retry_blocked))).fetchall()
    lang = C.LANGS[code]
    native = [s for s in lang.scripts if s is not C.LATIN]
    meta = {r["video_id"]: r for r in con.execute(
        "SELECT video_id, title, channel, duration, filter FROM candidates WHERE lang=?", (code,))}

    def native_hit(text: str | None) -> bool:
        return bool(native) and CAP.script_stats(text or "", native)["script_ratio"] > 0.3

    def tier(vid: str) -> int:
        # Search localises titles of foreign videos into the query language, so a
        # native-script title alone is weak evidence; channel names are not
        # localised. Seeds and native channel names go first.
        m = meta[vid]
        if m["filter"] == "seed":
            return 0
        if native_hit(m["channel"]):
            return 1
        if native_hit(m["title"]) or not native:
            return 2
        return 3

    rank = {"seed": 0, "cc_subs_month": 1, "cc_subs_year": 2, "cc_subs_newest": 3, "cc_subs": 4}
    rows = sorted(rows, key=lambda r: rank.get(r["filter"], 5))
    out, seen = [], set()
    for t in (0, 1, 2, 3):  # round-robin over channels within each tier
        by_ch: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            vid = r["video_id"]
            d = meta[vid]["duration"]
            if d and not (C.MIN_DURATION_S <= d <= C.MAX_DURATION_S):
                continue
            if tier(vid) == t:
                by_ch[r["channel_id"] or vid].append(vid)
        while any(by_ch.values()):
            for ch in list(by_ch):
                if by_ch[ch]:
                    v = by_ch[ch].pop(0)
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
    return out


def cmd_probe(args, con) -> None:
    ytd = yt_dlp()
    langs = args.langs or list(C.LANGS)
    blocks = 0
    with ytd.YoutubeDL(ydl_opts(args.player_clients, ignore_no_formats_error=True)) as ydl:
        for code in langs:
            n_ok, secs = lang_progress(con, code)
            todo = ordered_candidates(con, code, args.retry_blocked)
            log(f"[{code}] accepted={n_ok} captioned_min={secs/60:.1f} to_probe={len(todo)}")
            for i, vid in enumerate(todo[: args.limit or None]):
                n_ok, secs = lang_progress(con, code)
                if n_ok >= C.TARGET_VIDEOS or secs >= C.TARGET_SECONDS * 3:
                    break
                # a hi query may yield hi-en and vice versa: let both be tried
                cand_langs = [code] + (["hi-en"] if code == "hi" else ["hi"] if code == "hi-en" else [])
                try:
                    row = probe_one(ydl, con, vid, cand_langs)
                    blocks = 0
                except Blocked as exc:
                    blocks += 1
                    row = {"video_id": vid, "lang": code, "status": "blocked",
                           "reason": str(exc), "probed_at": DB.now(),
                           "url": f"https://www.youtube.com/watch?v={vid}"}
                    log(f"  BLOCKED {vid}: {str(exc)[:120]}")
                DB.upsert(con, "videos", row)
                log(f"  {row['status']:8s} {vid} [{row.get('lang')}] {row.get('upload_date')} "
                    f"{(row.get('title') or '')[:50]!r} {row.get('reason') or ''}")
                if blocks >= MAX_CONSECUTIVE_BLOCKS:
                    log(f"{blocks} consecutive bot checks: YouTube is blocking this IP; stopping.")
                    return
                polite_sleep(C.SLEEP_BETWEEN_VIDEOS)


# ------------------------------------------------------------------- fetch --

def disk_used() -> int:
    return sum(p.stat().st_size for p in C.DATA_DIR.rglob("*") if p.is_file())


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "pcm_s16le", str(dst)],
                   check=True)


def cut_chunks(full_wav: Path, chunks: list, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with wave.open(str(full_wav)) as w:
        sr, nframes = w.getframerate(), w.getnframes()
        assert sr == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        for i, ch in enumerate(chunks):
            a, b = int(ch.start * sr), min(nframes, int(ch.end * sr))
            w.setpos(a)
            frames = w.readframes(b - a)
            p = out_dir / f"{stem}_{i:04d}.wav"
            with wave.open(str(p), "wb") as o:
                o.setnchannels(1)
                o.setsampwidth(2)
                o.setframerate(sr)
                o.writeframes(frames)
            paths.append(p)
    return paths


def chunk_speakers(markers: int, video_guess: int) -> int:
    if video_guess <= 1:
        return 1
    # a turn marker inside a chunk usually means someone else speaks in it
    return min(video_guess, markers + 1) if markers else 1


def segment_video(con, v, full_wav: Path | None) -> int:
    """Compute cue-aligned chunks for an accepted video; cut audio if given."""
    vdir = C.DATA_DIR / v["lang"] / v["video_id"]
    cues = CAP.build_cues(CAP.parse_caption_file(C.DATA_DIR / v["sub_path"]))
    audio_dur = None
    if full_wav:
        with wave.open(str(full_wav)) as w:
            audio_dur = w.getnframes() / w.getframerate()
    chunks = CAP.select_spread(CAP.make_chunks(cues, audio_dur), C.PER_VIDEO_MAX_SECONDS)
    q = json.loads(v["quality"] or "{}")
    guess = q.get("num_speakers_guess", 1)
    stem = v["video_id"]
    paths = cut_chunks(full_wav, chunks, vdir / "chunks", stem) if full_wav else [None] * len(chunks)
    con.execute("DELETE FROM chunks WHERE video_id=?", (v["video_id"],))
    for i, (ch, p) in enumerate(zip(chunks, paths)):
        con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)", (
            f"ytcap_{v['lang']}_{v['video_id']}_{i:04d}", v["video_id"], v["lang"], i,
            round(ch.start, 3), round(ch.end, 3), round(ch.duration, 3), ch.text,
            chunk_speakers(ch.speakers_hint, guess),
            str(p.relative_to(C.DATA_DIR)) if p else None))
    con.commit()
    return len(chunks)


def cmd_fetch(args, con) -> None:
    ytd = yt_dlp()
    where = "status='accepted' AND (audio_status IS NULL OR audio_status!='done')"
    params: list = []
    if args.langs:
        where += f" AND lang IN ({','.join('?' * len(args.langs))})"
        params += args.langs
    todo = con.execute(f"SELECT * FROM videos WHERE {where} ORDER BY lang, upload_date DESC",
                       params).fetchall()
    log(f"{len(todo)} accepted videos need audio")
    blocks = 0
    for v in todo:
        if disk_used() > C.DISK_CAP_BYTES * 0.95:
            log("disk cap reached; stopping fetch")
            return
        tmp = Path(tempfile.mkdtemp(prefix="ytcap_", dir=C.DATA_DIR))
        try:
            opts = ydl_opts(args.player_clients, skip_download=False, format="bestaudio/best",
                            outtmpl=str(tmp / "src.%(ext)s"))
            with ytd.YoutubeDL(opts) as ydl:
                ydl.download([v["url"]])
            src = next(tmp.glob("src.*"))
            full = tmp / "full.wav"
            to_wav(src, full)
            n = segment_video(con, v, full)
            DB.upsert(con, "videos", {**dict(v), "audio_status": "done"})
            log(f"  done {v['lang']} {v['video_id']} chunks={n}")
            blocks = 0
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            status = "blocked" if is_block_error(msg) else f"error:{msg[:120]}"
            blocks = blocks + 1 if status == "blocked" else 0
            DB.upsert(con, "videos", {**dict(v), "audio_status": status})
            log(f"  {status[:140]} {v['video_id']}")
            if blocks >= MAX_CONSECUTIVE_BLOCKS:
                log("YouTube is blocking audio downloads from this IP; stopping.")
                return
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        polite_sleep(C.SLEEP_BETWEEN_VIDEOS)


# ------------------------------------------------------------------ export --

def cmd_export(args, con) -> None:
    videos = con.execute("SELECT * FROM videos WHERE status='accepted' ORDER BY lang, video_id").fetchall()
    # chunk boundaries can be computed without audio (useful for planning)
    for v in videos:
        if not con.execute("SELECT 1 FROM chunks WHERE video_id=?", (v["video_id"],)).fetchone():
            segment_video(con, v, None)
    manifest_dir = C.MANIFEST_PATH.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sources, n_rows = {}, 0
    with C.INDEX_PATH.open("w", encoding="utf-8") as idx, \
            C.MANIFEST_PATH.open("w", encoding="utf-8") as man:
        for v in videos:
            chunks = con.execute("SELECT * FROM chunks WHERE video_id=? ORDER BY idx",
                                 (v["video_id"],)).fetchall()
            have_audio = v["audio_status"] == "done"
            secs = sum(c["duration"] for c in chunks)
            sources[v["video_id"]] = {
                "title": v["title"], "channel": v["channel"], "url": v["url"],
                "upload_date": v["upload_date"], "license": v["license"],
                "language": v["lang"], "freshness": v["freshness"],
                "older_than_cutoff": v["freshness"] != "fresh",
                "caption_track": v["sub_lang"], "chunks": len(chunks) if have_audio else 0,
                "seconds_used": round(secs, 1) if have_audio else 0.0,
            }
            idx.write(json.dumps({
                **{k: v[k] for k in v.keys() if k != "quality"},
                "quality": json.loads(v["quality"] or "{}"),
                "chunks": [dict(c) for c in chunks],
            }, ensure_ascii=False) + "\n")
            if not have_audio:
                continue
            for c in chunks:
                if not c["audio_path"]:
                    continue
                rel = os.path.relpath(C.DATA_DIR / c["audio_path"], manifest_dir)
                man.write(json.dumps({
                    "id": c["id"], "audio_path": rel, "reference_text": c["reference_text"],
                    "language": v["lang"], "code_mixed": C.LANGS[v["lang"]].code_mixed,
                    "num_speakers": c["num_speakers"], "speaker_segments": None,
                    "source": f"youtube_captions:{v['video_id']}:manual",
                    "license": C.LICENSE_STR, "split": args.split,
                }, ensure_ascii=False) + "\n")
                n_rows += 1
    C.SOURCES_PATH.write_text(json.dumps(sources, ensure_ascii=False, indent=2))
    log(f"wrote {C.INDEX_PATH} ({len(videos)} videos), {C.SOURCES_PATH}, "
        f"{C.MANIFEST_PATH} ({n_rows} rows)")


def reason_key(r: str | None) -> str:
    """'ta-IN:script_mismatch(0.21)' -> 'script_mismatch'; 'license:x' -> 'license'."""
    r = r or ""
    r = re.sub(r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*:(?=[a-z_]+)", "", r) if not r.startswith(
        ("license:", "duration:", "too_old:", "caption_download_error:")) else r
    return re.split(r"[(:]", r, maxsplit=1)[0]


def cmd_status(args, con) -> None:
    print(f"{'lang':6s} {'cand':>5s} {'acc':>4s} {'audio':>5s} {'chunks':>6s} {'min':>6s} "
          f"{'fresh':>5s} {'older':>5s} {'rej':>4s} {'blk':>4s} {'pre_fresh':>9s} {'pre_older':>9s}")
    for code in C.LANGS:
        cand = con.execute("SELECT COUNT(*) FROM candidates WHERE lang=?", (code,)).fetchone()[0]
        st = Counter(r[0] for r in con.execute("SELECT status FROM videos WHERE lang=?", (code,)))
        audio = con.execute("SELECT COUNT(*) FROM videos WHERE lang=? AND status='accepted' "
                            "AND audio_status='done'", (code,)).fetchone()[0]
        fr = Counter(r[0] for r in con.execute(
            "SELECT freshness FROM videos WHERE lang=? AND status='accepted'", (code,)))
        ch = con.execute("SELECT COUNT(*), COALESCE(SUM(c.duration),0) FROM chunks c JOIN videos v "
                         "USING(video_id) WHERE c.lang=? AND v.audio_status='done'", (code,)).fetchone()
        # pre-screened: CC + 2026 + duration/title OK, but captions/audio were blocked
        pre = Counter(r[0] for r in con.execute(
            "SELECT freshness FROM videos WHERE lang=? AND status='blocked' "
            "AND reason LIKE 'player_blocked%'", (code,)))
        print(f"{code:6s} {cand:5d} {st['accepted']:4d} {audio:5d} {ch[0]:6d} {ch[1]/60:6.1f} "
              f"{fr['fresh']:5d} {fr['older']:5d} {st['rejected']:4d} {st['blocked']:4d} "
              f"{pre['fresh']:9d} {pre['older']:9d}")
    reasons = Counter()
    for (r,) in con.execute("SELECT reason FROM videos WHERE status='rejected'"):
        reasons[reason_key(r)] += 1
    print("rejection reasons:", dict(reasons.most_common()))
    print(f"disk used: {disk_used()/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "probe", "fetch", "export", "status", "all"):
        p = sub.add_parser(name)
        p.add_argument("--langs", nargs="*", choices=list(C.LANGS))
        p.add_argument("--player-clients", default=None,
                       help="yt-dlp youtube player_client list, e.g. 'default' or 'tv,web_safari'")
        p.add_argument("--per-query", type=int, default=40)
        p.add_argument("--filters", nargs="*", default=["cc_subs_year", "cc_subs_month"],
                       choices=list(C.SEARCH_FILTERS))
        p.add_argument("--refresh", action="store_true", help="re-run searches already done")
        p.add_argument("--retry-blocked", action="store_true",
                       help="re-probe pre-screened videos whose player response was blocked")
        p.add_argument("--no-seeds", action="store_true", help="skip seeds.json channel listings")
        p.add_argument("--limit", type=int, default=0, help="max videos to probe per language")
        p.add_argument("--split", default="private", help="manifest split value")
    args = ap.parse_args()
    con = DB.connect(C.DB_PATH)
    steps = {"discover": [cmd_discover], "probe": [cmd_probe], "fetch": [cmd_fetch],
             "export": [cmd_export], "status": [cmd_status],
             "all": [cmd_discover, cmd_probe, cmd_fetch, cmd_export, cmd_status]}[args.cmd]
    for step in steps:
        step(args, con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
