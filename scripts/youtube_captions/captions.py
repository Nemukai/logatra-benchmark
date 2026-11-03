"""Caption parsing, cleaning, quality checks and cue-aligned chunking.

Pure functions only (no network), so they can be unit-tested offline:

    uv run python scripts/youtube_captions/captions.py some.vtt --lang ta
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:  # imported as a module from ytcap.py or run directly
    from . import config as C  # type: ignore
except ImportError:  # pragma: no cover
    import config as C  # type: ignore


@dataclass
class Cue:
    start: float
    end: float
    text: str           # cleaned text ("" if nothing speech-like is left)
    raw: str            # original cue text
    music: bool = False  # cue is a music / no-speech marker (or lyrics)
    speaker_change: int = 0  # number of speaker-turn markers seen in the cue


# ---- parsing ----------------------------------------------------------------

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"bad timestamp {s!r}")
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def parse_vtt_or_srt(text: str) -> list[tuple[float, float, str]]:
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = [ln for ln in block.split("\n")]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = ln.split("-->", 1)
                try:
                    start, end = _ts(a), _ts(b.strip().split(" ")[0])
                except ValueError:
                    break
                body = "\n".join(lines[i + 1:]).strip()
                if body:
                    cues.append((start, end, body))
                break
    return cues


def parse_json3(text: str) -> list[tuple[float, float, str]]:
    data = json.loads(text)
    cues = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        body = "".join(s.get("utf8", "") for s in segs).strip()
        if not body:
            continue
        start = ev["tStartMs"] / 1000
        cues.append((start, start + ev.get("dDurationMs", 0) / 1000, body))
    return cues


def parse_caption_file(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json3":
        return parse_json3(text)
    return parse_vtt_or_srt(text)


# ---- cleaning ---------------------------------------------------------------

_MUSIC_RE = re.compile(
    r"[♪♫♬🎵🎶]|\[(?:music|music playing|background music|song|singing|"
    r"संगीत|म्यूजिक|म्यूज़िक|गाना|সঙ্গীত|সংগীত|இசை|సంగీతం|ಸಂಗೀತ|സംഗീതം|ਸੰਗੀਤ|સંગીત|ସଂଗୀତ|موسیقی)\]"
    r"|\((?:music|संगीत|இசை|music playing)\)",
    re.I,
)
_NONSPEECH_ONLY_RE = re.compile(
    r"^\s*[\[(](?:music|applause|laughter|silence|no audio|inaudible|noise|"
    r"तालियाँ|तालियां|हंसी|संगीत|கைதட்டல்|இசை|ചിരി|ਤਾੜੀਆਂ)[\])]\s*$",
    re.I,
)
_TAG_RE = re.compile(r"<[^>]+>")                 # <c>, <i>, <00:00:01.000>, <v Name>
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")  # [laughs], (applause)
_SPEAKER_RE = re.compile(r"^\s*(?:>>+|-\s+|–\s+|—\s+|[A-Z][\w .']{0,30}:\s|[^\s:]{1,20}:\s)")
_VOICE_RE = re.compile(r"<v\s+([^>]+)>")
_WS_RE = re.compile(r"\s+")


def clean_cue_text(raw: str) -> tuple[str, bool, int]:
    """Return (clean_text, is_music, n_speaker_markers)."""
    raw = html.unescape(raw)
    music = bool(_MUSIC_RE.search(raw)) or bool(_NONSPEECH_ONLY_RE.match(raw.strip()))
    speakers = len(_VOICE_RE.findall(raw))
    lines_out = []
    for ln in raw.split("\n"):
        ln = _TAG_RE.sub("", ln)
        if _SPEAKER_RE.match(ln):
            speakers += 1
            ln = _SPEAKER_RE.sub("", ln, count=1)
        ln = _BRACKET_RE.sub(" ", ln)
        ln = ln.replace(">>", " ")
        lines_out.append(ln)
    text = _WS_RE.sub(" ", " ".join(lines_out)).strip()
    text = text.strip("-–— ")
    return text, music, speakers


def build_cues(raw_cues: list[tuple[float, float, str]]) -> list[Cue]:
    """Clean, sort, drop exact rolling duplicates, clip overlaps."""
    cues: list[Cue] = []
    for start, end, body in sorted(raw_cues, key=lambda c: (c[0], c[1])):
        if end <= start:
            continue
        text, music, spk = clean_cue_text(body)
        if cues and text and text == cues[-1].text and start - cues[-1].end < 0.5:
            cues[-1].end = max(cues[-1].end, end)  # duplicated rolling cue
            continue
        cues.append(Cue(start, end, text, body, music, spk))
    # clip overlaps so chunk boundaries never cut a neighbouring cue
    for a, b in zip(cues, cues[1:]):
        if a.end > b.start:
            a.end = max(a.start + 0.05, b.start)
    return cues


# ---- quality metrics --------------------------------------------------------

def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for rs in ranges for lo, hi in rs)


def script_stats(text: str, scripts) -> dict:
    letters = [ch for ch in text if unicodedata.category(ch)[0] in "LM"]
    n = len(letters) or 1
    in_script = sum(1 for ch in letters if _in_ranges(ord(ch), scripts))
    latin = sum(1 for ch in letters if _in_ranges(ord(ch), (C.LATIN,)))
    deva = sum(1 for ch in letters if _in_ranges(ord(ch), (C.DEVANAGARI,)))
    return {
        "letters": len(letters),
        "script_ratio": in_script / n,
        "latin_ratio": latin / n,
        "devanagari_ratio": deva / n,
    }


# Romanised Hindi function words: their density separates Hinglish from English.
_HINDI_ROMAN = set(
    "hai hain tha thi the ka ki ke ko se mein me nahi nahin kya kyun kaise toh "
    "bhi aur ya yeh ye woh wo hum tum aap mera meri mere apna apni kuch bahut "
    "raha rahi rahe hoga hogi karna karte karta kar diya liya gaya jab tab "
    "agar lekin matlab accha acha haan bhai yaar".split()
)
_WORD_RE = re.compile(r"[a-z]+")
_PUNCT = "\"'.,!?;:()[]{}\u0964\u0965-–—…“”‘’"


def hinglish_stats(text: str) -> dict:
    words = [w.strip(_PUNCT) for w in text.lower().split()]
    words = [w for w in words if w and not w.isdigit()]
    latin_words = [w for w in words if w.isascii() and _WORD_RE.fullmatch(w)]
    n_lat = len(latin_words) or 1
    hindi_roman = sum(1 for w in latin_words if w in _HINDI_ROMAN)
    return {
        "latin_word_share": len(latin_words) / (len(words) or 1),
        "roman_hindi_share": hindi_roman / n_lat,
    }


def caption_quality(cues: list[Cue], duration: float, lang: C.Lang) -> dict:
    speech = [c for c in cues if c.text and not c.music]
    all_text = " ".join(c.text for c in speech)
    cue_time = sum(c.end - c.start for c in speech)
    n_words = len(all_text.split())
    q = {
        "n_cues": len(cues),
        "n_speech_cues": len(speech),
        "music_cue_share": (sum(1 for c in cues if c.music) / len(cues)) if cues else 1.0,
        "cue_seconds": round(cue_time, 1),
        "coverage": round(cue_time / duration, 3) if duration else 0.0,
        "words": n_words,
        "wps_captioned": round(n_words / cue_time, 2) if cue_time else 0.0,
        "wps_overall": round(n_words / duration, 2) if duration else 0.0,
        "speaker_markers": sum(c.speaker_change for c in cues),
    }
    q.update({k: round(v, 3) if isinstance(v, float) else v
              for k, v in script_stats(all_text, lang.scripts).items()})
    q.update({k: round(v, 3) if isinstance(v, float) else v
              for k, v in hinglish_stats(all_text).items()})
    # letters (incl. vowel signs) per second is script-neutral; word rates differ a
    # lot between agglutinative Dravidian languages and Hindi/English
    q["lps_captioned"] = round(q["letters"] / cue_time, 2) if cue_time else 0.0
    q["lps_overall"] = round(q["letters"] / duration, 2) if duration else 0.0
    return q


def quality_reject_reason(q: dict, lang: C.Lang) -> str | None:
    if q["n_speech_cues"] < 10:
        return "too_few_cues"
    if q["script_ratio"] < C.MIN_SCRIPT_RATIO:
        return f"script_mismatch({q['script_ratio']:.2f})"
    if lang.code == "en-IN" and q["roman_hindi_share"] > 0.06:
        return "romanised_hindi_in_english_track"
    if q["music_cue_share"] > C.MAX_MUSIC_CUE_SHARE:
        return f"music_heavy({q['music_cue_share']:.2f})"
    if q["coverage"] < C.MIN_COVERAGE:
        return f"low_coverage({q['coverage']:.2f})"
    if q["lps_captioned"] < C.MIN_LPS or q["lps_overall"] < C.MIN_OVERALL_LPS:
        return f"summary_like(lps={q['lps_captioned']},{q['lps_overall']})"
    if q["lps_captioned"] > C.MAX_LPS:
        return f"implausible_rate(lps={q['lps_captioned']})"
    return None


def classify_hindi_variant(q: dict) -> str:
    """For hi / hi-en candidates: decide the manifest language from caption text."""
    if q["latin_ratio"] > 0.6:  # romanised track
        return "hi-en" if q["roman_hindi_share"] >= 0.08 else "en"
    if q["devanagari_ratio"] > 0.5 and q["latin_word_share"] >= 0.08:
        return "hi-en"
    return "hi"


# ---- chunking ---------------------------------------------------------------

@dataclass
class Chunk:
    start: float
    end: float
    text: str
    n_cues: int
    speakers_hint: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def make_chunks(cues: list[Cue], audio_duration: float | None = None) -> list[Chunk]:
    """Greedily merge consecutive cues into chunks <= CHUNK_MAX_S.

    Never splits a cue. A chunk ends at a long inter-cue gap (uncaptioned audio
    would otherwise sit inside the chunk) or at a music/empty cue, which is
    dropped. Padding is added but never reaches into a neighbouring cue.
    """
    chunks: list[Chunk] = []
    cur: list[Cue] = []

    def flush():
        if not cur:
            return
        start, end = cur[0].start, cur[-1].end
        text = " ".join(c.text for c in cur).strip()
        if text and C.CHUNK_MIN_S <= end - start <= C.CHUNK_HARD_MAX_S:
            spk = sum(c.speaker_change for c in cur)
            chunks.append(Chunk(start, end, text, len(cur), spk))
        cur.clear()

    for c in cues:
        bad = c.music or not c.text
        if bad:
            flush()
            continue
        if c.end - c.start > C.CHUNK_HARD_MAX_S:
            flush()
            continue
        if cur and (c.start - cur[-1].end > C.MAX_INTRA_CHUNK_GAP_S
                    or c.end - cur[0].start > C.CHUNK_MAX_S):
            flush()
        cur.append(c)
    flush()

    # padding, bounded by neighbouring cues and the audio length
    bounds = [(c.start, c.end) for c in cues]
    for ch in chunks:
        prev_end = max((e for s, e in bounds if e <= ch.start + 1e-3), default=0.0)
        next_start = min((s for s, e in bounds if s >= ch.end - 1e-3), default=1e9)
        ch.start = max(0.0, prev_end, ch.start - C.PAD_S) if prev_end < ch.start else ch.start
        ch.end = min(next_start, ch.end + C.PAD_S) if next_start > ch.end else ch.end
        if audio_duration:
            ch.end = min(ch.end, audio_duration)
        if ch.duration > C.CHUNK_HARD_MAX_S:
            ch.end = ch.start + C.CHUNK_HARD_MAX_S
    return [ch for ch in chunks if ch.duration >= C.CHUNK_MIN_S]


def select_spread(chunks: list[Chunk], max_seconds: float) -> list[Chunk]:
    """Pick chunks evenly across the video until max_seconds is reached."""
    total = sum(c.duration for c in chunks)
    if total <= max_seconds:
        return chunks
    k = max(1, int(len(chunks) * max_seconds / total))
    step = len(chunks) / k
    picked, secs = [], 0.0
    for i in range(k):
        ch = chunks[int(i * step)]
        if secs + ch.duration > max_seconds:
            break
        picked.append(ch)
        secs += ch.duration
    return picked


if __name__ == "__main__":  # quick offline inspection
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("caption_file", type=Path)
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--duration", type=float, default=0.0)
    a = ap.parse_args()
    cs = build_cues(parse_caption_file(a.caption_file))
    dur = a.duration or (cs[-1].end if cs else 0)
    lg = C.LANGS[a.lang]
    q = caption_quality(cs, dur, lg)
    print(json.dumps(q, ensure_ascii=False, indent=1))
    print("reject:", quality_reject_reason(q, lg))
    for ch in make_chunks(cs)[:5]:
        print(f"{ch.start:8.2f} {ch.end:8.2f} {ch.text[:100]}")
