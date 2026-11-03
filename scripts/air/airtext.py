"""PDF script extraction and text-quality classification for AIR bulletins.

Two extractors are tried (PyMuPDF and poppler's ``pdftotext``) and the one
with the better quality score is kept. Both read the PDF's ToUnicode maps, so
neither can repair a broken font encoding; we only *detect* problems:

text_status values
  ok                        expected script dominates, pre-base vowel signs present
  unicode_glyphmap_suspect  expected Unicode script, but pre-base vowel signs
                            (e.g. Devanagari i-matra, Tamil e/ai signs) are
                            missing or nearly so: the font's ToUnicode map
                            mis-maps reordered glyphs (Word "Utsaah", "Vrinda",
                            "Uni Ila.Sundaram" ...). Text is readable but has
                            systematic character errors; use with care.
  legacy_font_unconverted   expected script absent, Latin-1/ASCII junk instead
                            (non-Unicode fonts such as AkrutiAsmAnand); PDF kept.
  no_text_layer             (almost) no extractable text: scanned image PDF.
  wrong_script              text is clean Unicode but in an unexpected script.

Needs pymupdf (``uv run --with pymupdf``); pdftotext is optional.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_RANGES = [
    ("Deva", 0x0900, 0x097F), ("Beng", 0x0980, 0x09FF), ("Guru", 0x0A00, 0x0A7F),
    ("Gujr", 0x0A80, 0x0AFF), ("Orya", 0x0B00, 0x0B7F), ("Taml", 0x0B80, 0x0BFF),
    ("Telu", 0x0C00, 0x0C7F), ("Knda", 0x0C80, 0x0CFF), ("Mlym", 0x0D00, 0x0D7F),
    ("Tibt", 0x0F00, 0x0FFF), ("Lepc", 0x1C00, 0x1C4F), ("Olck", 0x1C50, 0x1C7F),
    ("Mtei", 0xABC0, 0xABFF), ("Arab", 0x0600, 0x06FF), ("Arab", 0x0750, 0x077F),
    ("Arab", 0xFB50, 0xFDFF), ("Arab", 0xFE70, 0xFEFF),
    ("Latn", 0x0041, 0x005A), ("Latn", 0x0061, 0x007A),
    ("Latin1", 0x00C0, 0x024F),  # accented Latin: hallmark of legacy-font junk
    # code points that never belong in an Indic script: misencoded glyphs
    ("Junk", 0x0370, 0x03FF), ("Junk", 0x1F00, 0x1FFF), ("Junk", 0xE000, 0xEFFF), ("Junk", 0xF100, 0xF8FF),
    ("Junk", 0x0400, 0x04FF), ("Junk", 0xFFFD, 0xFFFD),
]
# Pre-base (left-side) dependent vowel signs; Word PDFs that lose them produce
# systematically wrong text. Typical clean frequency is 1-4% of script chars.
PRE_BASE = {
    "Deva": "ि", "Beng": "িেৈ", "Guru": "ਿ", "Gujr": "િ",
    "Orya": "େୈ", "Taml": "ெேை", "Mlym": "െേൈ",
}


_IND_VOWELS = {"A", "AA", "I", "II", "U", "UU", "VOCALIC R", "VOCALIC RR", "VOCALIC L",
               "VOCALIC LL", "E", "EE", "AI", "O", "OO", "AU", "SHORT E", "SHORT O",
               "CANDRA E", "CANDRA O", "SHORT A"}


def _is_consonant(ch: str) -> bool:
    try:
        n = unicodedata.name(ch)
    except ValueError:
        return False
    if "NUKTA" in n:
        return True
    return " LETTER " in n and n.split(" LETTER ", 1)[1] not in _IND_VOWELS


def orphan_sign_rate(text: str) -> float | None:
    """Share of Indic vowel signs/viramas NOT preceded by a consonant.

    Clean Unicode text is near 0; PDFs whose fonts map reordered glyphs to
    wrong code points show 2-30%.
    """
    bad = n = 0
    for prev, ch in zip(" " + text, text):
        if not ("\u0900" <= ch <= "\u0DFF"):
            continue
        nm = unicodedata.name(ch, "")
        if "VOWEL SIGN" in nm or nm.endswith("SIGN VIRAMA"):
            n += 1
            bad += not _is_consonant(prev)
    return bad / n if n >= 50 else None


ORPHAN_MAX = 0.01


def script_of(ch: str) -> str | None:
    o = ord(ch)
    for name, lo, hi in SCRIPT_RANGES:
        if lo <= o <= hi:
            return name
    return None


def script_profile(text: str) -> Counter:
    return Counter(s for s in map(script_of, text) if s)


def extract_pymupdf(pdf: Path) -> str:
    import pymupdf
    with pymupdf.open(pdf) as d:
        return "\n".join(p.get_text() for p in d)


def extract_pdftotext(pdf: Path) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        return ""
    r = subprocess.run([exe, "-enc", "UTF-8", str(pdf), "-"], capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", errors="replace")


def pdf_meta(pdf: Path) -> dict:
    import pymupdf
    with pymupdf.open(pdf) as d:
        m = d.metadata or {}
        fonts = sorted({f[3] for p in d for f in p.get_fonts()})
        return {"pages": d.page_count, "creation_date": m.get("creationDate") or None,
                "mod_date": m.get("modDate") or None, "producer": m.get("producer") or None,
                "fonts": [re.sub(r"^[A-Z]{6}\+", "", f) for f in fonts]}


def classify(text: str, scripts: list[str]) -> dict:
    prof = script_profile(text)
    letters = sum(prof.values())
    expected = scripts[0]
    exp_n = sum(prof[s] for s in scripts)
    out = {"script_counts": dict(prof.most_common(4)), "letters": letters}
    if letters < 200:
        return {**out, "status": "no_text_layer", "score": 0.0}
    dominant = prof.most_common(1)[0][0]
    exp_share = exp_n / letters
    if expected == "Latn":
        status = "ok" if prof["Latn"] / letters > 0.8 and prof["Latin1"] / letters < 0.05 else (
            "wrong_script" if dominant not in ("Latn", "Latin1") else "legacy_font_unconverted")
        return {**out, "status": status, "expected_share": round(exp_share, 3),
                "score": exp_share if status == "ok" else 0.1}
    # Indic / other non-Latin expected. Legacy fonts give mostly ASCII + Latin-1.
    if exp_share < 0.4:
        status = "wrong_script" if dominant not in ("Latn", "Latin1") else "legacy_font_unconverted"
        return {**out, "status": status, "expected_share": round(exp_share, 3), "score": exp_share}
    main = max(scripts, key=lambda s: prof[s])
    pb = PRE_BASE.get(main)
    pb_rate = None
    status = "ok"
    if pb:
        pb_rate = sum(text.count(c) for c in pb) / max(prof[main], 1)
        if pb_rate < 0.008:
            status = "unicode_glyphmap_suspect"
    orphan = orphan_sign_rate(text)
    if orphan is not None and orphan > ORPHAN_MAX:
        status = "unicode_glyphmap_suspect"
    junk = prof["Junk"] / letters
    if junk > 0.005:
        status = "unicode_glyphmap_suspect"
    score = exp_share + (0.5 if status == "ok" else 0) - (orphan or 0)
    return {**out, "status": status, "expected_share": round(exp_share, 3),
            "prebase_rate": None if pb_rate is None else round(pb_rate, 4),
            "orphan_sign_rate": None if orphan is None else round(orphan, 4),
            "junk_rate": round(junk, 4), "score": score - junk}


def clean_text(text: str) -> str:
    """NFC, drop control/format chars (keeping ZWJ/ZWNJ) and rule lines, collapse whitespace.

    Private-use code points are kept on purpose: they mark misencoded glyphs
    and must stay visible to classify().
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub("[\uf000-\uf0ff]", "\u2022", text)  # Symbol/Wingdings bullets
    text = "".join(ch for ch in text if ch in "\n\t\u200c\u200d"
                   or unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Cn"))
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"[ \t ]+", " ", ln).strip()
        if not ln or re.fullmatch(r"[\W_]+", ln):
            continue
        lines.append(ln)
    return "\n".join(lines)


def extract_best(pdf: Path, scripts: list[str]) -> dict:
    cands = []
    for name, fn in (("pymupdf", extract_pymupdf), ("pdftotext", extract_pdftotext)):
        try:
            t = clean_text(fn(pdf))
        except Exception as e:  # noqa: BLE001
            t = ""
            print(f"  ! {name} failed on {pdf.name}: {e}")
        if t:
            cands.append((classify(t, scripts), name, t))
    if not cands:
        return {"text": "", "extractor": None, "status": "no_text_layer", "quality": {}}
    q, name, t = max(cands, key=lambda c: (c[0]["score"], len(c[2])))
    return {"text": t, "extractor": name, "status": q["status"], "quality": q}


# ---- date found in the script header (pairing evidence) -------------------
_DIGITS = str.maketrans({chr(z + i): str(i) for z in
                         (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66,
                          0x06F0, 0x0660) for i in range(10)})
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def header_dates(text: str, head_chars: int = 800) -> list[str]:
    """ISO dates mentioned near the top of the script (any digit system)."""
    h = text[:head_chars].translate(_DIGITS)
    found = []
    for d, m, y in re.findall(r"\b(\d{1,2})\s*[-./ ]\s*(\d{1,2})\s*[-./ ]\s*(20\d{2}|\d{2})\b", h):
        y = int(y) + (2000 if len(y) == 2 else 0)
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31:
            found.append(f"{y:04d}-{int(m):02d}-{int(d):02d}")
    for d, mon, y in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3})[a-z]*[,.]?\s+(20\d{2})", h):
        if mon.lower() in _MONTHS:
            found.append(f"{int(y):04d}-{_MONTHS[mon.lower()]:02d}-{int(d):02d}")
    for mon, d, y in re.findall(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})[,.]?\s+(20\d{2})", h):
        if mon.lower() in _MONTHS:
            found.append(f"{int(y):04d}-{_MONTHS[mon.lower()]:02d}-{int(d):02d}")
    if not found:  # month spelled in an Indic language: keep day+year only
        for d, y in re.findall(r"\b(\d{1,2})\s+\S{2,15}\s*,?\s+(20\d{2})\b", h):
            found.append(f"{int(y):04d}-??-{int(d):02d}")
    return list(dict.fromkeys(found))


def date_matches(found: list[str], iso: str) -> bool | None:
    """True/False if the header carries a date that does/doesn't match; None if none."""
    if not found:
        return None
    return any(f == iso or (f[5:7] == "??" and f[:4] == iso[:4] and f[8:] == iso[8:]) for f in found)
