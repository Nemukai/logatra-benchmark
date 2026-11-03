"""WER/CER with an Indic-aware normalization pipeline, plus optional DER.

Normalization steps (applied to both reference and hypothesis before scoring):

1. Unicode NFC normalization (Indic scripts have many composed/decomposed
   ambiguities; NFC makes them comparable).
2. Lowercasing (affects Latin text only; Indic scripts are unicameral).
3. Punctuation stripping, including Indic punctuation: danda (U+0964),
   double danda (U+0965), Urdu full stop/comma/question mark, and the
   standard ASCII/Unicode set.
4. Digit normalization: Devanagari (and other Indic block) digits are mapped
   to ASCII 0-9 so "१२३" == "123".
5. Whitespace collapsing.

Script-agnostic comparison (implemented): for Hinglish the same word can be
validly written in Devanagari ("नमस्ते") or Latin ("namaste"), and plain WER
counts that as an error. ``romanize`` transliterates Devanagari runs to Latin
(IAST, diacritics stripped, schwa deletion) and ``script_agnostic_cer`` scores
on the romanized strings. The report shows it as a separate "script-agnostic"
column and never silently replaces strict WER with it.

Since v2 the projection covers every script in the benchmark (all Brahmic
blocks via indic_transliteration, plus Urdu's Perso-Arabic through a shared
consonant-skeleton key; see ``consonant_skeleton`` for why). The Devanagari
path is unchanged, so Hindi/Hinglish script-agnostic numbers are identical.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass

import jiwer

# Indic punctuation beyond the ASCII set.
_INDIC_PUNCT = (
    "।"  # devanagari danda ।
    "॥"  # devanagari double danda ॥
    "۔"  # arabic (urdu) full stop ۔
    "،"  # arabic comma ،
    "؟"  # arabic question mark ؟
    "؛"  # arabic semicolon ؛
    "‘’“”"  # curly quotes
    "–—"  # dashes
    "…"  # ellipsis
)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _INDIC_PUNCT)

# Digit blocks: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
# Kannada, Malayalam, plus Arabic-Indic (Urdu).
_DIGIT_BLOCKS = [
    0x0966,  # devanagari ०
    0x09E6,  # bengali
    0x0A66,  # gurmukhi
    0x0AE6,  # gujarati
    0x0B66,  # oriya
    0x0BE6,  # tamil
    0x0C66,  # telugu
    0x0CE6,  # kannada
    0x0D66,  # malayalam
    0x0660,  # arabic-indic (urdu)
    0x06F0,  # extended arabic-indic
]
_DIGIT_TABLE = {base + i: ord("0") + i for base in _DIGIT_BLOCKS for i in range(10)}

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Indic-aware normalization applied before WER/CER computation."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.translate(_DIGIT_TABLE)
    text = text.translate(_PUNCT_TABLE)
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass(slots=True)
class ASRScores:
    wer: float
    cer: float
    num_reference_words: int


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")

# Brahmic blocks other than Devanagari, each with the indic_transliteration
# scheme used to project it to Latin. Tamil uses TAMIL_SUB: the plain TAMIL
# scheme maps த/ட to voiced aspirates ("dh"), which no other script would
# produce for the same sound.
_BRAHMIC_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0980, 0x09FF, "bengali"),    # Bengali and Assamese
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil_subscripted"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
)
_OTHER_BRAHMIC_RE = re.compile(
    "|".join(f"[{chr(lo)}-{chr(hi)}]+" for lo, hi, _ in _BRAHMIC_BLOCKS)
)
# Letters the indic_transliteration schemes miss or render differently from
# their Devanagari counterpart (nukta letters are matched both precomposed and as
# base + nukta): Assamese/Odia "wa", and the flapped r (Devanagari ड़/ढ़ give "r").
_BRAHMIC_FIXUPS: tuple[tuple[str, str], ...] = (
    ("ৱ", "ব"), ("ୱ", "ଵ"),
    ("ড়", "র"), ("ঢ়", "রহ"),
    ("ଡ଼", "ର"), ("ଢ଼", "ରହ"),
)
_INDIC_LEFTOVER_RE = re.compile(r"[\u0900-\u0D7F]")

# Perso-Arabic (Urdu). Presentation forms are folded by NFKC first.
_PERSO_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")
_HAS_PERSO_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
# Letter-by-letter Urdu -> Latin. Urdu writes consonants plus long vowels
# only (short vowels are optional diacritics, almost never written), so this
# is a consonantal transcription; it is only ever compared through
# ``consonant_skeleton`` below, never read as a pronunciation.
_URDU_TO_LATIN = {
    "ا": "a", "آ": "a", "أ": "a", "إ": "i", "ٱ": "a",
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "c",
    "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r",
    "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t",
    "ظ": "z", "ع": "", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "ك": "k",
    "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "v", "ؤ": "v",
    "ہ": "h", "ه": "h", "ھ": "h", "ۃ": "h", "ۀ": "h", "ء": "", "ئ": "y",
    "ی": "y", "ي": "y", "ى": "y", "ے": "e", "ۓ": "e",
}
_ARABIC_DIACRITIC_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def _strip_marks(latin: str) -> str:
    decomposed = unicodedata.normalize("NFD", latin)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _devanagari_to_latin(run: str) -> str:
    from indic_transliteration import sanscript

    latin = sanscript.transliterate(run, sanscript.DEVANAGARI, sanscript.IAST)
    # Anusvara is closer to "n" than "m" for this purpose (asking/अस्किंग).
    latin = latin.replace("ṁ", "n").replace("ṃ", "n")
    # Strip diacritics: NFD then drop combining marks.
    latin = _strip_marks(latin)
    # Hindi schwa deletion: IAST keeps the inherent word-final "a"
    # ("स्टार्ट" -> "starta") that Hindi doesn't pronounce; drop it.
    return re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b", "", latin)


def _brahmic_to_latin(run: str) -> str:
    from indic_transliteration import sanscript

    cp = ord(run[0])
    scheme = next(s for lo, hi, s in _BRAHMIC_BLOCKS if lo <= cp <= hi)
    for src, dst in _BRAHMIC_FIXUPS:
        # Nukta letters occur both precomposed and as base + nukta.
        run = run.replace(src, dst).replace(unicodedata.normalize("NFD", src), dst)
    latin = sanscript.transliterate(run, scheme, sanscript.IAST)
    latin = latin.replace("ṁ", "n").replace("ṃ", "n")
    latin = _strip_marks(latin)
    # Anything the scheme left untransliterated (rare signs) is dropped so it
    # cannot masquerade as a word-level mismatch.
    latin = _INDIC_LEFTOVER_RE.sub("", latin)
    # Same word-final schwa rule as Devanagari, so the same word written in
    # two Brahmic scripts projects to the same Latin string.
    return re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b", "", latin)


def _perso_arabic_to_latin(run: str) -> str:
    run = _ARABIC_DIACRITIC_RE.sub("", run)
    return "".join(_URDU_TO_LATIN.get(c, c) for c in run)


def romanize(text: str) -> str:
    """Script-agnostic projection: every Indic script -> Latin, everything
    else passed through. Used for the script-agnostic distance, never for the
    strict WER headline.

    - Devanagari: IAST, diacritics stripped, anusvara -> n, word-final schwa
      deleted. "आई वांट टू स्टार्ट" -> "ai vant tu start", so script choice
      stops counting as an error while real content differences still do.
      This path is byte-for-byte the original (v1) projection.
    - Bengali/Assamese, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada,
      Malayalam: the same treatment through indic_transliteration's scheme
      for that block, so Odia read speech transcribed in Devanagari (which
      Whisper does) compares on content rather than scoring 0.
    - Perso-Arabic (Urdu): a consonantal letter mapping; see
      ``consonant_skeleton`` for how it is compared.
    """
    try:
        from indic_transliteration import sanscript  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "script-agnostic comparison needs indic-transliteration; "
            "run `uv sync`."
        ) from exc

    text = _DEVANAGARI_RE.sub(lambda m: _devanagari_to_latin(m.group(0)), text)
    text = _OTHER_BRAHMIC_RE.sub(lambda m: _brahmic_to_latin(m.group(0)), text)
    if _HAS_PERSO_ARABIC_RE.search(text):
        text = unicodedata.normalize("NFKC", text)
        text = _PERSO_ARABIC_RE.sub(lambda m: _perso_arabic_to_latin(m.group(0)), text)
    return normalize_text(text)


# Consonant-skeleton key classes. Applied to already-romanized text.
_SKELETON_TABLE = str.maketrans({"q": "k", "f": "p", "z": "j", "x": "k"})
_SKELETON_DROP_RE = re.compile(r"[aeiouhvwy]")
_REPEAT_RE = re.compile(r"(.)\1+")


def consonant_skeleton(romanized: str) -> str:
    """Lossy phonetic key shared by Urdu and every Brahmic script.

    Why a skeleton instead of a better Urdu transliterator: Urdu is an abjad.
    Short vowels (zabar/zer/pesh) are almost never written, و and ی stand
    for both consonants (v, y) and long vowels (u/o, i/e), and ہ is both /h/
    and a word-final vowel marker. "دور" is daur, "کہ" is ki. No
    deterministic transliteration recovers the vowels a Devanagari writer
    spells out, so any vowel-bearing comparison would charge an error to
    nearly every word for script choice alone, which is exactly what the
    script-agnostic column exists to forgive. The key therefore keeps only
    what both writing systems reliably encode, the consonant frame:

    - drop vowels, h (aspiration and Urdu's vowel-h) and the semivowels v/w/y;
    - merge sounds Urdu spells with Perso-Arabic letters that Hindi writers
      often spell without nukta: q~k, f~p(h), z~j; retroflex/dental and
      s/sh distinctions are already gone after diacritic stripping;
    - collapse doubled letters (Urdu rarely writes the shadda for gemination);
    - a word with no consonants left ("है", "ہے", "या") becomes "_" so word
      counts, and therefore WER denominators, are preserved.

    It is deliberately lenient (two different words that share a consonant
    frame match), so it is only used when a Perso-Arabic string is on either
    side of the comparison; see ``script_agnostic_pair``. Strict WER is never
    affected.
    """
    out = []
    for w in romanized.split():
        # Leftover Indic signs the transliterator passed through (e.g. the
        # Devanagari candra-o in "मॉडर्न") carry no consonant information.
        w = _INDIC_LEFTOVER_RE.sub("", w)
        k = _SKELETON_DROP_RE.sub("", w.translate(_SKELETON_TABLE))
        k = _REPEAT_RE.sub(r"\1", k)
        out.append(k or "_")
    return " ".join(out)


def script_agnostic_pair(a: str, b: str) -> tuple[str, str]:
    """Project two transcripts onto a shared script-agnostic key.

    Both are romanized; if either contains Perso-Arabic, both are further
    reduced to ``consonant_skeleton`` so Urdu-vs-Devanagari (or Urdu-vs-Urdu,
    for consistency within the Urdu language cell) compares like with like.
    Pairs without Perso-Arabic are compared on the romanized strings, which
    keeps Hindi/Hinglish script-agnostic numbers identical to v1.
    """
    ra, rb = romanize(a), romanize(b)
    if _HAS_PERSO_ARABIC_RE.search(a) or _HAS_PERSO_ARABIC_RE.search(b):
        ra, rb = consonant_skeleton(ra), consonant_skeleton(rb)
    return ra, rb


def script_agnostic_wer(reference: str, hypothesis: str) -> float:
    """WER on the script-agnostic key (see ``script_agnostic_pair``)."""
    r, h = script_agnostic_pair(reference, hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(r, h)


def script_agnostic_cer(a: str, b: str) -> float:
    """Symmetric CER on the script-agnostic key; the consensus distance for
    code-mixed speech where providers legitimately differ on output script."""
    ra, rb = script_agnostic_pair(a, b)
    if not ra and not rb:
        return 0.0
    if not ra or not rb:
        return 1.0
    return (jiwer.cer(ra, rb) + jiwer.cer(rb, ra)) / 2


def score_asr(reference: str, hypothesis: str) -> ASRScores:
    """Compute WER and CER on normalized text.

    Edge case: an empty (normalized) reference cannot be scored by jiwer;
    we define WER/CER as 0.0 if the hypothesis is also empty, else 1.0.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        miss = 0.0 if not hyp else 1.0
        return ASRScores(wer=miss, cer=miss, num_reference_words=0)
    return ASRScores(
        wer=jiwer.wer(ref, hyp),
        cer=jiwer.cer(ref, hyp),
        num_reference_words=len(ref.split()),
    )


def score_der(
    reference_segments: list,  # list[manifest.SpeakerSegment]
    hypothesis_turns: list,    # list[adapters.SpeakerTurn]
    collar: float = 0.25,
) -> float:
    """Diarization Error Rate via pyannote.metrics (optional dependency).

    Install with: uv sync --extra diarization
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as exc:
        raise ImportError(
            "DER scoring needs pyannote.metrics; install with "
            "`uv sync --extra diarization`."
        ) from exc

    reference = Annotation()
    for seg in reference_segments:
        reference[Segment(seg.start, seg.end)] = seg.speaker
    hypothesis = Annotation()
    for turn in hypothesis_turns:
        hypothesis[Segment(turn.start, turn.end)] = turn.speaker

    metric = DiarizationErrorRate(collar=collar)
    return float(metric(reference, hypothesis))
