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


_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]+")


def romanize(text: str) -> str:
    """Script-agnostic projection: Devanagari runs -> Latin (IAST, diacritics
    stripped), everything else passed through. Used for the script-agnostic
    distance, never for the strict WER headline.

    "आई वांट टू स्टार्ट" -> "ai vant tu start", so script choice stops counting
    as an error while real content differences still do.
    """
    try:
        from indic_transliteration import sanscript
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "script-agnostic comparison needs indic-transliteration; "
            "run `uv sync`."
        ) from exc

    def _translit(m: re.Match) -> str:
        latin = sanscript.transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.IAST)
        # Anusvara is closer to "n" than "m" for this purpose (asking/अस्किंग).
        latin = latin.replace("ṁ", "n").replace("ṃ", "n")
        # Strip diacritics: NFD then drop combining marks.
        decomposed = unicodedata.normalize("NFD", latin)
        latin = "".join(c for c in decomposed if not unicodedata.combining(c))
        # Hindi schwa deletion: IAST keeps the inherent word-final "a"
        # ("स्टार्ट" -> "starta") that Hindi doesn't pronounce; drop it.
        return re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b", "", latin)

    return normalize_text(_DEVANAGARI_RE.sub(_translit, text))


def script_agnostic_cer(a: str, b: str) -> float:
    """Symmetric CER on romanized text; the consensus distance for code-mixed
    speech where providers legitimately differ on output script."""
    ra, rb = romanize(a), romanize(b)
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
