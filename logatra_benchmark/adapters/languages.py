"""Per-provider language-code mapping for the Logatra language set.

Canonical (manifest) codes:

    core:     hi, hi-en (Hinglish code-mix), en-IN, bn, ta, te, mr, gu, kn,
              ml, pa, or, as, ur
    planned:  kok, mai, ne, sd, ks, doi, mni, brx, sat, sa

Manifest quirks normalized by `canonical()`: "en" -> "en-IN", "od" -> "or",
and ("hi", code_mixed=True) -> "hi-en" (the held-out split labels Hinglish as
language "hi" with code_mixed true).

Each provider table maps canonical code -> provider code. Special values:

    AUTO     run without a language hint (provider auto-detects). Used where
             the provider has no code for the language but historically ran
             it anyway (kept for comparability with cached results), or for
             code-mixed audio where auto-detect is the documented best mode.
    missing  unsupported: the runner skips the sample with a recorded reason;
             it is not an error and costs nothing.

Sources (all checked 2026-11-03):
    Deepgram   https://developers.deepgram.com/docs/models-languages-overview
               https://developers.deepgram.com/docs/nova-3-language-prompting
    Google     https://cloud.google.com/speech-to-text/docs/models/chirp-3
    ElevenLabs https://elevenlabs.io/docs/overview/capabilities/speech-to-text
    AssemblyAI https://www.assemblyai.com/docs/pre-recorded-audio/supported-languages
               https://www.assemblyai.com/docs/pre-recorded-audio/universal-3-5-pro
    OpenAI     https://developers.openai.com/api/docs/guides/speech-to-text
               (whisper-1 hint list: github.com/openai/whisper tokenizer)
    Sarvam     https://docs.sarvam.ai/api/api-guides-tutorials/models/saaras
               https://docs.sarvam.ai/api/getting-started/models/saarika
    Azure      https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
    IndicConformer  model card ai4bharat/indic-conformer-600m-multilingual
               (gated on HF; list from AI4Bharat's published 22-language set)
"""

from __future__ import annotations

from .base import Unsupported

AUTO = ""  # run without a hint

CORE_LANGUAGES = ("hi", "hi-en", "en-IN", "bn", "ta", "te", "mr", "gu", "kn",
                  "ml", "pa", "or", "as", "ur")
PLANNED_LANGUAGES = ("kok", "mai", "ne", "sd", "ks", "doi", "mni", "brx", "sat", "sa")
ALL_LANGUAGES = CORE_LANGUAGES + PLANNED_LANGUAGES

_ALIASES = {"en": "en-IN", "en-in": "en-IN", "od": "or", "hinglish": "hi-en",
            "hi-latn": "hi-en"}


def canonical(language: str | None, code_mixed: bool = False) -> str | None:
    if not language:
        return None
    lang = language.strip()
    lang = _ALIASES.get(lang.lower(), lang)
    # Strip region suffixes like "ta-IN" -> "ta" (but keep en-IN, hi-en).
    if lang not in ALL_LANGUAGES and "-" in lang:
        base = lang.split("-")[0].lower()
        if base in ALL_LANGUAGES:
            lang = base
    if lang == "hi" and code_mixed:
        return "hi-en"
    return lang


# --- Deepgram Nova-3 -------------------------------------------------------
# Monolingual codes; "multi" is Nova-3 multilingual (code-switching), which is
# billed at the multilingual rate. Nova-3 has no Malayalam or Odia.
DEEPGRAM = {
    "hi": "hi", "hi-en": "multi", "en-IN": "en-IN", "bn": "bn", "ta": "ta",
    "te": "te", "mr": "mr", "gu": "gu", "kn": "kn", "pa": "pa", "ur": "ur",
    "as": "as", "ne": "ne",
}
# language_hint values sent with language=multi (only valid with multi).
DEEPGRAM_MULTI_HINTS = {"hi-en": ("hi", "en")}

# --- Google Cloud STT v2, chirp_3 -------------------------------------------
# en-IN and hi-IN are GA; the other Indic locales are Preview. No Urdu.
GOOGLE_CHIRP3 = {
    "hi": "hi-IN", "hi-en": "hi-IN", "en-IN": "en-IN", "bn": "bn-IN",
    "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "pa": "pa-Guru-IN", "or": "or-IN", "as": "as-IN",
    "ne": "ne-NP",
}

# --- ElevenLabs Scribe v2 (ISO 639-3) ---------------------------------------
ELEVENLABS = {
    "hi": "hin", "hi-en": AUTO, "en-IN": "eng", "bn": "ben", "ta": "tam",
    "te": "tel", "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal",
    "pa": "pan", "or": "ori", "as": "asm", "ur": "urd", "ne": "nep",
    "sd": "snd",
}

# --- AssemblyAI -------------------------------------------------------------
# Hindi (and English) run on Universal-3.5 Pro (native code-switching);
# everything else routes to Universal-2. No Odia.
ASSEMBLYAI = {
    "hi": "hi", "hi-en": "hi", "en-IN": "en", "bn": "bn", "ta": "ta",
    "te": "te", "mr": "mr", "gu": "gu", "kn": "kn", "ml": "ml", "pa": "pa",
    "ur": "ur", "as": "as", "ne": "ne", "sd": "sd", "sa": "sa",
}

# --- OpenAI (whisper-1, gpt-4o-*-transcribe, gpt-transcribe) ----------------
# ISO 639-1 hints. Whisper has no Odia: run it unhinted (as the cached
# results were produced). The adapter also retries unhinted on
# `unsupported_language`.
OPENAI = {
    "hi": "hi", "hi-en": "hi", "en-IN": "en", "bn": "bn", "ta": "ta",
    "te": "te", "mr": "mr", "gu": "gu", "kn": "kn", "ml": "ml", "pa": "pa",
    "or": AUTO, "as": "as", "ur": "ur", "ne": "ne", "sd": "sd", "sa": "sa",
}
# gpt-transcribe takes a list (`languages[]`); code-mix gets both.
OPENAI_MULTI = {"hi-en": ("hi", "en")}

# --- Sarvam -----------------------------------------------------------------
# saaras:v3 / saaras:v4: 22 Indic + English. hi-en -> "unknown" (auto) is the
# historical behaviour of this repo's Sarvam runs.
SARVAM_SAARAS = {
    "hi": "hi-IN", "hi-en": "unknown", "en-IN": "en-IN", "bn": "bn-IN",
    "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "pa": "pa-IN", "or": "od-IN", "as": "as-IN", "ur": "ur-IN",
    "kok": "kok-IN", "mai": "mai-IN", "ne": "ne-IN", "sd": "sd-IN",
    "ks": "ks-IN", "doi": "doi-IN", "mni": "mni-IN", "brx": "brx-IN",
    "sat": "sat-IN", "sa": "sa-IN",
}
# saarika:v2.5 (deprecated): 10 Indic + en-IN. as/ur historically ran with
# "unknown" (auto); kept for comparability. The planned languages are skipped.
SARVAM_SAARIKA = {
    "hi": "hi-IN", "hi-en": "unknown", "en-IN": "en-IN", "bn": "bn-IN",
    "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "pa": "pa-IN", "or": "od-IN", "as": "unknown",
    "ur": "unknown",
}

# --- Azure AI Speech, fast transcription ------------------------------------
AZURE = {
    "hi": "hi-IN", "hi-en": "hi-IN", "en-IN": "en-IN", "bn": "bn-IN",
    "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "pa": "pa-IN", "or": "or-IN", "as": "as-IN", "ur": "ur-IN",
    "ne": "ne-NP",
}
# Multiple candidate locales enable language identification.
AZURE_MULTI = {"hi-en": ("hi-IN", "en-IN")}

# --- AI4Bharat IndicConformer (open weights, local) -------------------------
INDICCONFORMER = {
    "hi": "hi", "hi-en": "hi", "bn": "bn", "ta": "ta", "te": "te", "mr": "mr",
    "gu": "gu", "kn": "kn", "ml": "ml", "pa": "pa", "or": "or", "as": "as",
    "ur": "ur", "kok": "kok", "mai": "mai", "ne": "ne", "sd": "sd", "ks": "ks",
    "doi": "doi", "mni": "mni", "brx": "brx", "sat": "sat", "sa": "sa",
}

TABLES: dict[str, dict[str, str]] = {
    "deepgram": DEEPGRAM,
    "google": GOOGLE_CHIRP3,
    "elevenlabs": ELEVENLABS,
    "assemblyai": ASSEMBLYAI,
    "openai": OPENAI,
    "sarvam-saaras": SARVAM_SAARAS,
    "sarvam-saarika": SARVAM_SAARIKA,
    "azure": AZURE,
    "indicconformer": INDICCONFORMER,
}


def lookup(table: str, language: str | None, code_mixed: bool = False) -> str | None:
    """Return the provider code (AUTO for no hint), or None if unsupported.

    A missing language (None/"") is always runnable with AUTO.
    """
    lang = canonical(language, code_mixed)
    if lang is None:
        return AUTO
    return TABLES[table].get(lang)


def unsupported_reason(table: str, language: str | None, code_mixed: bool = False) -> str | None:
    if lookup(table, language, code_mixed) is None:
        lang = canonical(language, code_mixed)
        return f"language {lang!r} not supported by {table} (see adapters/languages.py)"
    return None


def resolve(table: str, language: str | None, code_mixed: bool = False) -> str:
    """Like `lookup`, but raises `Unsupported` instead of returning None."""
    code = lookup(table, language, code_mixed)
    if code is None:
        raise Unsupported(unsupported_reason(table, language, code_mixed))
    return code


def coverage_matrix(languages=ALL_LANGUAGES) -> dict[str, dict[str, str]]:
    """{table: {lang: code | 'auto' | '-'}} for docs and `estimate` output."""
    out: dict[str, dict[str, str]] = {}
    for name, table in TABLES.items():
        row = {}
        for lang in languages:
            code = table.get(lang)
            row[lang] = "-" if code is None else ("auto" if code == AUTO else code)
        out[name] = row
    return out
