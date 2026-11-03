"""Shared configuration for the YouTube manual-caption collector.

Everything this pipeline writes lives under data/private/ (gitignored):

    data/private/youtube_captions/
        yt.sqlite                       state: candidates, videos, chunks
        index.jsonl                     one row per accepted video + its chunks
        sources.json                    attribution (title/channel/url/date/license)
        {lang}/{video_id}/info.json     trimmed yt-dlp metadata
        {lang}/{video_id}/captions.*    original manual caption file (untouched)
        {lang}/{video_id}/chunks/*.wav  16 kHz mono s16 cue-aligned chunks
    data/private/manifests/youtube_captions.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "private" / "youtube_captions"
DB_PATH = DATA_DIR / "yt.sqlite"
INDEX_PATH = DATA_DIR / "index.jsonl"
SOURCES_PATH = DATA_DIR / "sources.json"
MANIFEST_PATH = ROOT / "data" / "private" / "manifests" / "youtube_captions.jsonl"

# ---- policy knobs -----------------------------------------------------------
FRESH_AFTER = "20260901"      # after Sarvam saaras v4; preferred
OLDEST_OK = "20260101"        # accepted but flagged "older"
MIN_DURATION_S = 120
MAX_DURATION_S = 2 * 3600
MAX_VIDEOS_PER_CHANNEL = 2    # per language, for speaker/channel diversity
TARGET_VIDEOS = 15            # per language; stop probing once reached
TARGET_SECONDS = 60 * 60      # per language usable-speech target
PER_VIDEO_MAX_SECONDS = 6 * 60  # cap chunks taken per video (spread evenly)
DISK_CAP_BYTES = 6 * 1024**3

# caption quality
MIN_SCRIPT_RATIO = 0.60       # share of letters in the expected script(s)
MIN_COVERAGE = 0.35           # caption-cue time / video duration
# Rates are in letters (Unicode L*/M*, so vowel signs count) per second. Normal
# conversational speech captions run ~9-18 letters/s in every script here; a
# summary or partial translation falls far below.
MIN_LPS = 5.0                 # per captioned second
MAX_LPS = 35.0                # above this cues are probably mistimed
MIN_OVERALL_LPS = 2.5         # per second of the whole video
MAX_MUSIC_CUE_SHARE = 0.30

# chunking
CHUNK_MAX_S = 28.0
CHUNK_HARD_MAX_S = 30.0
CHUNK_MIN_S = 2.0
MAX_INTRA_CHUNK_GAP_S = 1.5   # a longer gap between cues ends the chunk
PAD_S = 0.15                  # padding around cue span (never into neighbours)

# politeness
SLEEP_BETWEEN_VIDEOS = (6.0, 14.0)
SLEEP_BETWEEN_SEARCHES = (3.0, 7.0)
YTDLP_SLEEP_REQUESTS = 1.5

LICENSE_STR = "CC-BY 3.0 (YouTube), private split"

# YouTube search "sp" filters (protobuf, base64). 0x30=CC, 0x28=subtitles,
# 0x10=type video, 0x08 0x05 = uploaded this year; top-level 0x08 0x02 = sort by date.
SEARCH_FILTERS = {
    "cc_subs": "EgQwASgB",
    "cc_subs_year": "EggIBRABKAEwAQ%3D%3D",
    "cc_subs_newest": "CAISBhABKAEwAQ%3D%3D",
}

# Unicode script ranges
DEVANAGARI = [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]
BENGALI = [(0x0980, 0x09FF)]
GURMUKHI = [(0x0A00, 0x0A7F)]
GUJARATI = [(0x0A80, 0x0AFF)]
ORIYA = [(0x0B00, 0x0B7F)]
TAMIL = [(0x0B80, 0x0BFF)]
TELUGU = [(0x0C00, 0x0C7F)]
KANNADA = [(0x0C80, 0x0CFF)]
MALAYALAM = [(0x0D00, 0x0D7F)]
ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
LATIN = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]


@dataclass(frozen=True)
class Lang:
    code: str                 # manifest language code
    name: str
    scripts: tuple            # accepted script ranges (union)
    sub_langs: tuple          # yt-dlp subtitle language prefixes accepted
    spoken: tuple             # acceptable spoken-language codes (metadata/orig ASR)
    code_mixed: bool = False
    queries: tuple = field(default_factory=tuple)


_GENERIC_EN = ("podcast", "interview", "news discussion", "lecture", "vlog", "talk")


def _q(native: tuple, english_name: str) -> tuple:
    return native + tuple(f"{english_name} {g}" for g in _GENERIC_EN)


LANGS: dict[str, Lang] = {
    "hi": Lang("hi", "Hindi", (DEVANAGARI,), ("hi",), ("hi",), False, _q(
        ("पॉडकास्ट", "इंटरव्यू", "बातचीत", "चर्चा", "व्याख्यान", "समाचार विश्लेषण",
         "कहानी", "किसान", "ग्रामीण जीवन"), "hindi")),
    "bn": Lang("bn", "Bengali", (BENGALI,), ("bn",), ("bn",), False, _q(
        ("পডকাস্ট", "সাক্ষাৎকার", "আলোচনা", "বক্তৃতা", "খবর বিশ্লেষণ", "গল্প"), "bengali")),
    "ta": Lang("ta", "Tamil", (TAMIL,), ("ta",), ("ta",), False, _q(
        ("பாட்காஸ்ட்", "நேர்காணல்", "கலந்துரையாடல்", "சொற்பொழிவு", "விவாதம்", "செய்தி"), "tamil")),
    "te": Lang("te", "Telugu", (TELUGU,), ("te",), ("te",), False, _q(
        ("పాడ్‌కాస్ట్", "ఇంటర్వ్యూ", "చర్చ", "ఉపన్యాసం", "వార్తలు విశ్లేషణ"), "telugu")),
    "mr": Lang("mr", "Marathi", (DEVANAGARI,), ("mr",), ("mr",), False, _q(
        ("पॉडकास्ट मराठी", "मुलाखत", "चर्चा मराठी", "व्याख्यान", "बातम्या विश्लेषण"), "marathi")),
    "gu": Lang("gu", "Gujarati", (GUJARATI,), ("gu",), ("gu",), False, _q(
        ("પોડકાસ્ટ", "ઇન્ટરવ્યુ", "ચર્ચા", "વ્યાખ્યાન", "સમાચાર"), "gujarati")),
    "kn": Lang("kn", "Kannada", (KANNADA,), ("kn",), ("kn",), False, _q(
        ("ಪಾಡ್‌ಕಾಸ್ಟ್", "ಸಂದರ್ಶನ", "ಚರ್ಚೆ", "ಉಪನ್ಯಾಸ", "ಸುದ್ದಿ ವಿಶ್ಲೇಷಣೆ"), "kannada")),
    "ml": Lang("ml", "Malayalam", (MALAYALAM,), ("ml",), ("ml",), False, _q(
        ("പോഡ്കാസ്റ്റ്", "അഭിമുഖം", "ചർച്ച", "പ്രഭാഷണം", "വാർത്ത വിശകലനം"), "malayalam")),
    "pa": Lang("pa", "Punjabi", (GURMUKHI,), ("pa",), ("pa",), False, _q(
        ("ਪੋਡਕਾਸਟ", "ਇੰਟਰਵਿਊ", "ਗੱਲਬਾਤ", "ਚਰਚਾ", "ਖ਼ਬਰਾਂ"), "punjabi")),
    "or": Lang("or", "Odia", (ORIYA,), ("or",), ("or",), False, _q(
        ("ପଡକାଷ୍ଟ", "ସାକ୍ଷାତକାର", "ଆଲୋଚନା", "ଭାଷଣ", "ଖବର"), "odia")),
    "as": Lang("as", "Assamese", (BENGALI,), ("as",), ("as",), False, _q(
        ("পডকাষ্ট", "সাক্ষাৎকাৰ", "আলোচনা", "বক্তৃতা", "বাতৰি"), "assamese")),
    "ur": Lang("ur", "Urdu", (ARABIC,), ("ur",), ("ur",), False, _q(
        ("پوڈکاسٹ", "انٹرویو", "گفتگو", "مباحثہ", "لیکچر", "خبریں تجزیہ"), "urdu")),
    "hi-en": Lang("hi-en", "Hinglish", (DEVANAGARI, LATIN), ("hi", "en"), ("hi", "en"), True, (
        "hinglish podcast", "hindi podcast startup", "hindi podcast finance",
        "hindi interview tech", "career podcast hindi", "business podcast hindi",
        "पॉडकास्ट हिंदी इंग्लिश", "upsc podcast hindi")),
    "en-IN": Lang("en-IN", "Indian English", (LATIN,), ("en",), ("en",), False, (
        "indian podcast english", "india interview english", "iit lecture",
        "nptel lecture", "indian startup podcast", "india policy discussion",
        "indian english news discussion", "india tech talk")),
}

# Title/category markers for content we do not want (songs, films, label uploads).
REJECT_TITLE_WORDS = (
    "song", "lyrical", "lyrics", "full movie", "trailer", "teaser", "video song",
    "audio jukebox", "jukebox", "official video", "music video", "bhajan", "aarti",
    "गाना", "गीत", "पाटा", "பாடல்", "పాట", "ಹಾಡು", "ഗാനം", "ਗੀਤ", "ગીત", "গান",
    "#shorts", "shorts",
)
REJECT_CATEGORIES = ("Music",)
