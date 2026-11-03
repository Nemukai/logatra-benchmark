"""Shared helpers for the AIR (All India Radio / newsonair.gov.in) collector.

Stdlib only, so every sub-command runs without touching pyproject. Text
extraction (airtext.py) additionally needs pymupdf, supplied at run time with
``uv run --with pymupdf``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "private" / "air"
CATALOG = DATA / "catalog.jsonl"          # every listing card seen (discover)
ITEMS = DATA / "items"                      # one JSON sidecar per fetched bulletin
INDEX = DATA / "index.jsonl"
SQLITE = DATA / "air.sqlite"
MANIFEST = REPO / "data" / "private" / "manifests" / "air.jsonl"

BASE = "https://newsonair.gov.in"
USER_AGENT = (
    "LogatraBenchmarkResearchBot/0.1 (+non-commercial speech-recognition "
    "evaluation; polite, low-rate crawler)"
)
DELAY_S = 1.5          # min seconds between requests (single connection)
_last_request = 0.0


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _throttle() -> None:
    global _last_request
    wait = _last_request + DELAY_S + random.uniform(0, 0.5) - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def get_text(url: str, retries: int = 4) -> str:
    """GET a page politely (throttled, retried with backoff)."""
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404, 407):
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}")


def download(url: str, dest: Path, retries: int = 8) -> int:
    """Download url to dest, resuming with HTTP Range after cut transfers.

    Idempotent: if dest exists it is kept. Writes to dest.part first and
    renames only once the size matches Content-Length. Returns size.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    total = None
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            break
        _throttle()
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 200 and have:      # server ignored Range
                    have = 0
                    part.unlink(missing_ok=True)
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    total = int(cr.rsplit("/", 1)[1])
                elif r.headers.get("Content-Length"):
                    total = have + int(r.headers["Content-Length"])
                with open(part, "ab") as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
        except urllib.error.HTTPError as e:
            if e.code == 416 and total is None:   # already complete
                break
            if e.code in (403, 404, 407, 410):
                raise
            time.sleep(4 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(4 * (attempt + 1))
        else:
            if total is None or part.stat().st_size >= total:
                break
    size = part.stat().st_size if part.exists() else 0
    if size == 0 or (total is not None and size < total):
        raise RuntimeError(f"incomplete download {url}: {size}/{total}")
    os.replace(part, dest)
    return size


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# --------------------------------------------------------------------------
# Language table. Keys are the site's `bulletins_lang` slugs.
# iso: BCP-47-ish code for the manifest. scripts: Unicode scripts a clean
# script may be written in (first = expected). Some site "languages" are
# mis-entered city names; they are resolved per card from the lang label.
# --------------------------------------------------------------------------
LANGS: dict[str, dict] = {
    "assamese":   {"iso": "as",  "scripts": ["Beng"]},
    "balti":      {"iso": "bft", "scripts": ["Arab", "Tibt"]},
    "bengali":    {"iso": "bn",  "scripts": ["Beng"]},
    "bhojpuri":   {"iso": "bho", "scripts": ["Deva"]},
    "bhutia":     {"iso": "sip", "scripts": ["Tibt", "Deva"]},
    "bodo":       {"iso": "brx", "scripts": ["Deva"]},
    "chhattisgarhi": {"iso": "hne", "scripts": ["Deva"]},
    "dogri":      {"iso": "doi", "scripts": ["Deva"]},
    "english":    {"iso": "en-IN", "scripts": ["Latn"]},
    "garhwali":   {"iso": "gbm", "scripts": ["Deva"]},
    "garo":       {"iso": "grt", "scripts": ["Latn"]},
    "gojri":      {"iso": "gju", "scripts": ["Arab", "Deva"]},
    "gujarati":   {"iso": "gu",  "scripts": ["Gujr"]},
    "hindi":      {"iso": "hi",  "scripts": ["Deva"]},
    "jaintia":    {"iso": "pbv", "scripts": ["Latn"]},
    "kannada":    {"iso": "kn",  "scripts": ["Knda"]},
    "karbi":      {"iso": "mjw", "scripts": ["Latn"]},
    "kashmiri":   {"iso": "ks",  "scripts": ["Arab"]},
    "khasi":      {"iso": "kha", "scripts": ["Latn"]},
    "kokborok":   {"iso": "trp", "scripts": ["Latn", "Beng"]},
    "konkani":    {"iso": "kok", "scripts": ["Deva"]},
    "kumauni":    {"iso": "kfy", "scripts": ["Deva"]},
    "ladakhi":    {"iso": "lbj", "scripts": ["Tibt"]},
    "lepcha":     {"iso": "lep", "scripts": ["Lepc", "Deva"]},
    "maithili":   {"iso": "mai", "scripts": ["Deva"]},
    "malayalam":  {"iso": "ml",  "scripts": ["Mlym"]},
    "manipuri":   {"iso": "mni", "scripts": ["Beng", "Mtei"]},
    "marathi":    {"iso": "mr",  "scripts": ["Deva"]},
    "mizo":       {"iso": "lus", "scripts": ["Latn"]},
    "nagamese":   {"iso": "nag", "scripts": ["Latn"]},
    "nepali":     {"iso": "ne",  "scripts": ["Deva"]},
    "odia":       {"iso": "or",  "scripts": ["Orya"]},
    "pahari":     {"iso": "him", "scripts": ["Deva"]},
    "punjabi":    {"iso": "pa",  "scripts": ["Guru"]},
    "purgi":      {"iso": "prx", "scripts": ["Arab", "Tibt"]},
    "rajasthani": {"iso": "raj", "scripts": ["Deva"]},
    "sambalpuri": {"iso": "spv", "scripts": ["Orya"]},
    "sanskrit":   {"iso": "sa",  "scripts": ["Deva"]},
    "santali":    {"iso": "sat", "scripts": ["Olck", "Deva"]},
    "sindhi":     {"iso": "sd",  "scripts": ["Arab", "Deva"]},
    "tamil":      {"iso": "ta",  "scripts": ["Taml"]},
    "telugu":     {"iso": "te",  "scripts": ["Telu"]},
    "urdu":       {"iso": "ur",  "scripts": ["Arab"]},
}
# Site language-filter slugs that are not languages (data-entry errors).
NON_LANG_SLUGS = {"bhuj", "chennai", "dibrugarh", "jammu", "chhattisgarh"}
LANG_ALIASES = {"oriya": "odia", "chhattisgarh": "chhattisgarhi", "kumaoni": "kumauni",
                "meitei": "manipuri", "bangla": "bengali"}


def norm_lang(label: str) -> str | None:
    """Map a card's language label to a LANGS key (None if unknown)."""
    k = slug(label).replace("-", "")
    k = LANG_ALIASES.get(k, k)
    return k if k in LANGS else None
