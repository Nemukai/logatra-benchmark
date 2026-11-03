"""Dated, sourced list prices for every benchmark entrant (USD per audio minute).

All prices are public pay-as-you-go list prices for pre-recorded/batch
transcription, checked on CHECKED (below) against the URL in `source`.
Credits, startup deals and volume tiers are NOT applied: these are the numbers
a reader of the leaderboard would pay.

Use from the scorer:

    from logatra_benchmark.pricing import record_cost_usd
    cost = record_cost_usd(provider, record, audio_s)

`record_cost_usd` prefers a provider-reported cost (`raw.usage.cost`, as
OpenRouter returns) and otherwise applies the list rate, choosing the right
variant (e.g. Deepgram multilingual for Hinglish, AssemblyAI Universal-2 vs
Universal-3.5 Pro) from the record's language or raw response.

INR: Sarvam bills natively in INR; USD is derived with INR_PER_USD (the
repo-wide convention, 83.5; override with LOGATRA_INR_PER_USD).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

CHECKED = "2026-11-03"


def inr_per_usd() -> float:
    try:
        return float(os.environ.get("LOGATRA_INR_PER_USD", "") or 83.5)
    except ValueError:
        return 83.5


INR_PER_USD = 83.5


@dataclass(frozen=True)
class Price:
    provider: str
    model: str
    usd_per_min: float | None          # None: billed per token, no flat rate
    source: str
    checked: str = CHECKED
    inr_per_min: float | None = None   # native INR price, if billed in INR
    variants: dict = field(default_factory=dict)  # variant -> usd_per_min
    estimate_usd_per_min: float | None = None     # rough figure for budgeting only
    notes: str = ""

    def rate(self, variant: str | None = None) -> float | None:
        if variant and variant in self.variants:
            return self.variants[variant]
        if self.inr_per_min is not None:
            return self.inr_per_min / inr_per_usd()
        return self.usd_per_min


_SARVAM_SRC = "https://docs.sarvam.ai/api/getting-started/pricing"
_SARVAM_NOTE = (
    "Rs 0.49/min pay-as-you-go (Pro Rs 0.45, Growth Rs 0.41), same rate for "
    "real-time REST, streaming and batch, billed per second, excl. GST. "
    "Diarization (batch) Rs 0.59/min. Third-party 'Rs 30/hr (Rs 45/hr "
    "diarized)' is the older per-hour sheet; current official is per-minute. "
    "The repo's previous Rs 0.75/min assumption is stale."
)
_OPENAI_SRC = "https://developers.openai.com/api/docs/pricing"
_OPENAI_DEPR = ("Deprecated by OpenAI; shuts down 2027-02-26 "
                "(https://developers.openai.com/api/docs/guides/speech-to-text).")

PRICES: dict[str, Price] = {
    "sarvam": Price("sarvam", "saarika:v2.5", None, _SARVAM_SRC, inr_per_min=0.49,
                    notes=_SARVAM_NOTE + " Saarika is deprecated by Sarvam."),
    "sarvam-saaras": Price("sarvam-saaras", "saaras:v3", None, _SARVAM_SRC,
                           inr_per_min=0.49, notes=_SARVAM_NOTE),
    "sarvam-saaras-v4": Price("sarvam-saaras-v4", "saaras:v4", None, _SARVAM_SRC,
                              inr_per_min=0.49, notes=_SARVAM_NOTE),
    "whisper": Price("whisper", "whisper-1", 0.006, _OPENAI_SRC, notes=_OPENAI_DEPR),
    "gpt-4o-transcribe": Price("gpt-4o-transcribe", "gpt-4o-transcribe", 0.006, _OPENAI_SRC,
                               notes="Token-billed ($2.50/1M audio in); OpenAI's "
                                     "estimated $0.006/min. " + _OPENAI_DEPR),
    "gpt-4o-mini-transcribe": Price("gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe", 0.003,
                                    _OPENAI_SRC,
                                    notes="Token-billed ($1.25/1M audio in); OpenAI's "
                                          "estimated $0.003/min. " + _OPENAI_DEPR),
    "gpt-transcribe": Price("gpt-transcribe", "gpt-transcribe", 0.0045,
                            "https://developers.openai.com/api/docs/models/gpt-transcribe",
                            notes="OpenAI's current recommended file transcription model."),
    "deepgram": Price("deepgram", "nova-3", 0.0043, "https://deepgram.com/pricing",
                      variants={"monolingual": 0.0043, "multilingual": 0.0052},
                      notes="Pre-recorded pay-as-you-go: Nova-3 monolingual $0.0043/min, "
                            "multilingual (language=multi, used for Hinglish) $0.0052/min; "
                            "Growth $0.0036/$0.0043. Billed per second."),
    "google": Price("google", "chirp_3", 0.016, "https://cloud.google.com/speech-to-text/pricing",
                    variants={"dynamic_batch": 0.003},
                    notes="Speech-to-Text v2 Standard recognition (chirp included), "
                          "0-500k min/month tier $0.016/min, 500k-1M $0.01, 1M-2M $0.008, "
                          "2M+ $0.004; dynamic batch $0.003/min. Per-second rounding per request."),
    "elevenlabs": Price("elevenlabs", "scribe_v2", 0.22 / 60, "https://elevenlabs.io/pricing/api",
                        notes="'Speech to Text $0.22 per hour (Scribe)' per the API pricing FAQ; "
                              "keyterm prompting +$0.05/h. Silence is billed. Plan-tier rates "
                              "on the interactive table were not machine-readable."),
    "assemblyai": Price("assemblyai", "universal-3-5-pro / universal-2", 0.21 / 60,
                        "https://www.assemblyai.com/pricing",
                        variants={"universal-3-5-pro": 0.21 / 60, "universal-2": 0.15 / 60},
                        notes="Pay-as-you-go: Universal-3.5 Pro $0.21/h (hi, en), "
                              "Universal-2 $0.15/h (other Indic). Diarization +$0.02/h."),
    "azure": Price("azure", "fast-transcription", 0.36 / 60,
                   "https://prices.azure.com/api/retail/prices "
                   "(meter 'Fast Transcription Speech To Text', centralindia)",
                   notes="$0.36/audio hour list; a $0.10/h 'Promo' meter appears in some "
                         "regions from 2026-09-01. The azure.microsoft.com page renders "
                         "prices client-side and could not be read directly."),
    "indicconformer": Price("indicconformer", "ai4bharat/indic-conformer-600m-multilingual", 0.0,
                            "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual",
                            notes="Open weights (MIT); local compute only."),
    "gemini": Price("gemini", "google/gemini-3.5-flash", None, "https://openrouter.ai/models",
                    estimate_usd_per_min=0.01,
                    notes="Token-billed via OpenRouter; actual cost from raw.usage.cost. "
                          "estimate_usd_per_min is a rough budgeting guess, not a list price."),
    "voxtral": Price("voxtral", "mistralai/voxtral-small-24b-2507", None,
                     "https://openrouter.ai/models", estimate_usd_per_min=0.005,
                     notes="Token-billed via OpenRouter; actual cost from raw.usage.cost. "
                           "Estimate is a rough guess."),
    "gpt-audio": Price("gpt-audio", "openai/gpt-audio", None, "https://openrouter.ai/models",
                       estimate_usd_per_min=0.04,
                       notes="Token-billed via OpenRouter ($32/1M audio tokens in at OpenAI); "
                             "actual cost from raw.usage.cost. Estimate is a rough guess."),
}

# Rough wall-clock model per request: (fixed overhead seconds, real-time factor).
# Used only by `estimate`; not measured for the new providers.
LATENCY: dict[str, tuple[float, float]] = {
    "sarvam": (1.0, 0.10), "sarvam-saaras": (1.0, 0.10), "sarvam-saaras-v4": (1.0, 0.10),
    "whisper": (1.5, 0.10), "gpt-4o-transcribe": (1.5, 0.10),
    "gpt-4o-mini-transcribe": (1.0, 0.08), "gpt-transcribe": (1.5, 0.10),
    "deepgram": (0.5, 0.03), "google": (1.0, 0.15), "elevenlabs": (1.0, 0.10),
    "assemblyai": (4.0, 0.10), "azure": (1.0, 0.10), "indicconformer": (0.2, 0.30),
    "gemini": (3.0, 0.20), "voxtral": (2.0, 0.15), "gpt-audio": (3.0, 0.30),
}

# Long clips go to async/batch paths with extra queueing overhead.
BATCH_OVERHEAD_S = {"sarvam-saaras": 20.0, "sarvam-saaras-v4": 20.0, "google": 30.0}


def get_price(provider: str) -> Price | None:
    return PRICES.get(provider)


def variant_for(provider: str, language: str | None, code_mixed: bool = False,
                raw: dict | None = None) -> str | None:
    """Pick the billing variant for a sample (or a cached response)."""
    from .adapters import languages as L

    canon = L.canonical(language, code_mixed)
    if provider == "deepgram":
        if raw:
            params = dict(raw.get("_logatra_params") or [])
            if params.get("language") == "multi":
                return "multilingual"
            if params:
                return "monolingual"
        return "multilingual" if L.DEEPGRAM.get(canon or "") == "multi" else "monolingual"
    if provider == "assemblyai":
        used = (raw or {}).get("speech_model_used")
        if used in ("universal-3-5-pro", "universal-2"):
            return used
        from .adapters.assemblyai import speech_models_for

        return speech_models_for(language, code_mixed)[0]
    return None


def usd_per_min(provider: str, language: str | None = None, code_mixed: bool = False,
                raw: dict | None = None, for_budget: bool = False) -> float | None:
    """List rate in USD/min; with for_budget, fall back to the rough estimate."""
    price = PRICES.get(provider)
    if price is None:
        return None
    rate = price.rate(variant_for(provider, language, code_mixed, raw))
    if rate is None and for_budget:
        rate = price.estimate_usd_per_min
    return rate


def estimate_cost_usd(provider: str, audio_s: float, language: str | None = None,
                      code_mixed: bool = False, for_budget: bool = True) -> float | None:
    rate = usd_per_min(provider, language, code_mixed, for_budget=for_budget)
    return None if rate is None else rate * audio_s / 60.0


def record_cost_usd(provider: str, record: dict, audio_s: float | None = None) -> float | None:
    """Cost of one cached result record (what the scorer should call).

    Order: raw.usage.cost (provider-reported) -> record['cost_usd'] (written by
    the runner at call time) -> list rate x audio seconds.
    """
    raw = record.get("raw") or {}
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    if isinstance(record.get("cost_usd"), (int, float)):
        return float(record["cost_usd"])
    secs = audio_s if audio_s is not None else record.get("audio_s")
    if secs is None:
        return None
    rate = usd_per_min(provider, record.get("language"), bool(record.get("code_mixed")), raw)
    return None if rate is None else rate * secs / 60.0


def estimate_wall_s(provider: str, durations: list[float], workers: int = 1,
                    sync_limit_s: float | None = None) -> float:
    overhead, rtf = LATENCY.get(provider, (2.0, 0.2))
    batch = BATCH_OVERHEAD_S.get(provider, 0.0)
    total = 0.0
    for d in durations:
        total += overhead + rtf * d
        if batch and sync_limit_s is not None and d > sync_limit_s:
            total += batch
    return total / max(workers, 1)


def price_table() -> list[dict]:
    """Rows for docs / CLI: provider, model, usd_per_min, usd_per_hour, source, checked."""
    rows = []
    for p in PRICES.values():
        rate = p.rate()
        rows.append({
            "provider": p.provider, "model": p.model,
            "usd_per_min": None if rate is None else round(rate, 6),
            "usd_per_hour": None if rate is None else round(rate * 60, 4),
            "inr_per_min": p.inr_per_min, "variants": dict(p.variants),
            "source": p.source, "checked": p.checked, "notes": p.notes,
        })
    return rows
