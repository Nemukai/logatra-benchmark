"""Pricing table and per-provider language mapping (pure, offline)."""

from __future__ import annotations

import pytest

from logatra_benchmark import pricing
from logatra_benchmark.adapters import PROVIDERS, get_adapter
from logatra_benchmark.adapters import languages as L


def test_every_provider_has_a_dated_sourced_price():
    for name in PROVIDERS:
        p = pricing.get_price(name)
        assert p is not None, name
        assert p.source.startswith("https://"), name
        assert p.checked == pricing.CHECKED
        assert p.rate() is not None or p.estimate_usd_per_min is not None, name


def test_sarvam_is_inr_native_and_not_the_stale_rate(monkeypatch):
    monkeypatch.delenv("LOGATRA_INR_PER_USD", raising=False)
    p = pricing.get_price("sarvam-saaras-v4")
    assert p.inr_per_min == 0.49
    assert p.rate() == pytest.approx(0.49 / 83.5)
    monkeypatch.setenv("LOGATRA_INR_PER_USD", "88")
    assert p.rate() == pytest.approx(0.49 / 88)


def test_variants_deepgram_and_assemblyai():
    assert pricing.usd_per_min("deepgram", "ta") == 0.0043
    assert pricing.usd_per_min("deepgram", "hi", code_mixed=True) == 0.0052
    assert pricing.usd_per_min("assemblyai", "hi") == pytest.approx(0.21 / 60)
    assert pricing.usd_per_min("assemblyai", "bn") == pytest.approx(0.15 / 60)
    raw = {"speech_model_used": "universal-2"}
    assert pricing.usd_per_min("assemblyai", "hi", raw=raw) == pytest.approx(0.15 / 60)


def test_estimate_cost_and_budget_fallback():
    assert pricing.estimate_cost_usd("whisper", 120) == pytest.approx(0.012)
    # token-billed: no list price, rough figure only for budgeting
    assert pricing.estimate_cost_usd("gemini", 60, for_budget=False) is None
    assert pricing.estimate_cost_usd("gemini", 60, for_budget=True) > 0


def test_record_cost_prefers_reported_usage():
    rec = {"raw": {"usage": {"cost": 0.0123}}, "audio_s": 60}
    assert pricing.record_cost_usd("gemini", rec) == 0.0123
    rec = {"raw": {}, "audio_s": 60, "language": "ta"}
    assert pricing.record_cost_usd("deepgram", rec) == pytest.approx(0.0043)
    rec = {"raw": {"_logatra_params": [["language", "multi"]]}, "audio_s": 60}
    assert pricing.record_cost_usd("deepgram", rec) == pytest.approx(0.0052)
    assert pricing.record_cost_usd("gemini", {"raw": {}}, 60) is None


def test_price_table_rows():
    rows = {r["provider"]: r for r in pricing.price_table()}
    assert rows["gpt-transcribe"]["usd_per_min"] == 0.0045
    assert rows["google"]["usd_per_hour"] == pytest.approx(0.96)


# --- language mapping ---------------------------------------------------------

def test_canonical_normalization():
    assert L.canonical("en") == "en-IN"
    assert L.canonical("od") == "or"
    assert L.canonical("hi", code_mixed=True) == "hi-en"
    assert L.canonical("ta-IN") == "ta"
    assert L.canonical(None) is None


@pytest.mark.parametrize("provider,lang,expected", [
    ("deepgram", "hi-en", "multi"), ("deepgram", "ne", "ne"),
    ("google", "pa", "pa-Guru-IN"), ("google", "ne", "ne-NP"),
    ("elevenlabs", "or", "ori"), ("elevenlabs", "hi-en", L.AUTO),
    ("assemblyai", "en-IN", "en"), ("openai", "or", L.AUTO),
    ("sarvam-saaras", "or", "od-IN"), ("sarvam-saaras", "kok", "kok-IN"),
    ("azure", "ur", "ur-IN"), ("indicconformer", "sat", "sat"),
])
def test_lookup(provider, lang, expected):
    assert L.lookup(provider, lang) == expected


@pytest.mark.parametrize("provider,lang", [
    ("deepgram", "ml"), ("deepgram", "or"), ("google", "ur"), ("assemblyai", "or"),
    ("elevenlabs", "mni"), ("sarvam-saarika", "kok"), ("azure", "brx"),
    ("indicconformer", "en-IN"),
])
def test_unsupported(provider, lang):
    assert L.lookup(provider, lang) is None
    assert "not supported" in L.unsupported_reason(provider, lang)


def test_every_core_language_has_some_provider():
    m = L.coverage_matrix()
    for lang in L.ALL_LANGUAGES:
        assert any(row[lang] != "-" for row in m.values()), lang


def test_adapter_supports_uses_matrix():
    assert get_adapter("deepgram").supports("ml") is not None
    assert get_adapter("deepgram").supports("hi", True) is None
    assert get_adapter("google").supports("ur") is not None
    assert get_adapter("sarvam-saaras-v4").supports("mni") is None
    assert get_adapter("sarvam").supports("mni") is not None
    from logatra_benchmark.adapters.base import adapter_supports

    assert adapter_supports(get_adapter("gemini"), "mni") is None  # prompt-based: all
