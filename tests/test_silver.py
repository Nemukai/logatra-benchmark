"""Tests for the silver-reference consensus logic (pure python, no network)."""

from logatra_benchmark.adapters import PROVIDERS, get_adapter
from logatra_benchmark.silver import _symmetric_wer


def test_symmetric_wer_identical():
    assert _symmetric_wer("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0


def test_symmetric_wer_ignores_punctuation_and_case():
    assert _symmetric_wer("Hello, World!", "hello world") == 0.0


def test_symmetric_wer_disjoint():
    assert _symmetric_wer("एक दो तीन", "alpha beta gamma") == 1.0


def test_symmetric_wer_empty_pair():
    assert _symmetric_wer("", "") == 0.0
    assert _symmetric_wer("kuch to hai", "") == 1.0


def test_symmetric_wer_is_symmetric():
    a, b = "main office ja raha hoon", "main office ja raha"
    assert _symmetric_wer(a, b) == _symmetric_wer(b, a)


def test_all_registered_providers_instantiate():
    for name in PROVIDERS:
        adapter = get_adapter(name)
        assert adapter.name
        assert callable(adapter.transcribe)


def test_openrouter_audio_model_override(monkeypatch):
    adapter = get_adapter("gemini")
    assert adapter.model == "google/gemini-3.5-flash"
    monkeypatch.setenv("LOGATRA_GEMINI_MODEL", "google/gemini-9-test")
    assert adapter.model == "google/gemini-9-test"


def test_romanize_devanagari_english():
    from logatra_benchmark.metrics import romanize, script_agnostic_cer
    # Same English content, one written in Devanagari, one in Latin.
    a = "आई वांट टू स्टार्ट बाय आस्किंग यू"
    b = "I want to start by asking you"
    assert script_agnostic_cer(a, b) < 0.35
    assert romanize("नमस्ते 123") == romanize("नमस्ते १२३")


def test_consensus_distance_prefers_script_agnostic():
    from logatra_benchmark.silver import _consensus_distance, _symmetric_wer
    a = "आई वांट टू स्टार्ट"
    b = "I want to start"
    assert _symmetric_wer(a, b) == 1.0
    assert _consensus_distance(a, b) < 0.5
