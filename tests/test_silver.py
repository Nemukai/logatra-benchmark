"""Tests for the silver-reference consensus logic (pure python, no network)."""

from koe_benchmark.adapters import PROVIDERS, get_adapter
from koe_benchmark.silver import _symmetric_wer


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
    monkeypatch.setenv("KOE_GEMINI_MODEL", "google/gemini-9-test")
    assert adapter.model == "google/gemini-9-test"
