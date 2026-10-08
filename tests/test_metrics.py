"""Tests for the Indic normalization pipeline and WER/CER scoring.

Pure python, no network.
"""

import unicodedata

from logatra_benchmark.metrics import normalize_text, score_asr


class TestNormalizeText:
    def test_nfc_normalization(self):
        # क + ़ (decomposed) vs क़ (composed) must normalize identically
        decomposed = "क़"
        composed = "क़"
        assert unicodedata.normalize("NFC", composed) == unicodedata.normalize(
            "NFC", decomposed
        )
        assert normalize_text(decomposed) == normalize_text(composed)

    def test_lowercase_latin(self):
        assert normalize_text("Namaste India") == "namaste india"

    def test_strips_ascii_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_strips_devanagari_danda(self):
        assert normalize_text("मैं घर जा रहा हूँ।") == "मैं घर जा रहा हूँ"
        assert normalize_text("समाप्त॥") == "समाप्त"

    def test_strips_urdu_punctuation(self):
        assert normalize_text("آپ کیسے ہیں؟") == "آپ کیسے ہیں"

    def test_devanagari_digits_normalized(self):
        assert normalize_text("१२३") == "123"

    def test_bengali_and_tamil_digits_normalized(self):
        assert normalize_text("৪৫") == "45"
        assert normalize_text("௬௭") == "67"

    def test_arabic_indic_digits_normalized(self):
        assert normalize_text("٤٢") == "42"

    def test_whitespace_collapsed(self):
        assert normalize_text("  एक   दो \n तीन ") == "एक दो तीन"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestScoreASR:
    def test_exact_match_after_normalization(self):
        scores = score_asr("मैं घर जा रहा हूँ।", "मैं घर जा रहा हूँ")
        assert scores.wer == 0.0
        assert scores.cer == 0.0
        assert scores.num_reference_words == 5

    def test_digit_variants_match(self):
        scores = score_asr("कमरा १२३", "कमरा 123")
        assert scores.wer == 0.0

    def test_one_substitution(self):
        scores = score_asr("एक दो तीन चार", "एक दो पाँच चार")
        assert scores.wer == 0.25

    def test_full_miss(self):
        scores = score_asr("नमस्ते", "")
        assert scores.wer == 1.0

    def test_empty_reference_empty_hypothesis(self):
        scores = score_asr("", "")
        assert scores.wer == 0.0
        assert scores.num_reference_words == 0

    def test_empty_reference_nonempty_hypothesis(self):
        scores = score_asr("।", "कुछ")  # reference normalizes to empty
        assert scores.wer == 1.0

    def test_case_insensitive_english(self):
        scores = score_asr("Open The Door", "open the door")
        assert scores.wer == 0.0


class TestManifestRoundTrip:
    def test_sample_roundtrip(self, tmp_path):
        from logatra_benchmark.manifest import Sample, SpeakerSegment, load_manifest, write_manifest

        sample = Sample(
            id="hi-public-0001",
            audio_path="audio/0001.wav",
            reference_text="नमस्ते दुनिया",
            language="hi",
            code_mixed=False,
            num_speakers=2,
            speaker_segments=[
                SpeakerSegment(speaker="spk0", start=0.0, end=2.5),
                SpeakerSegment(speaker="spk1", start=2.5, end=4.0),
            ],
            source="ai4bharat/IndicVoices",
            license="CC-BY-4.0",
            split="public",
        )
        path = tmp_path / "m.jsonl"
        write_manifest([sample], path)
        loaded = load_manifest(path)
        assert len(loaded) == 1
        assert loaded[0] == sample

    def test_invalid_split_rejected(self):
        import pytest

        from logatra_benchmark.manifest import Sample

        with pytest.raises(ValueError, match="split"):
            Sample(
                id="x",
                audio_path="x.wav",
                reference_text="",
                language="hi",
                split="train",
            )
