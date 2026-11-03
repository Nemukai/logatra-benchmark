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


# ---------------------------------------------------------------------------
# Script-agnostic projection (v2: all benchmark scripts)
# ---------------------------------------------------------------------------

from logatra_benchmark.metrics import (  # noqa: E402
    consonant_skeleton,
    romanize,
    script_agnostic_pair,
    script_agnostic_wer,
)

# fleurs_ur_pk_test_0000: Urdu reference vs sarvam-saaras (saaras v3) output,
# which is a near-verbatim rendering of the same Urdu speech in Devanagari.
UR_REF = (
    "ان کے دور میں خفیف تلوُّث ایزا مسئلہ نہیں تھا جیسا کہ وہ آج ہے ان کی سکونت "
    "ایسے شہروں یا حوزات میں ہے جہا ں تک بآسانی پہنچا جا سکتا ہے بہ نسبت ان "
    "مقامات کے جو ماڈرن زمانے میں تعمیر کئے گئے"
)
UR_HYP_DEVANAGARI = (
    "इनके दौर में खफीब तलौवस ऐसा मसला नहीं था जैसा कि वह आज है। इनकी सुकूनत "
    "ऐसे शहरों या हौजात में है जहाँ तक बा आसानी पहुंचा जा सकता है बानिस्पति इन "
    "मकामात के जो मॉडर्न ज़माने में तामीर किए गए।"
)


class TestScriptAgnostic:
    def test_devanagari_projection_unchanged(self):
        # v1 behaviour, byte for byte: Hindi/Hinglish numbers must not move.
        assert romanize("आई वांट टू स्टार्ट") == "ai vant tu start"
        # v1 quirk kept for comparability: final-"a" deletion runs after
        # diacritic stripping, so it also drops a long final ā ("जा" -> "j").
        assert romanize("नमस्ते, दुनिया!") == "namaste duniy"
        assert romanize("मैं office जा रहा हूँ") == "main office j rah hu"

    def test_hinglish_cross_script_is_free(self):
        assert script_agnostic_wer("स्टार्ट", "start") == 0.0
        assert script_agnostic_wer("नमस्ते स्टार्ट", "namaste start") == 0.0

    def test_urdu_real_case_scores_far_above_zero(self):
        assert score_asr(UR_REF, UR_HYP_DEVANAGARI).wer == 1.0  # strict unchanged
        sa = script_agnostic_wer(UR_REF, UR_HYP_DEVANAGARI)
        assert sa < 0.4, sa  # 1 - sa = accuracy; was 0% before v2

    def test_urdu_vs_devanagari_same_words(self):
        assert script_agnostic_wer("وہ آج نہیں تھا", "वह आज नहीं था") == 0.0
        assert script_agnostic_wer("مسئلہ", "मसला") == 0.0
        assert script_agnostic_wer("زمانے", "ज़माने") == 0.0
        assert script_agnostic_wer("زمانے", "जमाने") == 0.0  # nukta dropped

    def test_urdu_content_errors_still_count(self):
        assert script_agnostic_wer("وہ آج نہیں تھا", "वह कल नहीं था") == 0.25

    def test_urdu_vs_urdu_identical_is_zero(self):
        assert script_agnostic_wer(UR_REF, UR_REF) == 0.0

    def test_skeleton_keeps_word_count(self):
        sk = consonant_skeleton(romanize("है या ہے"))
        assert sk == "_ _ _"

    def test_skeleton_only_when_perso_arabic_present(self):
        a, b = script_agnostic_pair("नमस्ते", "namaste")
        assert (a, b) == ("namaste", "namaste")
        a, b = script_agnostic_pair("نمستے", "namaste")
        assert a == b == "nmst"

    def test_odia_vs_devanagari(self):
        # Whisper writes Odia speech in Devanagari.
        # (Odia, like Bengali, does not distinguish b/v in writing, so words
        # with व are a known residual mismatch; this sentence avoids them.)
        odia = "ମାଓ ଦ୍ୱାରା ଆୟୋଜିତ ସହରରେ"
        deva = "माओ द्वारा आयोजित सहररे"
        assert score_asr(odia, deva).wer == 1.0
        assert script_agnostic_wer(odia, deva) == 0.0

    def test_bengali_assamese_vs_devanagari(self):
        assert script_agnostic_wer("আমাৰ নাম", "आमार नाम") == 0.0  # Assamese ra
        assert script_agnostic_wer("আমার নাম", "आमार नाम") == 0.0  # Bengali

    def test_other_brahmic_scripts_vs_devanagari(self):
        pairs = [
            ("ਪੰਜਾਬੀ ਵਿੱਚ", "पंजाबी विच्च"),       # Gurmukhi
            ("ગુજરાતી ભાષા", "गुजराती भाषा"),     # Gujarati
            ("తెలుగు భాష", "तेलुगु भाष"),          # Telugu
            ("ಕನ್ನಡ ಭಾಷೆ", "कन्नड भाषे"),          # Kannada
            ("മലയാളം", "मलयालं"),                 # Malayalam
        ]
        for native, deva in pairs:
            assert script_agnostic_wer(native, deva) == 0.0, (native, romanize(native), romanize(deva))

    def test_tamil_romanizes_without_spurious_aspirates(self):
        assert romanize("தமிழ் நாடு") == "tamil natu"

    def test_every_script_produces_latin(self):
        samples = ["অসমীয়া", "বাংলা", "ગુજરાતી", "ਪੰਜਾਬੀ", "ଓଡ଼ିଆ", "தமிழ்",
                   "తెలుగు", "ಕನ್ನಡ", "മലയാളം", "हिन्दी", "مراٹھی"]
        for t in samples:
            r = romanize(t)
            assert r and all(c.isascii() for c in r), (t, r)
