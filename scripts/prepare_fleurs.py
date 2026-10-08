"""Prepare FLEURS Indic test splits for the Logatra STT benchmark.

Streams the test split of each Indic FLEURS config from the ungated
HuggingFace dataset google/fleurs (no login required), caps at 100
utterances per language, writes 16 kHz mono WAVs under
data/audio/fleurs/<config>/ and one manifest per language at
data/manifests/fleurs_<lang>.jsonl.

Idempotent: a language is skipped if its manifest already exists.

Run from the repo root:

    uv run --with 'datasets>=3.6,<4' --with soundfile --with librosa \
        python scripts/prepare_fleurs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from logatra_benchmark.manifest import Sample, write_manifest  # noqa: E402

MAX_PER_LANG = 100
TARGET_SR = 16_000

# FLEURS config -> BCP-47-ish language code for the manifest
CONFIGS: dict[str, str] = {
    "hi_in": "hi",
    "bn_in": "bn",
    "ta_in": "ta",
    "te_in": "te",
    "kn_in": "kn",
    "ml_in": "ml",
    "mr_in": "mr",
    "gu_in": "gu",
    "pa_in": "pa",
    "ur_pk": "ur",
    "as_in": "as",
    "or_in": "or",
}

AUDIO_DIR = REPO_ROOT / "data" / "audio" / "fleurs"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"


def prepare_language(config: str, lang: str) -> None:
    manifest_path = MANIFEST_DIR / f"fleurs_{lang}.jsonl"
    if manifest_path.exists():
        print(f"[skip] {manifest_path} already exists")
        return

    out_dir = AUDIO_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fleurs] streaming {config} test split ...")
    ds = load_dataset("google/fleurs", config, split="test", streaming=True)

    samples: list[Sample] = []
    for idx, ex in enumerate(ds):
        if idx >= MAX_PER_LANG:
            break
        audio = ex["audio"]
        array = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != TARGET_SR:
            raise RuntimeError(f"{config}: unexpected sampling rate {sr}")
        if array.ndim != 1:
            array = array.mean(axis=1)  # downmix, FLEURS is mono already

        text = (ex.get("transcription") or "").strip()
        if not text:
            continue

        # FLEURS sentence ids repeat across speakers; use stream index for uniqueness.
        sample_id = f"fleurs_{config}_test_{idx:04d}"
        wav_path = out_dir / f"{sample_id}.wav"
        sf.write(wav_path, array, TARGET_SR, subtype="PCM_16")

        samples.append(
            Sample(
                id=sample_id,
                audio_path=str(wav_path.relative_to(MANIFEST_DIR, walk_up=True)),
                reference_text=text,
                language=lang,
                code_mixed=False,
                num_speakers=1,
                source="fleurs",
                license="CC-BY-4.0",
                split="public",
            )
        )

    if not samples:
        print(f"[warn] {config}: no samples collected, manifest not written")
        return

    write_manifest(samples, manifest_path)
    print(f"[done] {config}: {len(samples)} samples -> {manifest_path}")


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for config, lang in CONFIGS.items():
        prepare_language(config, lang)


if __name__ == "__main__":
    main()
