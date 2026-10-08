"""Prepare MUCS 2021 subtask 2 Hindi-English (Hinglish) test set.

Source: OpenSLR 104 (https://www.openslr.org/104/), Hindi-English_test.tar.gz
(~5.2 hours, 3136 utterances, CC BY-SA 4.0). The archive ships Kaldi-style
metadata (text / segments / wav.scp); long source wavs are already 16 kHz
mono PCM, so segments are cut directly with the stdlib wave module.

Outputs:
    data/raw/mucs2021/            downloaded + extracted archive
    data/audio/mucs2021/          one 16 kHz mono WAV per utterance
    data/manifests/mucs2021_hi_en.jsonl

Idempotent: exits early if the manifest exists; re-download and per-file
cutting are skipped when outputs already exist.

Run from the repo root:

    uv run python scripts/prepare_mucs.py
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from logatra_benchmark.manifest import Sample, write_manifest  # noqa: E402

URL = "https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz"
RAW_DIR = REPO_ROOT / "data" / "raw" / "mucs2021"
AUDIO_DIR = REPO_ROOT / "data" / "audio" / "mucs2021"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "mucs2021_hi_en.jsonl"
TARGET_SR = 16_000


def download_and_extract() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tarball = RAW_DIR / "Hindi-English_test.tar.gz"
    test_dir = RAW_DIR / "test"
    if not test_dir.is_dir():
        if not tarball.exists():
            print(f"[mucs] downloading {URL} ...")
            subprocess.run(
                ["curl", "-sL", "--retry", "3", "-o", str(tarball), URL], check=True
            )
        print("[mucs] extracting ...")
        with tarfile.open(tarball) as tf:
            tf.extractall(RAW_DIR)
    return test_dir


def read_kaldi_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, rest = line.partition(" ")
        out[key] = rest.strip()
    return out


def cut_segment(src: wave.Wave_read, start: float, end: float, out_path: Path) -> None:
    sr = src.getframerate()
    start_frame = int(start * sr)
    n_frames = max(0, int(end * sr) - start_frame)
    src.setpos(start_frame)
    frames = src.readframes(n_frames)
    with wave.open(str(out_path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(sr)
        dst.writeframes(frames)


def main() -> None:
    if MANIFEST_PATH.exists():
        print(f"[skip] {MANIFEST_PATH} already exists")
        return

    test_dir = download_and_extract()
    tr_dir = test_dir / "transcripts"
    text = read_kaldi_map(tr_dir / "text")
    segments = read_kaldi_map(tr_dir / "segments")
    wav_scp = read_kaldi_map(tr_dir / "wav.scp")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    open_wavs: dict[str, wave.Wave_read] = {}
    skipped = 0
    try:
        for utt_id in sorted(text):
            transcript = text[utt_id].strip()
            seg = segments.get(utt_id)
            if not transcript or seg is None:
                skipped += 1
                continue
            wav_id, start_s, end_s = seg.split()
            start, end = float(start_s), float(end_s)

            out_path = AUDIO_DIR / f"{utt_id}.wav"
            if not out_path.exists():
                if wav_id not in open_wavs:
                    src_path = test_dir / wav_scp[wav_id]
                    open_wavs[wav_id] = wave.open(str(src_path), "rb")
                    wf = open_wavs[wav_id]
                    if wf.getframerate() != TARGET_SR or wf.getnchannels() != 1:
                        raise RuntimeError(f"{wav_id}: not 16 kHz mono")
                cut_segment(open_wavs[wav_id], start, end, out_path)

            samples.append(
                Sample(
                    id=f"mucs2021_{utt_id}",
                    audio_path=str(out_path.relative_to(MANIFEST_DIR, walk_up=True)),
                    reference_text=transcript,
                    language="hi-en",
                    code_mixed=True,
                    num_speakers=1,
                    source="mucs2021",
                    license="CC-BY-SA-4.0",
                    split="public",
                )
            )
    finally:
        for wf in open_wavs.values():
            wf.close()

    write_manifest(samples, MANIFEST_PATH)
    print(f"[done] {len(samples)} samples -> {MANIFEST_PATH} (skipped {skipped})")


if __name__ == "__main__":
    main()
