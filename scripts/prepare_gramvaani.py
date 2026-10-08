"""Prepare Gramvaani Hindi Dev (5h) + Eval (3h) labeled sets.

Source: OpenSLR 118 (https://www.openslr.org/118/), GV_Dev_5h.tar.gz and
GV_Eval_3h.tar.gz only (NOT the 100h/1000h sets). Telephone-quality
community radio speech; per-utterance MP3s are resampled to 16 kHz mono
WAV with ffmpeg. License: free for academic use, commercial use requires
permission from contact@gramvaani.org.

Outputs:
    data/raw/gramvaani/             downloaded + extracted archives
    data/audio/gramvaani/<set>/     16 kHz mono WAVs
    data/manifests/gramvaani_hi.jsonl

Idempotent: exits early if the manifest exists; downloads and per-file
conversions are skipped when outputs already exist.

Run from the repo root:

    uv run python scripts/prepare_gramvaani.py
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from logatra_benchmark.manifest import Sample, write_manifest  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"
BASE_URL = "https://openslr.elda.org/resources/118"
SETS = ("GV_Dev_5h", "GV_Eval_3h")
RAW_DIR = REPO_ROOT / "data" / "raw" / "gramvaani"
AUDIO_DIR = REPO_ROOT / "data" / "audio" / "gramvaani"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "gramvaani_hi.jsonl"


def download_and_extract(set_name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    set_dir = RAW_DIR / set_name
    if not set_dir.is_dir():
        tarball = RAW_DIR / f"{set_name}.tar.gz"
        if not tarball.exists():
            url = f"{BASE_URL}/{set_name}.tar.gz"
            print(f"[gramvaani] downloading {url} ...")
            subprocess.run(
                ["curl", "-sL", "--retry", "3", "-o", str(tarball), url], check=True
            )
        print(f"[gramvaani] extracting {set_name} ...")
        with tarfile.open(tarball) as tf:
            tf.extractall(RAW_DIR)
    return set_dir


def read_kaldi_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, rest = line.partition("\t") if "\t" in line else line.partition(" ")
        out[key] = rest.strip()
    return out


def convert(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"[warn] ffmpeg failed for {src.name}: {proc.stderr.decode()[:200]}")
        return False
    return True


def prepare_set(set_name: str) -> list[Sample]:
    set_dir = download_and_extract(set_name)
    text = read_kaldi_map(set_dir / "text")
    mp3_scp = read_kaldi_map(set_dir / "mp3.scp")

    out_dir = AUDIO_DIR / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, Path, Path]] = []
    for utt_id in sorted(text):
        transcript = text[utt_id].strip()
        rel_mp3 = mp3_scp.get(utt_id)
        if not transcript or rel_mp3 is None:
            continue
        src = set_dir / rel_mp3
        if not src.exists():
            print(f"[warn] missing audio {src}")
            continue
        jobs.append((utt_id, transcript, src, out_dir / f"{utt_id}.wav"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = list(pool.map(lambda j: convert(j[2], j[3]), jobs))

    samples: list[Sample] = []
    for (utt_id, transcript, _src, dst), converted in zip(jobs, ok):
        if not converted:
            continue
        samples.append(
            Sample(
                id=f"gramvaani_{set_name}_{utt_id}",
                audio_path=str(dst.relative_to(MANIFEST_DIR, walk_up=True)),
                reference_text=transcript,
                language="hi",
                code_mixed=False,
                num_speakers=1,
                source="gramvaani_telephony",
                license="academic-use-only (commercial use needs contact@gramvaani.org)",
                split="public",
            )
        )
    print(f"[gramvaani] {set_name}: {len(samples)} samples")
    return samples


def main() -> None:
    if MANIFEST_PATH.exists():
        print(f"[skip] {MANIFEST_PATH} already exists")
        return
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for set_name in SETS:
        samples.extend(prepare_set(set_name))
    write_manifest(samples, MANIFEST_PATH)
    print(f"[done] {len(samples)} samples -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
