"""Segment WAV audio into speech chunks at silence boundaries.

Takes one or more 16 kHz mono WAV files (e.g. sections downloaded from one
YouTube video), detects silences with ffmpeg silencedetect, and cuts chunks
of roughly --min-dur..--max-dur seconds, preferring cuts at silence
midpoints. Chunks shorter than --keep-min seconds or with more than 50%
silence are discarded. If more than --max-chunks chunks are produced, they
are evenly subsampled across all inputs so no region dominates.

Idempotent: chunk boundaries are computed deterministically from the input
audio; existing chunk files with the right name are not re-cut, and stale
chunk files matching the prefix are removed.

Usage:
    python scripts/segment_audio.py \
        --out-dir data/audio/heldout_yt/VIDEOID \
        --prefix chunk_ --max-chunks 12 \
        data/audio/heldout_yt/VIDEOID/sec0.wav [sec1.wav ...]

Prints a JSON summary (one object per kept chunk) to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

SILENCE_START_RE = re.compile(r"silence_start: ([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end: ([0-9.]+)")


@dataclass
class Chunk:
    src: Path
    start: float
    end: float
    speech_frac: float

    @property
    def dur(self) -> float:
        return self.end - self.start


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def detect_silences(path: Path, noise_db: float, min_silence: float) -> list[tuple[float, float]]:
    """Return [(silence_start, silence_end), ...] for one file."""
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = out.stderr
    starts = [float(m) for m in SILENCE_START_RE.findall(log)]
    ends = [float(m) for m in SILENCE_END_RE.findall(log)]
    silences: list[tuple[float, float]] = []
    duration = probe_duration(path)
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else duration
        silences.append((s, e))
    return silences


def silence_overlap(silences: list[tuple[float, float]], start: float, end: float) -> float:
    """Seconds of silence inside [start, end]."""
    total = 0.0
    for s, e in silences:
        total += max(0.0, min(e, end) - max(s, start))
    return total


def chunk_one_file(
    path: Path,
    min_dur: float,
    max_dur: float,
    keep_min: float,
    noise_db: float,
    min_silence: float,
) -> list[Chunk]:
    duration = probe_duration(path)
    silences = detect_silences(path, noise_db, min_silence)

    # Candidate cut points: midpoints of detected silences, plus file edges.
    cuts = sorted({0.0, duration} | {(s + e) / 2 for s, e in silences})

    chunks: list[Chunk] = []
    t = 0.0
    while t < duration - keep_min:
        lo, hi = t + min_dur, t + max_dur
        in_window = [c for c in cuts if lo <= c <= hi]
        if in_window:
            end = in_window[-1]  # latest silence cut inside the window
        elif duration - t <= max_dur:
            end = duration  # tail fits in one chunk
        else:
            end = hi  # no silence nearby: hard cut at max_dur
        end = min(end, duration)
        dur = end - t
        if dur >= keep_min:
            sil = silence_overlap(silences, t, end)
            speech_frac = 1.0 - sil / dur if dur > 0 else 0.0
            if speech_frac >= 0.5:
                chunks.append(Chunk(src=path, start=t, end=end, speech_frac=speech_frac))
        # Resume after the silence we cut in (skip to next non-silent moment).
        nxt = end
        for s, e in silences:
            if s <= nxt < e:
                nxt = e
                break
        t = max(nxt, end)
    return chunks


def subsample(chunks: list[Chunk], max_chunks: int) -> list[Chunk]:
    """Evenly subsample so chunks are spread across all inputs, not the start."""
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    picked = [chunks[min(int(i * step), len(chunks) - 1)] for i in range(max_chunks)]
    # int() collisions are possible only when step < 1, which can't happen here.
    return picked


def cut_chunk(chunk: Chunk, dest: Path) -> None:
    if dest.exists():
        return
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{chunk.start:.3f}", "-t", f"{chunk.dur:.3f}",
         "-i", str(chunk.src), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("inputs", nargs="+", help="input WAV file(s), in playback order")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="chunk_")
    p.add_argument("--max-chunks", type=int, default=12)
    p.add_argument("--min-dur", type=float, default=10.0)
    p.add_argument("--max-dur", type=float, default=25.0)
    p.add_argument("--keep-min", type=float, default=3.0)
    p.add_argument("--noise-db", type=float, default=-32.0)
    p.add_argument("--min-silence", type=float, default=0.4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 1
        all_chunks.extend(
            chunk_one_file(path, args.min_dur, args.max_dur, args.keep_min,
                           args.noise_db, args.min_silence)
        )

    kept = subsample(all_chunks, args.max_chunks)

    # Remove stale chunks from previous runs with different boundaries.
    expected = {f"{args.prefix}{i:03d}.wav" for i in range(len(kept))}
    for old in out_dir.glob(f"{args.prefix}*.wav"):
        if old.name not in expected:
            old.unlink()

    summary = []
    for i, ch in enumerate(kept):
        dest = out_dir / f"{args.prefix}{i:03d}.wav"
        cut_chunk(ch, dest)
        summary.append({
            "path": str(dest),
            "src": str(ch.src),
            "start": round(ch.start, 3),
            "end": round(ch.end, 3),
            "dur": round(ch.dur, 3),
            "speech_frac": round(ch.speech_frac, 3),
        })

    json.dump(summary, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
