"""Segment downloaded held-out YouTube audio and build its manifest.

Expects the layout produced by _download_heldout.sh:
    data/audio/heldout_yt/"<video_id> <duration>"/full.wav        (short videos)
    data/audio/heldout_yt/"<video_id> <duration>"/sec*.wav        (long videos)
    data/raw/heldout_yt_meta/<video_id>.json                      (yt-dlp metadata)

Runs scripts/segment_audio.py per video (max 12 chunks, 10-25s, silence-cut),
then writes:
    data/manifests/heldout_yt.jsonl          manifest (reference_text empty)
    data/manifests/heldout_yt.sources.json   provenance per video

Idempotent: segmentation skips existing chunks; the manifest is rewritten.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koe_benchmark.manifest import Sample, write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "audio" / "heldout_yt"
META_DIR = ROOT / "data" / "raw" / "heldout_yt_meta"
MANIFEST_DIR = ROOT / "data" / "manifests"
SEGMENT = ROOT / "scripts" / "segment_audio.py"

# Per-video language call. Everything collected is Hindi-dominant with
# English mixing (Hinglish podcasts/interviews); mark code_mixed accordingly.
DEFAULT_LANGUAGE = "hi"
DEFAULT_CODE_MIXED = True


def main() -> int:
    samples: list[Sample] = []
    sources: dict[str, dict] = {}

    video_dirs = sorted(d for d in AUDIO_DIR.iterdir() if d.is_dir())
    if not video_dirs:
        print("no video directories found", file=sys.stderr)
        return 1

    for vdir in video_dirs:
        video_id = vdir.name.split(" ")[0]
        meta_path = META_DIR / f"{video_id}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        license_str = meta.get("license", "")
        if "Creative Commons" not in license_str:
            print(f"SKIP {video_id}: license is {license_str!r}", file=sys.stderr)
            continue

        inputs = sorted(vdir.glob("sec*.wav")) or sorted(vdir.glob("full.wav"))
        if not inputs:
            print(f"SKIP {video_id}: no wav inputs", file=sys.stderr)
            continue

        out = subprocess.run(
            [sys.executable, str(SEGMENT), "--out-dir", str(vdir),
             "--prefix", "chunk_", "--max-chunks", "12",
             *[str(p) for p in inputs]],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print(f"SKIP {video_id}: segmentation failed: {out.stderr[:300]}",
                  file=sys.stderr)
            continue
        chunks = json.loads(out.stdout)

        upload_date = meta.get("upload_date", "unknown")
        for i, ch in enumerate(chunks):
            chunk_path = Path(ch["path"]).resolve()
            samples.append(
                Sample(
                    id=f"heldout_yt_{video_id}_{i:03d}",
                    audio_path=os.path.relpath(chunk_path, MANIFEST_DIR),
                    reference_text="",
                    language=DEFAULT_LANGUAGE,
                    code_mixed=DEFAULT_CODE_MIXED,
                    num_speakers=1,
                    source=f"youtube:{video_id}:{upload_date}",
                    license="CC-BY-3.0 (YouTube, uploader-set)",
                    split="heldout",
                )
            )
        sources[video_id] = {
            "title": meta.get("title"),
            "channel": meta.get("channel"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "upload_date": upload_date,
            "license": license_str,
            "chunks": len(chunks),
            "seconds_used": round(sum(c["dur"] for c in chunks), 1),
        }
        print(f"OK {video_id}: {len(chunks)} chunks "
              f"({sources[video_id]['seconds_used']}s)", file=sys.stderr)

    write_manifest(samples, MANIFEST_DIR / "heldout_yt.jsonl")
    (MANIFEST_DIR / "heldout_yt.sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2)
    )
    total_s = sum(v["seconds_used"] for v in sources.values())
    print(f"manifest: {len(samples)} chunks, {total_s/60:.1f} min, "
          f"{len(sources)} videos", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
