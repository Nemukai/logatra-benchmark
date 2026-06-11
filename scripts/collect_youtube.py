"""Collect Creative Commons licensed YouTube audio for the heldout split.

Takes YouTube video/channel/playlist URLs (as args or a file via --urls-file),
keeps only videos whose YouTube metadata reports a Creative Commons license,
downloads audio as 16 kHz mono WAV, and appends manifest rows with empty
reference_text (to be filled by human transcription later).

Requires the `collect` extra:  uv sync --extra collect
(ffmpeg must be on PATH for the WAV conversion.)

Usage:
    uv run python scripts/collect_youtube.py \
        --language hi --out data/heldout/hi \
        --manifest data/manifests/heldout-hi.jsonl \
        https://www.youtube.com/watch?v=XXXX https://www.youtube.com/@somechannel

Notes:
- License filtering relies on the uploader having marked the video as
  "Creative Commons Attribution license (reuse allowed)" (CC-BY 3.0 on
  YouTube). Always spot-check: uploaders sometimes mislabel content they do
  not own. Keep source video ids in the manifest for attribution.
- For contamination resistance, prefer videos published AFTER the release
  date of the newest model under test; pass --published-after YYYYMMDD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `koe_benchmark` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koe_benchmark.manifest import Sample, write_manifest  # noqa: E402

CC_LICENSE_MARKER = "creative commons"


def require_yt_dlp():
    try:
        import yt_dlp  # noqa: F401

        return yt_dlp
    except ImportError:
        sys.exit("yt-dlp is not installed. Run: uv sync --extra collect")


def expand_entries(yt_dlp, urls: list[str], published_after: str | None) -> list[dict]:
    """Resolve channels/playlists to individual video metadata dicts."""
    opts = {
        "quiet": True,
        "extract_flat": False,  # need full metadata for the license field
        "skip_download": True,
        "ignoreerrors": True,
    }
    if published_after:
        opts["daterange"] = yt_dlp.utils.DateRange(start=published_after)

    entries: list[dict] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            info = ydl.extract_info(url, download=False)
            if info is None:
                print(f"  ! could not resolve {url}", file=sys.stderr)
                continue
            if info.get("_type") == "playlist":
                entries.extend(e for e in info.get("entries", []) if e)
            else:
                entries.append(info)
    return entries


def is_creative_commons(entry: dict) -> bool:
    license_str = (entry.get("license") or "").lower()
    return CC_LICENSE_MARKER in license_str


def download_audio(yt_dlp, video_id: str, out_dir: Path) -> Path | None:
    """Download one video's audio as 16 kHz mono WAV; return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{video_id}.wav"
    if target.exists():
        return target
    opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
        # 16 kHz mono: the common denominator across STT providers.
        "postprocessor_args": {"ffmpegextractaudio": ["-ar", "16000", "-ac", "1"]},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        code = ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    if code != 0 or not target.exists():
        print(f"  ! download failed for {video_id}", file=sys.stderr)
        return None
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("urls", nargs="*", help="YouTube video/channel/playlist URLs")
    parser.add_argument("--urls-file", help="file with one URL per line")
    parser.add_argument("--language", required=True, help='manifest language, e.g. "hi"')
    parser.add_argument("--out", required=True, help="directory for downloaded WAVs")
    parser.add_argument("--manifest", required=True, help="JSONL manifest to append to")
    parser.add_argument(
        "--published-after",
        default=None,
        metavar="YYYYMMDD",
        help="skip videos published before this date (contamination guard)",
    )
    parser.add_argument(
        "--code-mixed",
        action="store_true",
        help="mark collected rows as code-mixed (e.g. Hinglish playlists)",
    )
    args = parser.parse_args()

    urls = list(args.urls)
    if args.urls_file:
        urls += [
            u.strip()
            for u in Path(args.urls_file).read_text().splitlines()
            if u.strip() and not u.startswith("#")
        ]
    if not urls:
        parser.error("no URLs given (positional args or --urls-file)")

    yt_dlp = require_yt_dlp()
    out_dir = Path(args.out)
    manifest_path = Path(args.manifest)

    print(f"Resolving {len(urls)} URL(s)...")
    entries = expand_entries(yt_dlp, urls, args.published_after)
    cc_entries = [e for e in entries if is_creative_commons(e)]
    print(f"{len(entries)} videos found, {len(cc_entries)} with Creative Commons license")

    # Append to an existing manifest if present.
    existing: list[Sample] = []
    if manifest_path.exists():
        from koe_benchmark.manifest import load_manifest

        existing = load_manifest(manifest_path)
    existing_ids = {s.id for s in existing}

    new_samples: list[Sample] = []
    for entry in cc_entries:
        video_id = entry["id"]
        sample_id = f"yt-{args.language}-{video_id}"
        if sample_id in existing_ids:
            continue
        wav = download_audio(yt_dlp, video_id, out_dir)
        if wav is None:
            continue
        new_samples.append(
            Sample(
                id=sample_id,
                audio_path=str(wav),
                reference_text="",  # filled in by human transcription later
                language=args.language,
                code_mixed=args.code_mixed,
                num_speakers=1,  # update during transcription pass
                speaker_segments=None,
                source=f"youtube:{video_id}",
                license=entry.get("license") or "Creative Commons (verify)",
                split="heldout",
            )
        )
        print(f"  + {sample_id}: {entry.get('title', '')[:60]}")

    write_manifest(existing + new_samples, manifest_path)
    print(f"Wrote {len(new_samples)} new rows to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
