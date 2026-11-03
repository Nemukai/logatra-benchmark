"""Dataset manifest format and loader.

A manifest is a JSONL file: one JSON object per line describing one audio
sample. Fields:

    id              str   unique sample id (stable across runs; used as cache key)
    audio_path      str   path to audio file, absolute or relative to the manifest
    reference_text  str   human reference transcript ("" for not-yet-transcribed)
    language        str   BCP-47-ish code, e.g. "hi", "ta", "bn", "en-IN"
    code_mixed      bool  True if the utterance code-switches (e.g. Hinglish)
    num_speakers    int   number of distinct speakers in the audio
    speaker_segments list | None
                    optional RTTM-style segments for diarization scoring:
                    [{"speaker": "spk0", "start": 0.0, "end": 4.2}, ...]
    source          str   provenance, e.g. "ai4bharat/IndicVoices",
                    "youtube:VIDEO_ID", "self-recorded"
    license         str   license of the audio, e.g. "CC-BY-4.0"
    split           str   "public" (contaminated baseline), "heldout" (fresh,
                          published audio) or "private" (fresh audio that is
                          never published; see PRIVATE_REPORT_DIR)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Literal

Split = Literal["public", "heldout", "private"]

VALID_SPLITS: tuple[str, ...] = ("public", "heldout", "private")

# Splits whose texts and scores may appear in report/ and on Hugging Face.
PUBLISHABLE_SPLITS: tuple[str, ...] = ("public", "heldout")

# The only place private-split scores, rows or cases may be written. It is
# gitignored (data/private/) and never staged by scripts/publish_hf.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = REPO_ROOT / "data" / "private"
PRIVATE_REPORT_DIR = PRIVATE_DIR / "reports"


def is_private_path(path: str | Path) -> bool:
    """True if ``path`` resolves inside data/private/."""
    try:
        Path(path).resolve().relative_to(PRIVATE_DIR.resolve())
    except ValueError:
        return False
    return True


def require_private_output(path: str | Path) -> None:
    """Refuse to write private-split material anywhere but data/private/."""
    if not is_private_path(path):
        raise SystemExit(
            f"refusing to write private-split output to {path}: private rows "
            f"may only be written under {PRIVATE_REPORT_DIR} (gitignored)"
        )


@dataclass(slots=True)
class SpeakerSegment:
    """One diarization reference segment (RTTM-style)."""

    speaker: str
    start: float  # seconds
    end: float    # seconds

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"segment end {self.end} < start {self.start}")


@dataclass(slots=True)
class Sample:
    id: str
    audio_path: str
    reference_text: str
    language: str
    code_mixed: bool = False
    num_speakers: int = 1
    speaker_segments: list[SpeakerSegment] | None = None
    source: str = ""
    license: str = ""
    split: str = "public"

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"sample {self.id!r}: split must be one of {VALID_SPLITS}, got {self.split!r}"
            )

    def resolve_audio_path(self, manifest_dir: Path) -> Path:
        """Resolve audio_path relative to the manifest's directory."""
        p = Path(self.audio_path)
        return p if p.is_absolute() else manifest_dir / p

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        segments = d.get("speaker_segments")
        if segments is not None:
            segments = [SpeakerSegment(**s) for s in segments]
        return cls(
            id=d["id"],
            audio_path=d["audio_path"],
            reference_text=d.get("reference_text", ""),
            language=d["language"],
            code_mixed=bool(d.get("code_mixed", False)),
            num_speakers=int(d.get("num_speakers", 1)),
            speaker_segments=segments,
            source=d.get("source", ""),
            license=d.get("license", ""),
            split=d.get("split", "public"),
        )


def load_manifest(path: str | Path) -> list[Sample]:
    """Load a JSONL manifest, validating each row."""
    path = Path(path)
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            sample = Sample.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: invalid manifest row: {exc}") from exc
        if sample.id in seen_ids:
            raise ValueError(f"{path}:{lineno}: duplicate sample id {sample.id!r}")
        seen_ids.add(sample.id)
        samples.append(sample)
    return samples


def write_manifest(samples: Iterator[Sample] | list[Sample], path: str | Path) -> None:
    """Write samples to a JSONL manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(sample.to_json() + "\n")
