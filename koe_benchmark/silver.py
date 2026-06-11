"""Silver-reference pipeline: multi-model consensus transcription.

For audio without a human reference, transcribe with N independent providers
and use cross-model agreement as a quality gate:

1. Each configured provider transcribes the sample (cache-first, via the same
   results/ cache as the benchmark runner, so silver generation and benchmark
   runs never pay twice for the same call).
2. Pairwise distances between hypotheses = symmetric WER on normalized text
   (mean of wer(a, b) and wer(b, a), since WER is asymmetric).
3. The medoid hypothesis (lowest mean distance to all others) becomes the
   candidate reference.
4. If the medoid's mean distance <= --threshold, the row is accepted as a
   SILVER reference. Otherwise it goes to the REVIEW queue with all
   hypotheses attached, for human adjudication.

Silver references are marked in the output manifest via source suffix
":silver" so they can never be silently confused with human gold references.
Benchmark scores on silver rows measure provider *agreement*, not absolute
truth; for any provider that contributed to the consensus, treat its silver
scores as upper bounds (it cannot disagree with itself).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .adapters import NotConfigured
from .manifest import Sample, load_manifest, write_manifest
from .metrics import normalize_text, score_asr
from .runner import transcribe_sample


@dataclass(slots=True)
class SilverResult:
    sample_id: str
    status: str  # "silver" | "review" | "failed"
    chosen_provider: str | None = None
    reference_text: str | None = None
    mean_distance: float | None = None
    hypotheses: dict | None = None  # provider -> text


def _symmetric_wer(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 0.0
    if not na or not nb:
        return 1.0
    return (score_asr(na, nb).wer + score_asr(nb, na).wer) / 2


def generate_silver(
    manifest_path: str | Path,
    providers: list[str],
    results_dir: str | Path = "results",
    threshold: float = 0.20,
    min_hypotheses: int = 2,
    only_empty: bool = True,
) -> dict:
    """Generate silver references for a manifest.

    Writes:
        <manifest stem>.silver.jsonl   manifest with silver references filled in
        <manifest stem>.review.jsonl   review queue (one JSON object per line)

    Returns a summary dict.
    """
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    samples = load_manifest(manifest_path)

    out_samples: list[Sample] = []
    review_rows: list[dict] = []
    summary = {"silver": 0, "review": 0, "failed": 0, "skipped_has_reference": 0}
    dead_providers: dict[str, str] = {}

    for sample in samples:
        if only_empty and sample.reference_text:
            summary["skipped_has_reference"] += 1
            out_samples.append(sample)
            continue

        hypotheses: dict[str, str] = {}
        for provider in providers:
            if provider in dead_providers:
                continue
            try:
                record = transcribe_sample(
                    provider, sample, manifest_path.parent, results_dir
                )
            except NotConfigured as exc:
                dead_providers[provider] = str(exc)
                continue
            if record and "error" not in record and record.get("text", "").strip():
                hypotheses[provider] = record["text"].strip()

        if len(hypotheses) < min_hypotheses:
            summary["failed"] += 1
            review_rows.append(
                {
                    "sample_id": sample.id,
                    "audio_path": sample.audio_path,
                    "status": "failed",
                    "reason": f"only {len(hypotheses)} usable hypotheses "
                    f"(need {min_hypotheses})",
                    "hypotheses": hypotheses,
                }
            )
            out_samples.append(sample)
            continue

        # Medoid: hypothesis with the lowest mean symmetric WER to the others.
        names = list(hypotheses)
        mean_dist: dict[str, float] = {}
        for p in names:
            others = [q for q in names if q != p]
            mean_dist[p] = sum(
                _symmetric_wer(hypotheses[p], hypotheses[q]) for q in others
            ) / len(others)
        chosen = min(mean_dist, key=mean_dist.get)  # type: ignore[arg-type]

        if mean_dist[chosen] <= threshold:
            summary["silver"] += 1
            out_samples.append(
                Sample(
                    id=sample.id,
                    audio_path=sample.audio_path,
                    reference_text=hypotheses[chosen],
                    language=sample.language,
                    code_mixed=sample.code_mixed,
                    num_speakers=sample.num_speakers,
                    speaker_segments=sample.speaker_segments,
                    source=f"{sample.source}:silver({chosen},d={mean_dist[chosen]:.3f})",
                    license=sample.license,
                    split=sample.split,
                )
            )
        else:
            summary["review"] += 1
            review_rows.append(
                {
                    "sample_id": sample.id,
                    "audio_path": sample.audio_path,
                    "status": "review",
                    "mean_distance": round(mean_dist[chosen], 4),
                    "best_provider": chosen,
                    "hypotheses": hypotheses,
                    "pairwise_mean_distances": {
                        p: round(d, 4) for p, d in mean_dist.items()
                    },
                }
            )
            out_samples.append(sample)

    silver_path = manifest_path.with_suffix(".silver.jsonl")
    write_manifest(out_samples, silver_path)
    review_path = manifest_path.with_suffix(".review.jsonl")
    with review_path.open("w", encoding="utf-8") as fh:
        for row in review_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["out_manifest"] = str(silver_path)
    summary["review_queue"] = str(review_path)
    summary["dead_providers"] = dead_providers
    return summary
