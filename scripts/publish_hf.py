"""Stage and upload the redistributable benchmark data to Hugging Face.

What goes up (and why), see README "Data and licensing":

  dataset repo (default Nemukai/koe-benchmark, type dataset)
    manifests/          committed manifests (Gramvaani rows carry no references)
    audio/fleurs/       CC-BY-4.0
    audio/mucs2021/     CC-BY-SA-4.0
    audio/heldout_yt/   CC-BY-3.0 YouTube chunks only (chunk_*.wav, never full.wav)
    results/<system>/   raw provider responses, minus every Gramvaani sample
    report/             data.json + rows.json (metrics only)
    README.md           dataset card (hf/dataset_card.md)

  space repo (default Nemukai/koe-benchmark, type space, static)
    index.html + cases.js from report/

Gramvaani (OpenSLR 118) is academic-use-only with no redistribution grant and
is excluded everywhere: audio, references and provider outputs.

Usage:
    HF_TOKEN=hf_... uv run python scripts/publish_hf.py --stage-only
    HF_TOKEN=hf_... uv run python scripts/publish_hf.py
    HF_TOKEN=hf_... uv run python scripts/publish_hf.py --dataset Nemukai/koe-benchmark --space Nemukai/koe-benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "local" / "hf_stage"
RESULT_SYSTEMS = ("sarvam-saaras", "sarvam", "whisper", "gemini", "voxtral", "gpt-audio")
MANIFESTS = ("full_public.jsonl", "m1_public.jsonl", "m1_public_interleaved.jsonl",
             "mucs2021_hi_en.jsonl", "heldout_yt.jsonl", "heldout_yt.silver.jsonl",
             "heldout_yt.review.jsonl", "heldout_yt.adjudications.json",
             "heldout_yt.sources.json")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage() -> tuple[Path, Path]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    ds = STAGE / "dataset"
    sp = STAGE / "space"

    # manifests
    for name in MANIFESTS:
        link_or_copy(ROOT / "data/manifests" / name, ds / "manifests" / name)
    for f in sorted((ROOT / "data/manifests").glob("fleurs_*.jsonl")):
        link_or_copy(f, ds / "manifests" / f.name)

    # audio: only redistributable subsets, and held-out chunks only
    n_audio = 0
    for sub in ("fleurs", "mucs2021"):
        for wav in (ROOT / "data/audio" / sub).rglob("*.wav"):
            link_or_copy(wav, ds / "audio" / wav.relative_to(ROOT / "data/audio"))
            n_audio += 1
    for wav in (ROOT / "data/audio/heldout_yt").rglob("chunk_*.wav"):
        link_or_copy(wav, ds / "audio" / wav.relative_to(ROOT / "data/audio"))
        n_audio += 1

    # results: exclude gramvaani
    n_res = n_skip = 0
    for system in RESULT_SYSTEMS:
        for rec in (ROOT / "results" / system).glob("*.json"):
            if rec.name.startswith("gramvaani"):
                n_skip += 1
                continue
            link_or_copy(rec, ds / "results" / system / rec.name)
            n_res += 1

    # report metrics + card
    link_or_copy(ROOT / "report/data.json", ds / "report/data.json")
    link_or_copy(ROOT / "report/rows.json", ds / "report/rows.json")
    shutil.copy2(ROOT / "hf/dataset_card.md", ds / "README.md")
    shutil.copy2(ROOT / "ATTRIBUTION.md", ds / "ATTRIBUTION.md")

    # space
    link_or_copy(ROOT / "report/index.html", sp / "index.html")
    link_or_copy(ROOT / "report/cases.js", sp / "cases.js")
    shutil.copy2(ROOT / "hf/space_card.md", sp / "README.md")

    # sanity: nothing gramvaani, no full.wav
    bad = [p for p in ds.rglob("*") if p.is_file() and ("gramvaani" in p.name or p.name == "full.wav")]
    bad += [p for p in ds.rglob("gramvaani")]
    if bad:
        raise SystemExit(f"staging leaked restricted files: {bad[:5]}")
    for m in (ds / "manifests").glob("*.jsonl"):
        for line in m.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("source") == "gramvaani_telephony" and row.get("reference_text"):
                    raise SystemExit(f"{m.name}: Gramvaani reference leaked ({row['id']})")

    size = sum(p.stat().st_size for p in ds.rglob("*") if p.is_file()) / 1e9
    print(f"staged dataset: {n_audio} audio files, {n_res} result records "
          f"({n_skip} Gramvaani records excluded), {size:.2f} GB -> {ds}")
    print(f"staged space: {sp}")
    return ds, sp


def upload(ds: Path, sp: Path, dataset_id: str, space_id: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    print(f"uploading dataset -> https://huggingface.co/datasets/{dataset_id}")
    api.upload_large_folder(repo_id=dataset_id, repo_type="dataset", folder_path=str(ds))

    api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True)
    print(f"uploading space -> https://huggingface.co/spaces/{space_id}")
    api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(sp),
                      commit_message="Publish benchmark report")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", default="Nemukai/koe-benchmark")
    p.add_argument("--space", default="Nemukai/koe-benchmark")
    p.add_argument("--stage-only", action="store_true")
    args = p.parse_args()

    ds, sp = stage()
    if args.stage_only:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; staged only", file=sys.stderr)
        return 2
    upload(ds, sp, args.dataset, args.space, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
