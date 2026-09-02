"""Restore Gramvaani reference transcripts into the committed manifests.

The Gramvaani Hindi corpus (OpenSLR 118) is licensed for academic use only
and grants no redistribution right, so the manifests in this repository ship
Gramvaani rows with an empty ``reference_text``. Scoring skips rows without a
reference. To reproduce the Gramvaani numbers:

    uv run python scripts/prepare_gramvaani.py     # downloads OpenSLR 118 yourself
    uv run python scripts/merge_gramvaani_refs.py  # fills refs into the manifests

The merge is keyed on sample id and touches only rows whose source is
``gramvaani_telephony``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SOURCE = MANIFESTS / "gramvaani_hi.jsonl"
TARGETS = ("full_public.jsonl", "m1_public.jsonl", "m1_public_interleaved.jsonl")


def main() -> int:
    if not SOURCE.exists():
        print(f"{SOURCE} missing; run scripts/prepare_gramvaani.py first", file=sys.stderr)
        return 1
    refs = {}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            refs[row["id"]] = row["reference_text"]
    for name in TARGETS:
        path = MANIFESTS / name
        if not path.exists():
            continue
        out, filled = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == "gramvaani_telephony" and not row.get("reference_text"):
                ref = refs.get(row["id"])
                if ref:
                    row["reference_text"] = ref
                    filled += 1
            out.append(json.dumps(row, ensure_ascii=False))
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{name}: filled {filled} Gramvaani references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
