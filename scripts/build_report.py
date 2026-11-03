"""Build the HTML showcase: inject report/data.json into the template.

Per-case explorer data lives in report/cases.js (built by export_explorer.py)
and is loaded by the page via a script tag, which works over file://.

Refuses to publish private-split numbers: if data.json carries any
"private" split key, the output must be under data/private/reports/.

Usage: uv run python scripts/build_report.py [--out report/index.html]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logatra_benchmark.manifest import require_private_output  # noqa: E402


def has_private(data: dict) -> bool:
    """True if any aggregate in data.json is keyed on the private split."""
    if data.get("comparable_counts", {}).get("private"):
        return True
    for k in ("by_provider", "by_provider_comparable"):
        if any(key.endswith("|private") for key in data.get(k, {})):
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/index.html")
    p.add_argument("--data", default="report/data.json")
    args = p.parse_args()

    data_path = ROOT / args.data
    data = json.loads(data_path.read_text())
    if has_private(data):
        require_private_output(ROOT / args.out)
    cases = (ROOT / args.out).parent / "cases.js"
    if not cases.exists():
        print(f"{cases} missing; run scripts/export_explorer.py first",
              file=sys.stderr)
        return 1
    vendor = ROOT / "report" / "vendor" / "chart.umd.min.js"
    if not vendor.exists():
        print(f"{vendor} missing (vendored Chart.js 4.4.9)", file=sys.stderr)
        return 1

    html = (ROOT / "scripts" / "report_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    if out.parent.resolve() != (ROOT / "report").resolve():
        (out.parent / "vendor").mkdir(exist_ok=True)
        shutil.copy2(vendor, out.parent / "vendor" / vendor.name)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} kB + cases.js)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
