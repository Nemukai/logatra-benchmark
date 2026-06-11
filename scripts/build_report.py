"""Build the HTML showcase: inject report/data.json into the template.

Per-case explorer data lives in report/cases.js (built by export_explorer.py)
and is loaded by the page via a script tag, which works over file://.

Usage: uv run python scripts/build_report.py [--out report/index.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/index.html")
    args = p.parse_args()

    data = json.loads((ROOT / "report" / "data.json").read_text())
    if not (ROOT / "report" / "cases.js").exists():
        print("report/cases.js missing; run scripts/export_explorer.py first",
              file=sys.stderr)
        return 1

    html = (ROOT / "scripts" / "report_template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} kB + cases.js)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
