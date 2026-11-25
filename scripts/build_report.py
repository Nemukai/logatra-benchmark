"""Build the HTML showcase: inject report/data.json into the template.

Per-case explorer data lives in report/cases.js (built by export_explorer.py)
and is loaded by the page via a script tag, which works over file://.

Two targets, one template:

- standalone (default): report/index.html as committed. Works over file://
  and on the Hugging Face Space. cases.js sits beside the page, and the
  header links point at https://nemukai.com.
- site: the copy served at nemukai.com/labs/logatra-benchmark. Root-relative
  links, and cases.js by absolute path because the page is served without a
  trailing slash. Write it straight into the Website repo:

    uv run python scripts/build_report.py --target site \
        --out ../Website/public/labs/logatra-benchmark/index.html

Usage: uv run python scripts/build_report.py [--target standalone|site] [--out report/index.html]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_SLOT = "/*__DATA__*/null"

# What each placeholder in the template becomes, per target.
TARGETS = {
    "standalone": {
        "__SITE__": "https://nemukai.com",
        # The Space renders the page in an iframe; site links must leave it.
        "__TOP__": ' target="_top"',
        "__CASES_SRC__": "cases.js",
    },
    "site": {
        "__SITE__": "",
        "__TOP__": "",
        "__CASES_SRC__": "/labs/logatra-benchmark/cases.js",
    },
}


def render(template: str, data: dict, target: str = "standalone") -> str:
    """Fill the template for one target. Raises if a slot is missing or left over."""
    if DATA_SLOT not in template:
        raise ValueError(f"template has no {DATA_SLOT} slot")
    html = template
    for slot, value in TARGETS[target].items():
        if slot not in html:
            raise ValueError(f"template has no {slot} slot")
        html = html.replace(slot, value)
    # Data goes in last so nothing inside it can be mistaken for a slot.
    return html.replace(DATA_SLOT, json.dumps(data, ensure_ascii=False))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/index.html")
    p.add_argument("--target", choices=sorted(TARGETS), default="standalone")
    args = p.parse_args()

    data = json.loads((ROOT / "report" / "data.json").read_text())
    if not (ROOT / "report" / "cases.js").exists():
        print("report/cases.js missing; run scripts/export_explorer.py first",
              file=sys.stderr)
        return 1

    template = (ROOT / "scripts" / "report_template.html").read_text()
    html = render(template, data, args.target)

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} kB + cases.js, target={args.target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
