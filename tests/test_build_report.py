"""The report template builds cleanly for both targets."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_report  # noqa: E402

TEMPLATE = (ROOT / "scripts" / "report_template.html").read_text()
DATA = {"by_provider": {"sarvam|public": {"samples": 3}}, "note": "__SITE__ stays as data"}


def embedded_data(html: str) -> dict:
    line = next(l for l in html.splitlines() if l.startswith("const DATA = "))
    return json.loads(line.removeprefix("const DATA = ").removesuffix(";"))


@pytest.mark.parametrize("target", sorted(build_report.TARGETS))
def test_no_slot_survives_and_data_round_trips(target):
    html = build_report.render(TEMPLATE, DATA, target)
    outside_data = html.replace(json.dumps(DATA, ensure_ascii=False), "")
    assert not re.search(r"__[A-Z_]+__", outside_data)
    assert embedded_data(html) == DATA


def test_standalone_sits_beside_its_cases_and_links_out():
    html = build_report.render(TEMPLATE, DATA, "standalone")
    assert '<script src="cases.js"></script>' in html
    assert 'href="https://nemukai.com/labs" target="_top"' in html
    # No root-relative link: those break on the Space and over file://.
    assert not re.search(r'(?:href|src)="/', html)


def test_site_uses_root_relative_paths():
    html = build_report.render(TEMPLATE, DATA, "site")
    assert '<script src="/labs/logatra-benchmark/cases.js"></script>' in html
    assert 'href="/labs"' in html
    assert "target=\"_top\"" not in html
    assert "https://nemukai.com" not in html.replace(json.dumps(DATA, ensure_ascii=False), "")


def test_missing_slot_is_an_error():
    with pytest.raises(ValueError):
        build_report.render(TEMPLATE.replace("__CASES_SRC__", "cases.js"), DATA)


def test_page_copy_has_no_em_dash():
    assert "—" not in re.sub(r"\.replace\(/\[.*?\]/g", "", TEMPLATE)
