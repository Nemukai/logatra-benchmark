"""Discover AIR audio bulletins (with their linked text scripts) on newsonair.gov.in.

The site's bulletin custom post type is not exposed via the WordPress REST API
(``/wp-json/wp/v2/media`` returns X-WP-Total but empty bodies for these
attachments), so we crawl the public HTML listings instead:

  /regional-audio/  Regional News Unit (RNU) bulletins, filterable by
                    ?bulletins_lang=<slug>&bulletins_city=<slug>, 12 cards/page,
                    paginated as /regional-audio/page/N/?...
  /nsd-audio/       News Services Division regional-language bulletins
                    (Delhi/pooled), same card markup.

Each card is one CMS bulletin post carrying city/RNU, language, time, date,
the audio URL and (usually) the PDF script URL, so the CMS itself pairs them.

Writes/merges data/private/air/catalog.jsonl (key: post id). Resumable: cards
already in the catalog are kept; re-running only adds new ones.

Usage:
    python scripts/air/discover.py [--since 2026-09-01] [--max-pages 4]
                                   [--langs tamil,odia] [--sources regional-audio,nsd-audio]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
import urllib.parse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from airlib import BASE, CATALOG, get_text, now_iso, read_jsonl, write_jsonl  # noqa: E402

SOURCES = ("regional-audio", "nsd-audio")


def lang_options(page: str) -> list[str]:
    i = page.find('name="bulletins_lang"')
    if i < 0:
        return []
    j = page.find("</select>", i)
    return [v for v in re.findall(r'<option value="([^"]+)"', page[i:j]) if v]


def parse_cards(page: str, source: str, listing_url: str) -> list[dict]:
    out = []
    for c in page.split('class="bulletin-card')[1:]:
        c = c.split('class="bulletin-card')[0]
        sid = re.search(r"share-bulletin%2F%3Fid%3D(\d+)", c)
        audio = re.search(r'class="bulletin-icon">\s*<a[^>]*href="([^"]+)"', c)
        pdf = re.search(r'class="bulletin-download">\s*<a href="([^"]*)"', c)
        city = re.search(r'class="city">(.*?)</span>', c, re.S)
        lang = re.search(r'class="lang">(.*?)</span>', c, re.S)
        tm = re.search(r'class="bulletin-time">(.*?)</span>', c, re.S)
        dd = re.search(r'class="bulletin-date">(.*?)</span>', c, re.S)
        if not (sid and audio and dd):
            continue
        try:
            date = dt.datetime.strptime(html.unescape(dd.group(1)).strip(), "%d %B %Y").date().isoformat()
        except ValueError:
            date = None
        out.append({
            "post_id": int(sid.group(1)),
            "source": source,
            "city": html.unescape(city.group(1)).strip() if city else "",
            "lang_label": html.unescape(lang.group(1)).strip() if lang else "",
            "time": html.unescape(tm.group(1)).strip() if tm else "",
            "date": date,
            "audio_url": html.unescape(audio.group(1)).strip(),
            "pdf_url": html.unescape(pdf.group(1)).strip() if pdf and pdf.group(1).strip() else None,
            "listing_url": listing_url,
            "seen_at": now_iso(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-09-01", help="stop paging a language once cards are older")
    ap.add_argument("--max-pages", type=int, default=4, help="max listing pages per (source, language)")
    ap.add_argument("--langs", default="", help="comma list of site lang slugs (default: all in dropdown)")
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()

    catalog = {r["post_id"]: r for r in read_jsonl(CATALOG)}
    n0 = len(catalog)
    for source in args.sources.split(","):
        first = get_text(f"{BASE}/{source}/")
        langs = [l for l in args.langs.split(",") if l] or lang_options(first)
        print(f"[{source}] {len(langs)} language filters: {', '.join(langs)}", file=sys.stderr)
        for lang in langs:
            for pg in range(1, args.max_pages + 1):
                q = urllib.parse.urlencode({"bulletins_lang": lang})
                url = f"{BASE}/{source}/" + (f"page/{pg}/" if pg > 1 else "") + f"?{q}"
                try:
                    cards = parse_cards(get_text(url), source, url)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {url}: {e}", file=sys.stderr)
                    break
                new = 0
                for c in cards:
                    if c["post_id"] not in catalog:
                        catalog[c["post_id"]] = c
                        new += 1
                oldest = min((c["date"] for c in cards if c["date"]), default=None)
                print(f"  {source} {lang} p{pg}: {len(cards)} cards ({new} new), oldest {oldest}", file=sys.stderr)
                if len(cards) < 12 or (oldest and oldest < args.since):
                    break
            write_jsonl(CATALOG, sorted(catalog.values(), key=lambda r: -r["post_id"]))
    print(f"catalog: {n0} -> {len(catalog)} cards at {CATALOG}")


if __name__ == "__main__":
    main()
