"""Render the static figures for the Hugging Face dataset card and Space.

Reads report/data.json and writes SVGs plus 2x PNGs to hf/assets/. No plotting
dependency: the charts are plain SVG, rasterised with rsvg-convert when it is
on PATH (brew install librsvg).

    uv run python scripts/hf_figures.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf" / "assets"
LOGO = ROOT / "hf" / "logatra-logo.svg"

SAND, SAND2, TRACK = "#F2EBDD", "#ECE4D4", "#DED5C3"
INK, INK2, MUTED, SUBTLE = "#141210", "#3B352E", "#524A3E", "#8E8474"
KAI, KAI_SOFT = "#1F869A", "#9FCFD8"
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', Menlo, monospace"

# fixed display order: best clean held-out score first
SYSTEMS = [
    ("sarvam-saaras", "Sarvam saaras:v3"),
    ("sarvam-saaras-v4", "Sarvam saaras:v4"),
    ("gemini", "Gemini 3.5 Flash"),
    ("sarvam", "Sarvam saarika:v2.5"),
    ("gpt-audio", "OpenAI gpt-audio"),
    ("whisper", "OpenAI whisper-1"),
    ("voxtral", "Mistral Voxtral Small"),
]
LANGS = [("hi", "Hindi"), ("bn", "Bengali"), ("mr", "Marathi"), ("gu", "Gujarati"),
         ("pa", "Punjabi"), ("ta", "Tamil"), ("te", "Telugu"), ("kn", "Kannada"),
         ("ml", "Malayalam"), ("or", "Odia"), ("as", "Assamese"), ("ur", "Urdu"),
         ("hi-en", "Hinglish")]


def svg(w: int, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="{SAND}"/>{body}</svg>')


def text(x, y, s, size=14, fill=INK, family=SANS, anchor="start", weight=400, extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{escape(str(s))}</text>')


def header(title: str, sub: str) -> str:
    return text(48, 64, title, 30, INK, SERIF) + text(48, 92, sub, 14, MUTED)


def footer(w: int, h: int, note: str) -> str:
    return (f'<line x1="48" x2="{w - 48}" y1="{h - 44}" y2="{h - 44}" stroke="{TRACK}"/>'
            + text(48, h - 20, note, 12, SUBTLE)
            + text(w - 48, h - 20, "Logatra Benchmark · nemukai.com/labs", 12, SUBTLE, anchor="end"))


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def hatch_defs() -> str:
    return (f'<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{KAI_SOFT}"/>'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{KAI}" stroke-width="1.5"/></pattern></defs>')


def bar(x, y, w, h, fill):
    # square at the baseline, 4px rounded data end
    r = min(4, w / 2, h / 2)
    if w <= 0:
        return ""
    return (f'<path d="M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} '
            f'Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z" fill="{fill}"/>')


def logo_inner() -> tuple[str, str]:
    raw = LOGO.read_text(encoding="utf-8")
    vb = re.search(r'viewBox="([^"]+)"', raw).group(1)
    inner = re.sub(r"^.*?<svg[^>]*>|</svg>\s*$", "", raw, flags=re.S)
    return vb, inner


def hero(d: dict) -> str:
    w, h = 1200, 372
    vb, inner = logo_inner()
    n_clips = d["sample_count"]
    body = (f'<defs><clipPath id="tile"><rect x="64" y="56" width="148" height="148" rx="28"/></clipPath></defs>'
            f'<g clip-path="url(#tile)"><svg x="64" y="56" width="148" height="148" viewBox="{vb}">{inner}</svg></g>'
            + text(248, 110, "Logatra Benchmark", 64, INK, SERIF)
            + text(250, 152, "Speech-to-text on Indian languages and Hinglish, measured independently.", 21, INK2)
            + text(250, 186, "Built by Nemukai, the team behind Logatra.", 16, MUTED))
    stats = [("7", "STT engines"), ("12 + 1", "Indic languages + Hinglish"),
             (f"{n_clips:,}", "clips scored"), ("48 min", "fresh 2026 held-out audio")]
    x0, y0, cw = 64, 256, 268
    body += f'<line x1="64" x2="{w - 64}" y1="{y0 - 18}" y2="{y0 - 18}" stroke="{TRACK}"/>'
    for i, (big, small) in enumerate(stats):
        x = x0 + i * cw
        if i:
            body += f'<line x1="{x - 20}" x2="{x - 20}" y1="{y0}" y2="{y0 + 70}" stroke="{TRACK}"/>'
        body += text(x, y0 + 38, big, 34, INK, MONO, weight=500) + text(x, y0 + 66, small, 14, MUTED)
    return svg(w, h, body)


def leaderboard(d: dict) -> str:
    c = d["by_provider_comparable"]
    n_pub, n_ho = d["comparable_counts"]["public"], d["comparable_counts"]["heldout"]
    w, h = 1200, 520
    label_w, gap, panel_w = 210, 56, 430
    x1 = 48 + label_w
    x2 = x1 + panel_w + gap
    top, row = 170, 40
    body = hatch_defs() + header(
        "Word error rate, comparable clips only",
        "Every engine scored on the same clips. Lower is better.")
    for x, title in ((x1, f"Public split · {n_pub} clips"), (x2, f"Held-out 2026 audio · {n_ho} clips")):
        body += text(x, top - 34, title, 15, INK, weight=600)
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            gx = x + t * panel_w
            body += f'<line x1="{gx}" x2="{gx}" y1="{top - 16}" y2="{top + row * len(SYSTEMS) - 8}" stroke="{TRACK}" stroke-width="1"/>'
            body += text(gx, top + row * len(SYSTEMS) + 12, f"{int(t * 100)}%", 11, SUBTLE, MONO, "middle")
    for i, (key, name) in enumerate(SYSTEMS):
        y = top + i * row
        body += text(x1 - 16, y + 15, name, 14, INK, anchor="end")
        for x, split in ((x1, "public"), (x2, "heldout")):
            m = c[f"{key}|{split}"]
            v, self_ref = m["wer"], m["self_ref_rate"] > 0.01
            bw = min(v, 1.0) * panel_w
            body += bar(x, y, bw, 20, "url(#hatch)" if self_ref else KAI)
            label = pct(v) + (" *" if self_ref else "")
            if v > 1.0:
                body += (f'<path d="M{x + panel_w - 10},{y - 3} l8,13 l-8,13" fill="none" stroke="{SAND}" stroke-width="2.5"/>'
                         + text(x + panel_w - 8, y - 6, f"{pct(v)} (repetition loops)", 11, INK2, MONO, "end"))
            else:
                body += text(x + bw + 8, y + 15, label, 12.5, INK2, MONO)
    body += footer(w, h, "* Contributed to the silver held-out references, so its held-out score is an upper bound. "
                         "Hatched bars mark those systems.")
    return svg(w, h, body)


def cost_vs_accuracy(d: dict) -> str:
    c = d["by_provider_comparable"]
    w, h = 1200, 620
    px, py, pw, ph = 120, 140, 960, 380
    xmax, ymax = 1.5, 0.8
    body = header("Accuracy against price",
                  "Held-out word error rate against list price per audio hour. Down and to the left is better.")
    for t in range(0, 9):
        yv = t / 10
        gy = py + ph - yv / ymax * ph
        body += f'<line x1="{px}" x2="{px + pw}" y1="{gy}" y2="{gy}" stroke="{TRACK}"/>'
        body += text(px - 12, gy + 4, f"{int(yv * 100)}%", 11, SUBTLE, MONO, "end")
    for t in range(0, 7):
        xv = t * 0.25
        gx = px + xv / xmax * pw
        body += text(gx, py + ph + 22, f"${xv:.2f}", 11, SUBTLE, MONO, "middle")
    body += text(px + pw / 2, py + ph + 48, "US$ per audio hour", 13, MUTED, anchor="middle")
    body += text(px - 70, py + ph / 2, "Held-out WER", 13, MUTED, anchor="middle",
                 extra=f'transform="rotate(-90 {px - 70} {py + ph / 2})"')
    # (dx, label baseline offset from the saaras:v3 point or None for own point)
    placement = {"sarvam-saaras": (70, 0), "sarvam-saaras-v4": (70, -30), "sarvam": (70, 30),
                 "gemini": (-14, None), "gpt-audio": (-14, None), "whisper": (14, None), "voxtral": (14, None)}
    anchor_y = py + ph - c["sarvam-saaras|heldout"]["wer"] / ymax * ph
    for key, name in SYSTEMS:
        m = c[f"{key}|heldout"]
        cx = px + m["cost_usd_per_hour"] / xmax * pw
        cy = py + ph - m["wer"] / ymax * ph
        self_ref = m["self_ref_rate"] > 0.01
        fill = SAND if self_ref else KAI
        body += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{fill}" stroke="{KAI}" stroke-width="2.5"/>'
        dx, off = placement[key]
        anchor = "start" if dx > 0 else "end"
        ly = cy + 5 if off is None else anchor_y + off + 5
        if off is not None:
            body += (f'<polyline points="{cx + 10:.1f},{cy:.1f} {cx + dx - 26:.1f},{ly - 5:.1f} {cx + dx - 6:.1f},{ly - 5:.1f}" '
                     f'fill="none" stroke="{SUBTLE}" stroke-width="1"/>')
        body += text(cx + dx, ly, f"{name}  {pct(m['wer'])}{' *' if self_ref else ''}", 13, INK2,
                     anchor=anchor)
    body += footer(w, h, "Hollow points contributed to the held-out references (scores are upper bounds). "
                         "Prices are list prices at run time.")
    return svg(w, h, body)


def ramp(v: float) -> tuple[str, str]:
    """Sequential teal ramp, light to dark, capped at 100% WER."""
    lo, hi = (0xE4, 0xF1, 0xF2), (0x0E, 0x45, 0x50)
    t = max(0.0, min(v, 1.0)) ** 0.8
    rgb = [round(a + (b - a) * t) for a, b in zip(lo, hi)]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#%02X%02X%02X" % tuple(rgb), (INK if lum > 140 else SAND)


def language_grid(d: dict) -> str:
    L = d["by_provider_language"]
    w = 1200
    label_w, top, cw, ch = 210, 170, 66, 38
    x0 = 48 + label_w
    h = top + ch * len(SYSTEMS) + 110
    body = header("Script-agnostic WER by language",
                  "All scored clips per engine (FLEURS read speech per language; MUCS 2021 for Hinglish). Lower is better.")
    for j, (_, lname) in enumerate(LANGS):
        body += text(x0 + j * cw + cw / 2, top - 14, lname, 12, MUTED, anchor="middle")
    for i, (key, name) in enumerate(SYSTEMS):
        y = top + i * ch
        body += text(x0 - 16, y + ch / 2 + 5, name, 14, INK, anchor="end")
        for j, (lang, _) in enumerate(LANGS):
            m = L.get(f"{key}|{lang}")
            x = x0 + j * cw
            if not m:
                body += f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{SAND2}"/>'
                body += text(x + cw / 2, y + ch / 2 + 4, "not run", 10, SUBTLE, anchor="middle")
                continue
            v = m["wer_script_agnostic"]
            fill, ink = ramp(v)
            body += f'<rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{fill}"/>'
            body += text(x + cw / 2, y + ch / 2 + 4, ">100" if v > 1 else f"{v * 100:.0f}", 12.5, ink, MONO,
                         "middle", 500)
    ly = top + ch * len(SYSTEMS) + 26
    body += text(x0, ly + 12, "0%", 11, SUBTLE, MONO, "end", extra='dx="-6"')
    for k in range(40):
        fill, _ = ramp(k / 39)
        body += f'<rect x="{x0 + k * 6}" y="{ly}" width="6" height="14" fill="{fill}"/>'
    body += text(x0 + 246, ly + 12, "100%+", 11, SUBTLE, MONO)
    body += footer(w, h, "Sample counts differ by engine: OpenRouter engines ran a subset, gpt-audio only a few languages. "
                         "Counts in report/data.json.")
    return svg(w, h, body)


def main() -> int:
    d = json.loads((ROOT / "report/data.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOGO, OUT / "logatra-logo.svg")
    figures = {"hero": hero(d), "leaderboard": leaderboard(d),
               "cost-vs-accuracy": cost_vs_accuracy(d), "languages": language_grid(d)}
    rsvg = shutil.which("rsvg-convert")
    for name, doc in figures.items():
        (OUT / f"{name}.svg").write_text(doc, encoding="utf-8")
        if rsvg:
            subprocess.run([rsvg, "-z", "2", "-o", str(OUT / f"{name}.png"), str(OUT / f"{name}.svg")], check=True)
    if rsvg:
        subprocess.run([rsvg, "-w", "512", "-h", "512", "-o", str(OUT / "logatra-logo.png"), str(LOGO)], check=True)
    print(f"wrote {len(figures)} figures to {OUT}" + ("" if rsvg else " (SVG only: rsvg-convert not found)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
