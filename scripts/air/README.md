# AIR news bulletins (private eval set)

Collects recent All India Radio regional-news bulletins from
`newsonair.gov.in`: the original audio plus the bulletin script (PDF) that
the newsroom wrote for it. The output is a **private** evaluation split
(Prasar Bharati content, not redistributable). It goes under the gitignored
`data/private/` and must never be committed.

## Site structure (as of Nov 2026)

* WordPress + Polylang. `robots.txt` allows everything (`Disallow:` empty).
* Bulletins are a custom post type that the REST API does **not** expose.
  `/wp-json/wp/v2/media?mime_type=application/pdf` reports `X-WP-Total`
  but returns empty bodies, and `/sitemap/` returns 404. The crawler therefore
  reads the HTML listings:
  * `/regional-audio/?bulletins_lang=<slug>&bulletins_city=<slug>&bulletins_date=YYYY-MM-DD`,
    paginated as `/regional-audio/page/N/?...`, 12 cards per page, back to 2022-06.
    Each card is one bulletin post (`/share-bulletin/?id=<post_id>`) and holds
    the RNU/city, language, time, date, the audio URL (`.mp3`) and usually the
    script PDF URL. **The CMS does the pairing**, so we do not have to match
    files by name.
  * `/nsd-audio/` has the same markup for NSD (Delhi) regional-language bulletins.
    In its cards, "city" holds the language name.
  * `/bulletins-category/regional-text/` and `/bulletins-category/nsd-text/`
    are table views of the same PDFs (RNU, language, time, PDF, date).
  * Files live under `/wp-content/uploads/YYYY/MM/<free-form-name>.{mp3,pdf}`.
    Uploaders choose the file names, which are often misleading
    (`agara-1945.mp3` for a 19:55 slot), so we never pair by name.
* The language dropdown has about 47 entries. A few are city names entered by
  mistake (`bhuj`, `chennai`, `jammu`, `dibrugarh`). `airlib.resolve` fixes
  swapped city and language labels.

## Pipeline

```bash
# 1. crawl listings -> data/private/air/catalog.jsonl (stdlib only)
python scripts/air/discover.py --since 2026-09-01 --max-pages 4

# 2. pick, download, convert, extract, check pairing (needs pymupdf; pdftotext optional)
uv run --no-project --with pymupdf python scripts/air/fetch.py --per-lang 6 --budget-gb 7.5
#    more for selected languages later:
uv run --no-project --with pymupdf python scripts/air/fetch.py --per-lang 12 --langs hindi,bengali,tamil

# 3. index + sqlite + manifest
python scripts/air/build.py

# (optional) re-run text extraction after improving airtext.py, offline
uv run --no-project --with pymupdf python scripts/air/reextract.py
```

Every step can be resumed and re-run safely. `discover` merges cards into
the catalog. `fetch` skips any post that already has an
`items/<id>.json` (kept *or* rejected), resumes partial downloads with HTTP
Range, and stops at the disk budget. Crawling is polite: one connection, at
least 1.5 s plus jitter between requests, and a generic research User-Agent.

## Layout

```
data/private/air/
  catalog.jsonl                 every listing card seen
  items/<id>.json               per-bulletin record (source of truth)
  <lang>/<id>/audio.mp3         original audio
  <lang>/<id>/script.pdf        original script
  <lang>/<id>/audio_16k.wav     16 kHz mono s16 (duration = (bytes-44)/32000)
  <lang>/<id>/script.txt        extracted, cleaned text
  index.jsonl, air.sqlite       one row per kept bulletin (table `bulletins`)
  rejected.jsonl                dropped candidates and reasons
  manifest_with_suspect_text.jsonl
data/private/manifests/air.jsonl   repo manifest (split "private")
```

`id` = `air_<lang>_<rnu>_<YYYYMMDD>_<HHMM>_<post_id>`.

## Pairing evidence (`pairing_evidence` in index)

* `cms_same_post`: the audio and the PDF are linked from the same bulletin post.
  This is the primary evidence.
* `pdf_header_date_match`: the date printed in the script header (any Indic
  digit system) equals the listed date. This yields `confidence: high`.
* `pdf_creation_date` within -1..+1 day of the bulletin yields `medium` when the
  header date cannot be read (for example, legacy fonts).
* A candidate is rejected if its audio or PDF URL is reused by another post,
  if the header date mismatches, if the PDF is far older than the bulletin with no
  readable header date, or if the audio is shorter than 2 min or longer than 40 min.
* Time tokens in file names are recorded for information only.

## Text quality (`text_status`)

| status | meaning |
|---|---|
| `ok` | expected Unicode script, vowel-sign sequences well formed |
| `unicode_glyphmap_suspect` | Unicode script, but the PDF font's ToUnicode map mangles reordered/conjunct glyphs (orphan vowel signs, missing pre-base matras, Greek/PUA code points). Readable, with systematic character errors. |
| `legacy_font_unconverted` | non-Unicode font (e.g. AkrutiAsmAnand for Assamese, Bengali-script fonts for Manipuri). The extracted text is Latin-1 junk. The PDF is kept for OCR or conversion later. |
| `no_text_layer` | scanned/image PDF |
| `wrong_script` | clean text in an unexpected script |

**Legacy-font conversion.** `krutidev.py` converts Kruti Dev / KrutiPad
(Devanagari) spans to Unicode. It works per span and leaves English headers
in Times/Arial untouched. The Hindi-belt RNUs (Gorakhpur Bhojpuri, Raipur,
Jaipur, Patna, Lucknow) type their scripts in Kruti Dev, so their text comes
out clean (`text_extractor: pymupdf+krutidev`). We have no converter yet for
AkrutiAsmAnand (Assamese), AkPOri* (Odia/Sambalpuri), Nudi (Kannada),
Shipmoo (Lepcha) or the Bengali-script fonts used for Manipuri.

**Audio without scripts.** The listings carry audio but never a PDF for Bodo,
Garo, Jaintia, Karbi, Kashmiri, Khasi, Kokborok, Ladakhi, Nagamese, Pahari,
Sindhi, and almost never for Garhwali or Kumauni. `regional-text` lists no
scripts for them either.

The strict manifest `data/private/manifests/air.jsonl` fills
`reference_text` only for `ok` rows. Other rows have `""`, meaning not yet
transcribed. `manifest_with_suspect_text.jsonl` also includes the suspect
texts.

## Caveats and next steps

* Scripts are **not verbatim**. Anchors add greetings and sign-offs, repeat
  headlines at the end, skip or reorder items, and read numbers and
  abbreviations aloud. Some PDFs contain extra items that were never read.
  The Hindi and English bulletins of some RNUs also mix in other languages.
* Bulletins run 5 to 15 min. **Do not score them as whole files.** The next
  step is alignment followed by chunking to 30 s or less.
  `scripts/segment_audio.py` only cuts at silences and does not know the text,
  so it is not enough on its own. The suggested approach is:
  1. Split the script into sentences. Drop header lines (RNU name, date, reader).
  2. Run CTC forced alignment with an open multilingual acoustic model, for
     example torchaudio `MMS_FA` or `ctc-forced-aligner` (MMS-1b plus uroman
     romanisation, which covers every script here). Use a "star"/garbage
     token so script sentences that were not read can be skipped.
  3. Keep sentence groups whose per-frame alignment score passes a threshold.
     Merge them into windows of 30 s or less, cut at silence midpoints
     (reuse `detect_silences` from `segment_audio.py`), then spot-check by ear.
  4. For `legacy_font_unconverted` and `unicode_glyphmap_suspect` PDFs, first
     re-extract the text with OCR (Tesseract or IndicOCR on 300-dpi renders of
     the PDF pages), or use a font-specific legacy-to-Unicode table (Akruti or
     Shree-Lipi to Unicode), then re-run `reextract.py`.
* The manifest uses `split: "private"`. `logatra_benchmark/manifest.py`
  must list it in `VALID_SPLITS` before the loader will accept these rows.
