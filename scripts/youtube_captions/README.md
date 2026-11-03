# YouTube manual-caption collector (private split)

Builds a **private** multi-language evaluation set from Creative Commons
(CC-BY) YouTube videos that carry **uploader-provided** captions in the spoken
language. The captions become `reference_text`; no ASR (paid or otherwise) is
used anywhere.

Languages: `hi bn ta te mr gu kn ml pa or as ur`, plus `hi-en` (Hinglish) and
`en-IN` (Indian English).

Everything is written under `data/private/` (gitignored, never committed):

```
data/private/youtube_captions/
    yt.sqlite                        candidates, videos (status + reasons), chunks
    index.jsonl                      one row per accepted video, with its chunks
    sources.json                     attribution: title, channel, url, date, license
    discover.log / probe.log ...     run logs
    {lang}/{video_id}/info.json      trimmed yt-dlp metadata
    {lang}/{video_id}/captions.<track>.<ext>   original manual caption file
    {lang}/{video_id}/chunks/{video_id}_NNNN.wav   16 kHz mono s16 chunks
data/private/manifests/youtube_captions.jsonl    benchmark manifest rows
```

## Usage

Run from the repo root (no `pyproject.toml` changes needed):

```bash
Y="uv run --with yt-dlp --with yt-dlp-ejs python scripts/youtube_captions/ytcap.py"
$Y discover                 # YouTube search, CC + subtitles filters, per language
$Y probe                    # metadata + caption checks -> accepted / rejected
$Y fetch                    # audio for accepted videos -> cue-aligned WAV chunks
$Y export                   # index.jsonl, sources.json, manifest
$Y status                   # per-language table + rejection reasons + disk use
$Y all --langs ta te        # everything, restricted to some languages
```

Every step is resumable: searches already run, videos already probed and audio
already fetched are skipped (`blocked`/`error` videos are retried). Use
`--player-clients tv,web_safari` to try other yt-dlp player clients.

## Pipeline and checks

1. **Discover** (`discover`): native-script and English queries per language
   (`config.LANGS[*].queries`) against YouTube search with the `sp` filter for
   *Creative Commons + Subtitles/CC + Video*, both "this year" by relevance and
   "sort by upload date". Flat results only (no per-video requests).
2. **Probe** (`probe`), per candidate, round-robin over channels:
   - `license` must contain "Creative Commons" (YouTube's CC-BY 3.0 flag).
   - `upload_date >= 2026-01-01`; `>= 2026-09-01` is `fresh` (after Sarvam
     saaras v4), earlier 2026 uploads are kept but flagged `older`.
   - 2 min to 2 h; not live; no song/film/trailer/jukebox titles; not the
     YouTube *Music* category.
   - **Manual** subtitles only (`info["subtitles"]`, never
     `automatic_captions`) whose track language matches the target.
   - Spoken-language signal: YouTube's original-language ASR track (`xx-orig`
     in `automatic_captions`, only its *name* is read) or `info["language"]`
     must match, which rejects translated caption tracks. When no signal is
     available, videos with manual tracks in >= 3 languages are rejected as
     probable translation sets.
   - Caption text: >= 60% of letters in the expected script (Hinglish may be
     Devanagari or Latin; romanised tracks need >= 8% Hindi function words);
     >= 10 speech cues; <= 30% music cues; cue coverage >= 35% of the video;
     5 to 35 letters per captioned second and >= 2.5 letters per second
     overall (summaries and partial translations fall far below). Rates use
     letters, not words, because Dravidian words are long.
   - Hindi vs Hinglish is decided from the caption text (Latin-word share in
     Devanagari tracks, Hindi function words in romanised tracks).
   - At most 2 videos per channel per language; up to 15 videos per language.
3. **Fetch** (`fetch`): `bestaudio` -> ffmpeg 16 kHz mono `pcm_s16le` (the
   scorer estimates duration as `filesize / 32000`). Consecutive cues are
   merged up to 28 s (hard max 30 s), never splitting a cue; a gap > 1.5 s
   or a music / empty cue ends a chunk (the music cue is dropped). 0.15 s
   padding is added but never reaches into a neighbouring cue. At most 6 min
   of chunks per video, spread evenly over the video. The full-length audio is
   deleted after cutting; the original caption file is kept.
4. **Export** (`export`): manifest rows follow `logatra_benchmark/manifest.py`
   (`source = "youtube_captions:<id>:manual"`, `license = "CC-BY 3.0
   (YouTube), private split"`, `split = "private"`, audio paths relative to
   the manifest). Override the split with `--split heldout` if the loader in
   use does not yet accept `"private"`.

`reference_text` = cue texts joined with spaces, with WebVTT tags, `>>`/`-`
speaker turns, `Name:` prefixes and bracketed events (`[laughter]`,
`(applause)`) removed. `num_speakers` is a best guess (speaker markers in the
chunk, and "podcast/interview/..." in the title).

Offline check of a caption file: `uv run python scripts/youtube_captions/captions.py FILE.vtt --lang ta --duration 600`.

## Politeness and blocking

3 to 7 s between searches, 4 to 9 s between videos, `sleep_interval_requests`
1.5 s inside yt-dlp. No cookies, accounts or proxies. After 4 consecutive bot
checks ("Sign in to confirm you're not a bot", HTTP 429) the step stops and
the affected videos stay `blocked` in the DB, to be retried later or from a
residential network.

## Caveats

- Manual captions are not verbatim gold: uploaders clean up disfluencies,
  drop fillers, normalise numbers, sometimes paraphrase, and cue timing can
  be loose. Treat WER against these references as an approximate signal and
  human-review the public sample before release.
- The CC flag is set by uploaders, who occasionally mislabel content they do
  not own; spot-check before any public release.
