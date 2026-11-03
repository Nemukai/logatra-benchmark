"""Runner spend controls: cache skipping, budget cap, unsupported skips,
retries, private-split guard, and the CLI's --yes-spend / --dry-run / estimate.
All offline."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import httpx
import pytest

from logatra_benchmark import __main__ as cli
from logatra_benchmark import runner
from logatra_benchmark.adapters import Transcription, TranscriptionError, get_adapter


def make_wav(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(seconds * 16000))
    return path


def write_manifest(tmp_path: Path, rows: list[dict], name: str = "m.jsonl") -> Path:
    m = tmp_path / name
    for r in rows:
        make_wav(tmp_path / r["audio_path"], r.pop("_secs", 6.0))
    m.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return m


def row(i: int, lang: str = "hi", **kw) -> dict:
    d = {"id": f"s{i}", "audio_path": f"a/s{i}.wav", "reference_text": "एक दो तीन",
         "language": lang, "split": "public"}
    d.update(kw)
    return d


class FakeAdapter:
    """Counts calls; behaves like a list-priced provider ('whisper' pricing)."""

    name = "whisper"
    required_env = ()
    model = "fake-1"

    def __init__(self, fail_first: int = 0, fail_msg: str = "OpenAI HTTP 429 for x: slow"):
        self.calls = 0
        self.fail_first = fail_first
        self.fail_msg = fail_msg

    def supports(self, language, code_mixed=False):
        return "language 'ml' not supported" if language == "ml" else None

    def transcribe(self, audio_path, language=None):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise TranscriptionError(self.fail_msg)
        return Transcription(text="एक दो तीन", raw={"ok": True})


def test_cache_is_never_rebilled(tmp_path):
    m = write_manifest(tmp_path, [row(1), row(2)])
    res = tmp_path / "results"
    fake = FakeAdapter()
    r1 = runner.run_benchmark(m, ["whisper"], res, adapters={"whisper": fake})
    assert fake.calls == 2 and r1["rows"][0]["wer"] == 0.0
    cached = json.loads((res / "whisper" / "s1.json").read_text())
    assert cached["cost_usd"] == pytest.approx(0.006 * 6 / 60)
    assert cached["audio_s"] == pytest.approx(6.0) and cached["model"] == "fake-1"
    r2 = runner.run_benchmark(m, ["whisper"], res, adapters={"whisper": fake})
    assert fake.calls == 2  # nothing re-sent
    assert r2["budget"]["spent_usd"] == 0.0
    assert r2["rows"][0]["samples"] == 2


def test_budget_cap_stops_before_exceeding(tmp_path):
    # whisper list price $0.006/min; 60 s clips -> $0.006 each; cap $0.013 -> 2 calls.
    m = write_manifest(tmp_path, [row(i, _secs=60.0) for i in range(5)])
    fake = FakeAdapter()
    rep = runner.run_benchmark(m, ["whisper"], tmp_path / "r", max_usd=0.013,
                               adapters={"whisper": fake})
    assert fake.calls == 2
    assert rep["budget"]["stopped"] is True
    assert rep["budget"]["spent_usd"] <= 0.013
    assert rep["budget"]["samples_not_sent"]["whisper"] == 3
    assert rep["errors"] == {}


def test_budget_cap_with_concurrency(tmp_path):
    m = write_manifest(tmp_path, [row(i, _secs=60.0) for i in range(8)])
    fake = FakeAdapter()
    rep = runner.run_benchmark(m, ["whisper"], tmp_path / "r", max_usd=0.02, workers=4,
                               adapters={"whisper": fake})
    assert fake.calls == 3
    assert rep["budget"]["spent_usd"] <= 0.02


def test_unsupported_language_is_skipped_not_error(tmp_path):
    m = write_manifest(tmp_path, [row(1), row(2, lang="ml")])
    fake = FakeAdapter()
    rep = runner.run_benchmark(m, ["whisper"], tmp_path / "r", adapters={"whisper": fake})
    assert fake.calls == 1
    assert rep["errors"] == {}
    assert sum(rep["unsupported"]["whisper"].values()) == 1
    assert not (tmp_path / "r" / "whisper" / "s2.json").exists()


def test_retries_with_backoff_then_succeeds(tmp_path):
    m = write_manifest(tmp_path, [row(1)])
    fake = FakeAdapter(fail_first=2)
    sleeps: list[float] = []
    rep = runner.run_benchmark(m, ["whisper"], tmp_path / "r", adapters={"whisper": fake},
                               backoff=(1.0, 2.0, 3.0), sleep=sleeps.append)
    assert fake.calls == 3 and sleeps == [1.0, 2.0]
    assert rep["errors"] == {}


def test_non_retryable_error_not_cached(tmp_path):
    m = write_manifest(tmp_path, [row(1)])
    fake = FakeAdapter(fail_first=99, fail_msg="OpenAI HTTP 400 for x: bad")
    rep = runner.run_benchmark(m, ["whisper"], tmp_path / "r", adapters={"whisper": fake},
                               sleep=lambda s: None)
    assert fake.calls == 1 and rep["errors"] == {"whisper": 1}
    assert not (tmp_path / "r" / "whisper" / "s1.json").exists()


def test_limit_restricts_samples(tmp_path):
    m = write_manifest(tmp_path, [row(i) for i in range(5)])
    fake = FakeAdapter()
    runner.run_benchmark(m, ["whisper"], tmp_path / "r", limit=2, adapters={"whisper": fake})
    assert fake.calls == 2


def test_missing_env_skips_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    m = write_manifest(tmp_path, [row(1)])
    rep = runner.run_benchmark(m, ["deepgram"], tmp_path / "r")
    assert "DEEPGRAM_API_KEY" in rep["skipped"]["deepgram"]


def test_end_to_end_deepgram_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    m = write_manifest(tmp_path, [row(1, lang="hi", code_mixed=True), row(2, lang="or")])
    seen = []

    def handler(req):
        seen.append(dict(req.url.params.multi_items()))
        return httpx.Response(200, json={"results": {"channels": [{"alternatives": [
            {"transcript": "एक दो तीन", "words": []}]}]}})

    ad = get_adapter("deepgram", client=httpx.Client(transport=httpx.MockTransport(handler)))
    rep = runner.run_benchmark(m, ["deepgram"], tmp_path / "r", adapters={"deepgram": ad})
    assert len(seen) == 1 and seen[0]["language"] == "multi"
    rec = json.loads((tmp_path / "r" / "deepgram" / "s1.json").read_text())
    assert rec["cost_usd"] == pytest.approx(0.0052 * 6 / 60)  # multilingual rate
    assert rep["unsupported"]["deepgram"]


def test_private_split_refuses_public_results_dir(tmp_path):
    m = write_manifest(tmp_path, [row(1, split="private")])
    with pytest.raises(ValueError, match="private"):
        runner.run_benchmark(m, ["whisper"], tmp_path / "results", adapters={"whisper": FakeAdapter()})


def test_default_results_dir_for_private_manifest():
    from logatra_benchmark.manifest import PRIVATE_DIR

    d = runner.default_results_dir([PRIVATE_DIR / "manifests" / "air.jsonl"])
    assert d == PRIVATE_DIR / "results"
    assert runner.default_results_dir(["data/manifests/m1_public.jsonl"]) == Path("results")
    assert runner.default_results_dir(["x.jsonl"], split="private") == PRIVATE_DIR / "results"


# --- planning / CLI ---------------------------------------------------------------

@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)


def test_plan_counts_cache_unsupported_and_cost(tmp_path, no_network):
    m = write_manifest(tmp_path, [row(1, _secs=60.0), row(2, lang="ml", _secs=60.0),
                                  row(3, _secs=120.0)])
    res = tmp_path / "r"
    (res / "deepgram").mkdir(parents=True)
    (res / "deepgram" / "s3.json").write_text("{}")
    p = runner.plan([m], ["deepgram", "gemini"], res)
    dg = p["providers"]["deepgram"]
    assert dg["cached"] == 1 and dg["to_call"] == 1 and dg["unsupported"] == {"ml": 1}
    assert dg["est_usd"] == pytest.approx(0.0043)
    assert p["audio_minutes"] == pytest.approx(4.0)
    assert p["providers"]["gemini"]["est_usd"] is None
    assert p["providers"]["gemini"]["est_usd_rough"] > 0


def test_plan_assumes_duration_for_missing_audio(tmp_path, no_network):
    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps(row(1)) + "\n")
    p = runner.plan([m], ["whisper"], tmp_path / "r", assume_seconds=30)
    assert p["missing_audio_files"] == 1
    assert p["providers"]["whisper"]["est_usd"] == pytest.approx(0.003)


def test_cli_estimate_offline(tmp_path, capsys, no_network):
    m = write_manifest(tmp_path, [row(1, _secs=60.0)])
    rc = cli.main(["estimate", "--manifest", str(m), "--providers", "deepgram,sarvam-saaras-v4",
                   "--results-dir", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert rc == 0 and "deepgram" in out and "audio: 1.0 min" in out


def test_cli_run_requires_yes_spend(tmp_path, capsys, monkeypatch, no_network):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    monkeypatch.delenv("LOGATRA_YES_SPEND", raising=False)
    m = write_manifest(tmp_path, [row(1)])
    rc = cli.main(["run", "--manifest", str(m), "--providers", "deepgram",
                   "--results-dir", str(tmp_path / "r")])
    assert rc == 2
    assert "--yes-spend" in capsys.readouterr().err


def test_cli_run_without_yes_spend_ok_when_all_cached(tmp_path, capsys, monkeypatch, no_network):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    m = write_manifest(tmp_path, [row(1)])
    res = tmp_path / "r"
    (res / "deepgram").mkdir(parents=True)
    (res / "deepgram" / "s1.json").write_text(json.dumps(
        {"sample_id": "s1", "provider": "deepgram", "text": "एक दो तीन", "elapsed_s": 0.1}))
    rc = cli.main(["run", "--manifest", str(m), "--providers", "deepgram",
                   "--results-dir", str(res), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["wer"] == 0.0


def test_cli_dry_run_reports_missing_keys(tmp_path, capsys, monkeypatch, no_network):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    m = write_manifest(tmp_path, [row(1)])
    rc = cli.main(["run", "--manifest", str(m), "--providers", "elevenlabs", "--dry-run",
                   "--results-dir", str(tmp_path / "r")])
    assert rc == 3
    assert "ELEVENLABS_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    assert cli.main(["run", "--manifest", str(m), "--providers", "elevenlabs", "--dry-run",
                     "--results-dir", str(tmp_path / "r")]) == 0


def test_cli_rejects_unknown_provider(tmp_path):
    m = write_manifest(tmp_path, [row(1)])
    with pytest.raises(SystemExit):
        cli.main(["estimate", "--manifest", str(m), "--providers", "nope"])
