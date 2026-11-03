"""Tests for report scoring: clustering, bootstrap CIs, private-split guards.

Pure python, no network, no results cache needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from logatra_benchmark.manifest import (
    PRIVATE_REPORT_DIR,
    VALID_SPLITS,
    Sample,
    is_private_path,
    require_private_output,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score_report = _load("score_report")


def _row(sid, prov, wer, words=10, split="public", sa=None, cluster=None):
    return {"sample_id": sid, "provider": prov, "wer": wer,
            "wer_sa": wer if sa is None else sa, "ref_words": words, "split": split,
            "cluster": cluster or score_report.cluster_of(sid, split)}


class TestCluster:
    def test_heldout_clusters_by_video(self):
        assert score_report.cluster_of("heldout_yt_Arw8-1YR2Ys_000", "heldout") == "Arw8-1YR2Ys"
        assert score_report.cluster_of("heldout_yt_a_b_c_017", "heldout") == "a_b_c"

    def test_public_clusters_by_sample(self):
        assert score_report.cluster_of("fleurs_hi_in_test_0001", "public") == "fleurs_hi_in_test_0001"


class TestBootstrap:
    def test_ci_brackets_point_estimate(self):
        rows = []
        for i in range(60):
            rows.append(_row(f"s{i}", "a", 0.2 + 0.1 * (i % 3)))
            rows.append(_row(f"s{i}", "b", 0.5 + 0.1 * (i % 2)))
        bs = score_report.bootstrap(rows, ["a", "b"], b=400, pairwise=True)
        lo, hi = bs["providers"]["a"]["wer_ci"]
        assert lo <= 0.3 <= hi and hi - lo < 0.1
        glo, ghi = bs["pairwise"]["b|a"]["wer_ci"]
        assert glo > 0  # b clearly worse than a: not a tie

    def test_identical_systems_tie(self):
        rows = [_row(f"s{i}", p, (i % 5) / 10) for i in range(40) for p in ("a", "b")]
        bs = score_report.bootstrap(rows, ["a", "b"], b=200, pairwise=True)
        assert bs["pairwise"]["a|b"]["wer_ci"] == [0.0, 0.0]

    def test_cluster_bootstrap_is_wider_than_sample_bootstrap(self):
        # Two videos with very different difficulty: resampling videos must
        # show more uncertainty than pretending the 40 chunks are independent.
        rows = []
        for v, base in (("vidA", 0.1), ("vidB", 0.6)):
            for i in range(20):
                sid = f"heldout_yt_{v}_{i:03d}"
                rows.append(_row(sid, "a", base, split="heldout"))
        clustered = score_report.bootstrap(rows, ["a"], b=500)
        flat = score_report.bootstrap([dict(r, cluster=r["sample_id"]) for r in rows], ["a"], b=500)
        w_c = clustered["providers"]["a"]["wer_ci"][1] - clustered["providers"]["a"]["wer_ci"][0]
        w_f = flat["providers"]["a"]["wer_ci"][1] - flat["providers"]["a"]["wer_ci"][0]
        assert w_c > w_f
        assert clustered["providers"]["a"]["n_clusters"] == 2

    def test_deterministic(self):
        rows = [_row(f"s{i}", "a", (i % 7) / 10) for i in range(30)]
        assert score_report.bootstrap(rows, ["a"], b=100) == score_report.bootstrap(rows, ["a"], b=100)


class TestPrivateSplit:
    def test_private_is_a_valid_split(self):
        assert "private" in VALID_SPLITS
        Sample(id="x", audio_path="a.wav", reference_text="r", language="hi", split="private")

    def test_private_output_guard(self):
        assert is_private_path(PRIVATE_REPORT_DIR / "data.json")
        assert not is_private_path(ROOT / "report" / "data.json")
        with pytest.raises(SystemExit):
            require_private_output(ROOT / "report" / "data.json")
        require_private_output(PRIVATE_REPORT_DIR / "data.json")

    def test_score_report_refuses_private_to_report_dir(self, monkeypatch, tmp_path):
        m = tmp_path / "m.jsonl"
        m.write_text("")
        monkeypatch.setattr(sys, "argv", ["score_report", "--include-private",
                                          "--out", str(ROOT / "report" / "x.json"), str(m)])
        with pytest.raises(SystemExit):
            score_report.main()

    def test_score_report_drops_private_rows_by_default(self, monkeypatch, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\0" * 64000)
        res = tmp_path / "results"
        for sid, split in (("pub_1", "public"), ("priv_1", "private")):
            (res / "whisper").mkdir(parents=True, exist_ok=True)
            (res / "whisper" / f"{sid}.json").write_text(json.dumps({"text": "secret words"}))
        m = tmp_path / "m.jsonl"
        m.write_text("\n".join(json.dumps({
            "id": sid, "audio_path": "a.wav", "reference_text": "secret words",
            "language": "hi", "split": split}) for sid, split in
            (("pub_1", "public"), ("priv_1", "private"))) + "\n")
        out = tmp_path / "out" / "data.json"
        monkeypatch.setattr(sys, "argv", ["score_report", "--results-dir", str(res),
                                          "--bootstrap", "50", "--out", str(out), str(m)])
        assert score_report.main() == 0
        rows = json.loads((out.parent / "rows.json").read_text())
        assert [r["sample_id"] for r in rows] == ["pub_1"]
        assert "private" not in json.loads(out.read_text())["comparable_counts"]

    def test_build_report_detects_private(self):
        build = _load("build_report")
        assert build.has_private({"by_provider": {"whisper|private": {}}})
        assert not build.has_private({"by_provider": {"whisper|public": {}}})
