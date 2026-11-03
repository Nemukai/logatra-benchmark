"""Offline tests for provider adapters: request shape + response parsing.

Every HTTP call goes through httpx.MockTransport; no network, no real keys
(the env vars below are dummy strings).
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import httpx
import pytest

from logatra_benchmark.adapters import PROVIDERS, Unsupported, get_adapter
from logatra_benchmark.adapters.base import missing_env


def make_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


class Recorder:
    """MockTransport handler that records requests and replays canned responses."""

    def __init__(self, routes):
        self.routes = routes  # list of (predicate(request) -> bool, response factory)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        for pred, resp in self.routes:
            if pred(request):
                return resp(request) if callable(resp) else resp
        return httpx.Response(404, text=f"no route for {request.method} {request.url}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def form_fields(request: httpx.Request) -> dict[str, list[str]]:
    """Parse a multipart body into {name: [values]} (files show as '<file>')."""
    ctype = request.headers["content-type"]
    boundary = ctype.split("boundary=")[1].encode()
    out: dict[str, list[str]] = {}
    for part in request.content.split(b"--" + boundary):
        if b"Content-Disposition" not in part:
            continue
        head, _, body = part.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        val = "<file>" if b"filename=" in head else body.rstrip(b"\r\n").decode()
        out.setdefault(name, []).append(val)
    return out


@pytest.fixture
def wav(tmp_path):
    return make_wav(tmp_path / "s1.wav", 1.0)


# --- registry ---------------------------------------------------------------

def test_registry_has_new_entrants():
    for name in ("deepgram", "google", "elevenlabs", "assemblyai", "azure",
                 "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-transcribe",
                 "indicconformer"):
        assert name in PROVIDERS
        assert get_adapter(name).name == name


def test_missing_env_reports_required_vars(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert missing_env(get_adapter("deepgram")) == ["DEEPGRAM_API_KEY"]
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    assert missing_env(get_adapter("deepgram")) == []
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setenv("SARVAM_API_SUBSCRIPTION_KEY", "dummy")
    assert missing_env(get_adapter("sarvam-saaras-v4")) == []


# --- Deepgram -----------------------------------------------------------------

DG_RESPONSE = {
    "metadata": {"duration": 1.0},
    "results": {"channels": [{"alternatives": [{
        "transcript": "नमस्ते दुनिया",
        "words": [
            {"word": "नमस्ते", "start": 0.0, "end": 0.4, "speaker": 0},
            {"word": "दुनिया", "start": 0.5, "end": 0.9, "speaker": 1},
        ],
    }]}]},
}


def test_deepgram_monolingual_request_and_parse(monkeypatch, wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-dg")
    rec = Recorder([(lambda r: r.url.path == "/v1/listen", httpx.Response(200, json=DG_RESPONSE))])
    ad = get_adapter("deepgram", client=rec.client())
    out = ad.transcribe(wav, language="ta")
    req = rec.requests[0]
    assert req.method == "POST"
    assert req.url.host == "api.deepgram.com"
    assert req.headers["authorization"] == "Token dummy-dg"
    assert req.headers["content-type"] == "audio/wav"
    q = dict(req.url.params.multi_items())
    assert q["model"] == "nova-3" and q["language"] == "ta" and q["smart_format"] == "false"
    assert "language_hint" not in q
    assert req.content == wav.read_bytes()
    assert out.text == "नमस्ते दुनिया"
    assert [w.speaker for w in out.words] == ["spk0", "spk1"]
    assert len(out.speaker_turns) == 2
    assert out.raw["_logatra_model"] == "nova-3"


def test_deepgram_codemix_uses_multi_with_hints(monkeypatch, wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    rec = Recorder([(lambda r: True, httpx.Response(200, json=DG_RESPONSE))])
    ad = get_adapter("deepgram", client=rec.client())
    ad.transcribe(wav, language="hi", code_mixed=True)
    items = rec.requests[0].url.params.multi_items()
    assert ("language", "multi") in items
    assert [v for k, v in items if k == "language_hint"] == ["hi", "en"]


def test_deepgram_unsupported_language_raises(monkeypatch, wav):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    ad = get_adapter("deepgram", client=Recorder([]).client())
    assert ad.supports("ml") is not None
    with pytest.raises(Unsupported):
        ad.transcribe(wav, language="or")


def test_deepgram_http_error_is_retryable_text(monkeypatch, wav):
    from logatra_benchmark.adapters import TranscriptionError

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
    rec = Recorder([(lambda r: True, httpx.Response(429, text="slow down"))])
    with pytest.raises(TranscriptionError, match="HTTP 429"):
        get_adapter("deepgram", client=rec.client()).transcribe(wav, language="hi")


# --- ElevenLabs ---------------------------------------------------------------

EL_RESPONSE = {
    "language_code": "hin", "language_probability": 0.98, "text": "नमस्ते दुनिया",
    "words": [
        {"text": "नमस्ते", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
        {"text": "दुनिया", "start": 0.5, "end": 0.9, "type": "word", "speaker_id": "speaker_0"},
    ],
}


def test_elevenlabs_request_and_parse(monkeypatch, wav):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy-el")
    rec = Recorder([(lambda r: r.url.path == "/v1/speech-to-text", httpx.Response(200, json=EL_RESPONSE))])
    out = get_adapter("elevenlabs", client=rec.client()).transcribe(wav, language="hi")
    req = rec.requests[0]
    assert req.headers["xi-api-key"] == "dummy-el"
    f = form_fields(req)
    assert f["model_id"] == ["scribe_v2"]
    assert f["language_code"] == ["hin"]
    assert f["file"] == ["<file>"]
    assert f["timestamps_granularity"] == ["word"]
    assert out.text == "नमस्ते दुनिया"
    assert [w.text for w in out.words] == ["नमस्ते", "दुनिया"]  # spacing dropped
    assert out.speaker_turns[0].end == 0.9


def test_elevenlabs_codemix_omits_language(monkeypatch, wav):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy")
    rec = Recorder([(lambda r: True, httpx.Response(200, json=EL_RESPONSE))])
    get_adapter("elevenlabs", client=rec.client()).transcribe(wav, language="hi-en")
    assert "language_code" not in form_fields(rec.requests[0])


# --- AssemblyAI ---------------------------------------------------------------

def test_assemblyai_upload_submit_poll(monkeypatch, wav):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "dummy-aai")
    polls = iter([
        {"id": "t1", "status": "processing"},
        {"id": "t1", "status": "completed", "text": "বাংলা কথা",
         "speech_model_used": "universal-2",
         "words": [{"text": "বাংলা", "start": 0, "end": 400}, {"text": "কথা", "start": 500, "end": 900}]},
    ])
    rec = Recorder([
        (lambda r: r.url.path == "/v2/upload", httpx.Response(200, json={"upload_url": "https://cdn/x"})),
        (lambda r: r.url.path == "/v2/transcript" and r.method == "POST",
         httpx.Response(200, json={"id": "t1", "status": "queued"})),
        (lambda r: r.url.path == "/v2/transcript/t1", lambda r: httpx.Response(200, json=next(polls))),
    ])
    ad = get_adapter("assemblyai", client=rec.client(), sleep=lambda s: None)
    out = ad.transcribe(wav, language="bn")
    up, submit = rec.requests[0], rec.requests[1]
    assert up.headers["authorization"] == "dummy-aai"
    assert up.content == wav.read_bytes()
    body = json.loads(submit.content)
    assert body["audio_url"] == "https://cdn/x"
    assert body["language_code"] == "bn"
    assert body["speech_models"] == ["universal-2"]
    assert "language_detection" not in body
    assert len(rec.requests) == 4  # upload, submit, 2 polls
    assert out.text == "বাংলা কথা"
    assert out.words[1].start == 0.5


def test_assemblyai_hindi_routes_to_universal_35(monkeypatch, wav):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "dummy")
    ad = get_adapter("assemblyai")
    body = ad.build_request("u", "hi", code_mixed=True)
    assert body["speech_models"] == ["universal-3-5-pro", "universal-2"]
    assert body["language_code"] == "hi"
    monkeypatch.setenv("LOGATRA_ASSEMBLYAI_DETECT", "1")
    body = ad.build_request("u", "hi")
    assert body["language_detection"] is True and "language_code" not in body
    with pytest.raises(Unsupported):
        ad.build_request("u", "or")


def test_assemblyai_error_status(monkeypatch, wav):
    from logatra_benchmark.adapters import TranscriptionError

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "dummy")
    rec = Recorder([
        (lambda r: r.url.path == "/v2/upload", httpx.Response(200, json={"upload_url": "u"})),
        (lambda r: r.method == "POST", httpx.Response(200, json={"id": "t", "status": "error",
                                                                "error": "bad audio"})),
    ])
    with pytest.raises(TranscriptionError, match="bad audio"):
        get_adapter("assemblyai", client=rec.client(), sleep=lambda s: None).transcribe(wav, "hi")


# --- OpenAI -------------------------------------------------------------------

def test_whisper_verbose_json_and_hint(monkeypatch, wav):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-oa")
    monkeypatch.delenv("LOGATRA_OPENAI_STT_MODEL", raising=False)
    rec = Recorder([(lambda r: True, httpx.Response(200, json={
        "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]}))])
    out = get_adapter("whisper", client=rec.client()).transcribe(wav, language="en-IN")
    f = form_fields(rec.requests[0])
    assert f["model"] == ["whisper-1"]
    assert f["response_format"] == ["verbose_json"]
    assert f["language"] == ["en"]
    assert rec.requests[0].headers["authorization"] == "Bearer dummy-oa"
    assert out.words[0].end == 0.5 and out.raw["_logatra_model"] == "whisper-1"


@pytest.mark.parametrize("name", ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"])
def test_gpt4o_transcribe_uses_json(monkeypatch, wav, name):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("LOGATRA_OPENAI_STT_MODEL", "should-not-apply")
    rec = Recorder([(lambda r: True, httpx.Response(200, json={
        "text": "தமிழ்", "usage": {"type": "tokens", "input_tokens": 50}}))])
    out = get_adapter(name, client=rec.client()).transcribe(wav, language="ta")
    f = form_fields(rec.requests[0])
    assert f["model"] == [name] and f["response_format"] == ["json"] and f["language"] == ["ta"]
    assert out.text == "தமிழ்"


def test_gpt_transcribe_languages_list(monkeypatch, wav):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    rec = Recorder([(lambda r: True, httpx.Response(200, json={
        "text": "kya scene hai", "languages": [{"code": "hi"}]}))])
    get_adapter("gpt-transcribe", client=rec.client()).transcribe(wav, language="hi-en")
    f = form_fields(rec.requests[0])
    assert f["languages[]"] == ["hi", "en"] and "language" not in f


def test_openai_unsupported_language_falls_back_unhinted(monkeypatch, wav):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("LOGATRA_OPENAI_STT_MODEL", raising=False)
    calls = iter([
        httpx.Response(400, json={"error": {"code": "unsupported_language"}}),
        httpx.Response(200, json={"text": "অসমীয়া"}),
    ])
    rec = Recorder([(lambda r: True, lambda r: next(calls))])
    out = get_adapter("whisper", client=rec.client()).transcribe(wav, language="as")
    assert "language" in form_fields(rec.requests[0])
    assert "language" not in form_fields(rec.requests[1])
    assert out.raw["_logatra_hint_dropped"] is True


# --- Sarvam -------------------------------------------------------------------

def test_sarvam_rest_short_clip(monkeypatch, wav):
    monkeypatch.setenv("SARVAM_API_KEY", "dummy-sv")
    monkeypatch.delenv("LOGATRA_SARVAM_MODEL", raising=False)
    rec = Recorder([(lambda r: r.url.path == "/speech-to-text",
                     httpx.Response(200, json={"transcript": "ଓଡ଼ିଆ", "language_code": "od-IN"}))])
    out = get_adapter("sarvam-saaras-v4", client=rec.client()).transcribe(wav, language="or")
    req = rec.requests[0]
    assert req.headers["api-subscription-key"] == "dummy-sv"
    f = form_fields(req)
    assert f["model"] == ["saaras:v4"] and f["mode"] == ["transcribe"]
    assert f["language_code"] == ["od-IN"]
    assert out.text == "ଓଡ଼ିଆ"


def test_sarvam_hinglish_stays_unknown(monkeypatch, wav):
    monkeypatch.setenv("SARVAM_API_KEY", "dummy")
    rec = Recorder([(lambda r: True, httpx.Response(200, json={"transcript": "x"}))])
    get_adapter("sarvam-saaras", client=rec.client()).transcribe(wav, language="hi-en")
    assert form_fields(rec.requests[0])["language_code"] == ["unknown"]


def test_sarvam_batch_path_for_long_clip(monkeypatch, tmp_path):
    monkeypatch.setenv("SARVAM_API_KEY", "dummy-sv")
    monkeypatch.delenv("LOGATRA_SARVAM_MODEL", raising=False)
    long_wav = make_wav(tmp_path / "long.wav", 31.0)
    states = iter(["Running", "Completed"])

    def status(_):
        st = next(states)
        body = {"job_id": "job_1", "job_state": st}
        if st == "Completed":
            body["job_details"] = [{"inputs": [{"file_name": "long.wav"}],
                                    "outputs": [{"file_name": "0.json"}], "state": "Success"}]
        return httpx.Response(200, json=body)

    rec = Recorder([
        (lambda r: r.url.path == "/speech-to-text/job/v1" and r.method == "POST",
         httpx.Response(200, json={"job_id": "job_1", "job_state": "Accepted"})),
        (lambda r: r.url.path.endswith("/upload-files"),
         httpx.Response(200, json={"job_id": "job_1", "upload_urls": {
             "long.wav": {"file_url": "https://blob.example/in/long.wav?sas=1"}}})),
        (lambda r: r.method == "PUT" and r.url.host == "blob.example", httpx.Response(201)),
        (lambda r: r.url.path.endswith("/start"), httpx.Response(200, json={"job_state": "Running"})),
        (lambda r: r.url.path.endswith("/status"), status),
        (lambda r: r.url.path.endswith("/download-files"),
         httpx.Response(200, json={"download_urls": {
             "0.json": {"file_url": "https://blob.example/out/0.json?sas=2"}}})),
        (lambda r: r.method == "GET" and r.url.path == "/out/0.json",
         httpx.Response(200, json={"transcript": "लंबी क्लिप", "language_code": "hi-IN"})),
    ])
    ad = get_adapter("sarvam-saaras-v4", client=rec.client(), sleep=lambda s: None)
    out = ad.transcribe(long_wav, language="hi")
    paths = [(r.method, r.url.path) for r in rec.requests]
    assert ("POST", "/speech-to-text") not in paths  # REST endpoint never hit
    init = json.loads(rec.requests[0].content)
    assert init["job_parameters"]["model"] == "saaras:v4"
    assert init["job_parameters"]["language_code"] == "hi-IN"
    assert init["job_parameters"]["mode"] == "transcribe"
    put = next(r for r in rec.requests if r.method == "PUT")
    assert put.headers["x-ms-blob-type"] == "BlockBlob"
    assert json.loads([r for r in rec.requests if r.url.path.endswith("/download-files")][0].content) \
        == {"job_id": "job_1", "files": ["0.json"]}
    assert out.text == "लंबी क्लिप"
    assert out.raw["_logatra_batch"]["job_id"] == "job_1"


def test_saarika_long_clip_is_unsupported(monkeypatch, tmp_path):
    monkeypatch.setenv("SARVAM_API_KEY", "dummy")
    monkeypatch.delenv("LOGATRA_SARVAM_MODEL", raising=False)
    long_wav = make_wav(tmp_path / "long.wav", 45.0)
    rec = Recorder([])
    with pytest.raises(Unsupported, match="Batch"):
        get_adapter("sarvam", client=rec.client()).transcribe(long_wav, language="hi")
    assert rec.requests == []


# --- Google -------------------------------------------------------------------

def test_google_recognize_request_and_parse(monkeypatch, wav):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    monkeypatch.delenv("LOGATRA_GOOGLE_LOCATION", raising=False)
    rec = Recorder([(lambda r: r.url.path.endswith(":recognize"), httpx.Response(200, json={
        "results": [{"alternatives": [{"transcript": "ਪੰਜਾਬੀ"}], "languageCode": "pa-guru-in"},
                    {"alternatives": [{"transcript": "ਗੱਲ"}]}]}))])
    ad = get_adapter("google", client=rec.client(), token_provider=lambda: "tok")
    out = ad.transcribe(wav, language="pa")
    req = rec.requests[0]
    assert req.url.host == "us-speech.googleapis.com"
    assert req.url.path == "/v2/projects/proj-1/locations/us/recognizers/_:recognize"
    assert req.headers["authorization"] == "Bearer tok"
    body = json.loads(req.content)
    assert body["config"]["model"] == "chirp_3"
    assert body["config"]["languageCodes"] == ["pa-Guru-IN"]
    assert body["content"]
    assert out.text == "ਪੰਜਾਬੀ ਗੱਲ"


def test_google_long_clip_without_bucket_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.delenv("LOGATRA_GOOGLE_GCS_BUCKET", raising=False)
    long_wav = make_wav(tmp_path / "l.wav", 61.0)
    ad = get_adapter("google", client=Recorder([]).client(), token_provider=lambda: "t")
    with pytest.raises(Unsupported, match="GCS"):
        ad.transcribe(long_wav, language="hi")
    with pytest.raises(Unsupported):
        ad.transcribe(long_wav, language="ur")  # Urdu not on chirp_3


def test_google_batch_with_bucket(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("LOGATRA_GOOGLE_GCS_BUCKET", "bkt")
    long_wav = make_wav(tmp_path / "l.wav", 61.0)
    uri_holder = {}

    def batch(req):
        uri_holder["uri"] = json.loads(req.content)["files"][0]["uri"]
        return httpx.Response(200, json={"name": "projects/p/locations/us/operations/op1"})

    def poll(req):
        return httpx.Response(200, json={"name": "op1", "done": True, "response": {"results": {
            uri_holder["uri"]: {"inlineResult": {"transcript": {"results": [
                {"alternatives": [{"transcript": "लंबा"}]}]}}}}}})

    rec = Recorder([
        (lambda r: r.url.host == "storage.googleapis.com" and r.method == "POST", httpx.Response(200, json={})),
        (lambda r: r.url.path.endswith(":batchRecognize"), batch),
        (lambda r: "/operations/" in r.url.path, poll),
        (lambda r: r.method == "DELETE", httpx.Response(204)),
    ])
    ad = get_adapter("google", client=rec.client(), token_provider=lambda: "t", sleep=lambda s: None)
    out = ad.transcribe(long_wav, language="hi")
    assert uri_holder["uri"].startswith("gs://bkt/logatra-tmp/")
    assert out.text == "लंबा"
    assert rec.requests[-1].method == "DELETE"  # temp object cleaned up


# --- Azure --------------------------------------------------------------------

def test_azure_fast_transcription(monkeypatch, wav):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "dummy-az")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "centralindia")
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    rec = Recorder([(lambda r: True, httpx.Response(200, json={
        "durationMilliseconds": 1000,
        "combinedPhrases": [{"text": "اردو بات"}],
        "phrases": [{"text": "اردو بات", "words": [
            {"text": "اردو", "offsetMilliseconds": 0, "durationMilliseconds": 400}]}]}))])
    out = get_adapter("azure", client=rec.client()).transcribe(wav, language="ur")
    req = rec.requests[0]
    assert req.url.host == "centralindia.api.cognitive.microsoft.com"
    assert req.url.path == "/speechtotext/transcriptions:transcribe"
    assert req.url.params["api-version"] == "2025-10-15"
    assert req.headers["ocp-apim-subscription-key"] == "dummy-az"
    f = form_fields(req)
    assert json.loads(f["definition"][0])["locales"] == ["ur-IN"]
    assert f["audio"] == ["<file>"]
    assert out.text == "اردو بات" and out.words[0].end == 0.4


def test_azure_hinglish_multi_locale():
    d = get_adapter("azure").definition("hi", code_mixed=True)
    assert d["locales"] == ["hi-IN", "en-IN"]


# --- OpenRouter (existing adapter, now client-injectable) -------------------------

def test_openrouter_request(monkeypatch, wav):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    rec = Recorder([(lambda r: True, httpx.Response(200, json={
        "choices": [{"message": {"content": " hi "}}], "usage": {"cost": 0.0012}}))])
    out = get_adapter("gemini", client=rec.client()).transcribe(wav, language="hi")
    body = json.loads(rec.requests[0].content)
    assert body["messages"][0]["content"][1]["input_audio"]["format"] == "wav"
    assert out.text == "hi" and out.raw["usage"]["cost"] == 0.0012


# --- IndicConformer (skeleton; never loads a model in tests) --------------------

def test_indicconformer_language_support():
    ad = get_adapter("indicconformer")
    assert ad.supports("brx") is None
    assert ad.supports("en-IN") is not None
    assert ad.required_env == ()
