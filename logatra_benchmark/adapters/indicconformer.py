"""AI4Bharat IndicConformer adapter (open weights, local inference). SKELETON.

Model: ai4bharat/indic-conformer-600m-multilingual (Hugging Face, MIT, gated:
accept the terms on the model page and `huggingface-cli login` first).
22 scheduled Indian languages (as, bn, brx, doi, gu, hi, kn, kok, ks, mai, ml,
mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur); no English.

Not run by the benchmark yet. Usage per the model card (verify when enabling;
the card is gated and was not re-read on 2026-11-03):

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
    wav, sr = torchaudio.load(path)            # resample to 16 kHz mono
    text = model(wav, "hi", "ctc")             # or "rnnt"

Install the optional extra: `uv sync --extra indicconformer`
(torch, torchaudio, transformers, onnxruntime). Cost is local compute only:
pricing.py lists it at $0/min.

Env:
    LOGATRA_INDICCONFORMER_MODEL   default ai4bharat/indic-conformer-600m-multilingual
    LOGATRA_INDICCONFORMER_DECODER ctc (default) | rnnt
    LOGATRA_INDICCONFORMER_DEVICE  cpu (default) | cuda
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from . import languages
from .base import NotConfigured, Transcription, TranscriptionError

DEFAULT_MODEL = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerAdapter:
    name = "indicconformer"
    required_env: tuple[str, ...] = ()
    wants_code_mixed = True
    local = True  # no network, no billing

    _lock = threading.Lock()
    _model = None

    @property
    def model(self) -> str:
        return os.environ.get("LOGATRA_INDICCONFORMER_MODEL", "").strip() or DEFAULT_MODEL

    @property
    def decoder(self) -> str:
        return os.environ.get("LOGATRA_INDICCONFORMER_DECODER", "").strip() or "ctc"

    def supports(self, language: str | None, code_mixed: bool = False) -> str | None:
        return languages.unsupported_reason("indicconformer", language, code_mixed)

    def _load(self):
        with self._lock:
            if IndicConformerAdapter._model is None:
                try:
                    import torch  # noqa: F401
                    from transformers import AutoModel
                except ImportError as exc:
                    raise NotConfigured(
                        "IndicConformer needs local deps: `uv sync --extra indicconformer`"
                    ) from exc
                device = os.environ.get("LOGATRA_INDICCONFORMER_DEVICE", "cpu")
                m = AutoModel.from_pretrained(self.model, trust_remote_code=True)
                IndicConformerAdapter._model = m.to(device) if hasattr(m, "to") else m
            return IndicConformerAdapter._model

    def transcribe(self, audio_path: Path, language: str | None = None,
                   code_mixed: bool = False) -> Transcription:
        lang = languages.resolve("indicconformer", language, code_mixed)
        if not lang:
            raise TranscriptionError("IndicConformer needs an explicit language")
        model = self._load()
        try:
            import torchaudio
        except ImportError as exc:
            raise NotConfigured("IndicConformer needs torchaudio") from exc
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        text = model(wav, lang, self.decoder)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return Transcription(text=str(text).strip(),
                             raw={"_logatra_model": self.model, "decoder": self.decoder,
                                  "language": lang, "usage": {"cost": 0.0}})
