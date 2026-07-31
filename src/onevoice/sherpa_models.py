from __future__ import annotations

from dataclasses import dataclass


_ASR_RELEASE_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
)
_VI_HF_BASE = (
    "https://huggingface.co/hynt/"
    "Zipformer-30M-RNNT-Streaming-6000h/resolve/main"
)


@dataclass(frozen=True, slots=True)
class SherpaStreamingModelSpec:
    id: str
    language: str
    directory: str
    encoder: str
    decoder: str
    joiner: str
    tokens: str = "tokens.txt"
    modeling_unit: str = "bpe"
    bpe_model: str | None = "bpe.model"
    bpe_model_url: str | None = None
    bpe_vocab: str | None = "bpe.vocab"
    hotword_case: str = "preserve"
    archive: str | None = None
    remote_files: tuple[tuple[str, str], ...] = ()
    chunk_size: int | None = None
    left_context: int | None = None
    license: str = "upstream-specific"
    source_url: str = ""

    @property
    def archive_url(self) -> str | None:
        return (
            f"{_ASR_RELEASE_BASE}/{self.archive}"
            if self.archive
            else None
        )


VI_STREAMING_ZIPFORMER = SherpaStreamingModelSpec(
    id="hynt-zipformer-vi-30m-streaming-6000h-chunk-32",
    language="vi",
    directory="hynt-zipformer-vi-30m-streaming-6000h-chunk-32",
    encoder="encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
    decoder="decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
    joiner="joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
    remote_files=(
        (
            "encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
            f"{_VI_HF_BASE}/encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
        ),
        (
            "decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
            f"{_VI_HF_BASE}/decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
        ),
        (
            "joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
            f"{_VI_HF_BASE}/joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx",
        ),
        ("tokens.txt", f"{_VI_HF_BASE}/config.json"),
    ),
    bpe_model_url=f"{_VI_HF_BASE}/bpe.model",
    hotword_case="upper",
    chunk_size=32,
    left_context=128,
    license="cc-by-nc-nd-4.0",
    source_url=(
        "https://huggingface.co/hynt/"
        "Zipformer-30M-RNNT-Streaming-6000h"
    ),
)

EN_STREAMING_ZIPFORMER = SherpaStreamingModelSpec(
    id="sherpa-onnx-streaming-zipformer-en-2023-06-26",
    language="en",
    directory="sherpa-onnx-streaming-zipformer-en-2023-06-26",
    archive="sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    encoder="encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder="decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    joiner="joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    hotword_case="upper",
    chunk_size=16,
    left_context=128,
    source_url=(
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/"
        "online-transducer/zipformer-transducer-models.html"
    ),
)

ZH_STREAMING_ZIPFORMER = SherpaStreamingModelSpec(
    id="sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
    language="zh",
    directory="sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
    archive="sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2",
    encoder="encoder.int8.onnx",
    decoder="decoder.onnx",
    joiner="joiner.int8.onnx",
    modeling_unit="cjkchar",
    bpe_model=None,
    bpe_vocab=None,
    source_url=(
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/"
        "online-transducer/zipformer-transducer-models.html"
    ),
)

KO_STREAMING_ZIPFORMER = SherpaStreamingModelSpec(
    id="sherpa-onnx-streaming-zipformer-korean-2024-06-16",
    language="ko",
    directory="sherpa-onnx-streaming-zipformer-korean-2024-06-16",
    archive="sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2",
    encoder="encoder-epoch-99-avg-1.int8.onnx",
    decoder="decoder-epoch-99-avg-1.onnx",
    joiner="joiner-epoch-99-avg-1.int8.onnx",
    source_url=(
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/"
        "online-transducer/zipformer-transducer-models.html"
    ),
)


SHERPA_STREAMING_MODELS: dict[str, SherpaStreamingModelSpec] = {
    spec.id: spec
    for spec in (
        VI_STREAMING_ZIPFORMER,
        EN_STREAMING_ZIPFORMER,
        ZH_STREAMING_ZIPFORMER,
        KO_STREAMING_ZIPFORMER,
    )
}

DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE = {
    spec.language: spec.id for spec in SHERPA_STREAMING_MODELS.values()
}


@dataclass(frozen=True, slots=True)
class SherpaPunctuationModelSpec:
    id: str
    language: str
    directory: str
    archive: str
    model: str
    bpe_vocab: str

    @property
    def archive_url(self) -> str:
        return (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            f"punctuation-models/{self.archive}"
        )


EN_ONLINE_PUNCTUATION = SherpaPunctuationModelSpec(
    id="sherpa-onnx-online-punct-en-2024-08-06",
    language="en",
    directory="sherpa-onnx-online-punct-en-2024-08-06",
    archive="sherpa-onnx-online-punct-en-2024-08-06.tar.bz2",
    model="model.int8.onnx",
    bpe_vocab="bpe.vocab",
)
