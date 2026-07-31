from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from onevoice.backends.asr import asr_model_options, validate_asr_selection
from onevoice.backends.asr_sherpa import SherpaOnnxZipformerAsrBackend
from onevoice.config import AsrConfig
from onevoice.models import AsrWordTiming, SpeechSegment
from onevoice.registry import registry
from onevoice.sherpa_models import (
    DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE,
    SHERPA_STREAMING_MODELS,
)


def _segment(sample_count: int, *, final: bool = False) -> SpeechSegment:
    return SpeechSegment(
        samples=np.arange(sample_count, dtype=np.float32),
        sample_rate=16_000,
        started_at=0.0,
        ended_at=sample_count / 16_000,
        is_final=final,
    )


def _config(language: str = "vi") -> AsrConfig:
    return AsrConfig(
        backend="sherpa_onnx",
        model=DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE[language],
        language=language,
    )


def _assets(root: Path, config: AsrConfig) -> None:
    backend = SherpaOnnxZipformerAsrBackend(config)
    spec = backend.model_spec
    for name in (spec.encoder, spec.decoder, spec.joiner, spec.tokens):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")


class _FakeStream:
    def __init__(self) -> None:
        self.accepted: list[tuple[int, np.ndarray]] = []
        self.ready = 0
        self.finished = False

    def accept_waveform(
        self, sample_rate: int, samples: np.ndarray
    ) -> None:
        self.accepted.append((sample_rate, samples.copy()))
        self.ready += 1

    def input_finished(self) -> None:
        self.finished = True
        self.ready += 1


class _FakeRecognizer:
    def __init__(
        self, texts: tuple[str, ...] = ("xin", "xin chào", "xin chào")
    ) -> None:
        self.texts = texts
        self.streams: list[_FakeStream] = []
        self.hotword_inputs: list[str | None] = []
        self.decode_count = 0

    def create_stream(
        self, hotwords: str | None = None
    ) -> _FakeStream:
        self.hotword_inputs.append(hotwords)
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    @staticmethod
    def is_ready(stream: _FakeStream) -> bool:
        return stream.ready > 0

    def decode_stream(self, stream: _FakeStream) -> None:
        stream.ready -= 1
        self.decode_count += 1

    def get_result(self, stream: _FakeStream) -> str:
        del stream
        index = min(max(self.decode_count - 1, 0), len(self.texts) - 1)
        return self.texts[index]


class _FakeTimedRecognizer(_FakeRecognizer):
    @staticmethod
    def tokens(stream: _FakeStream) -> list[str]:
        del stream
        return [" HOW", " A", "RE", " YOU", " I", " AM", " FINE"]

    @staticmethod
    def timestamps(stream: _FakeStream) -> list[float]:
        del stream
        return [0.1, 0.3, 0.35, 0.6, 0.9, 1.1, 1.3]


@pytest.mark.parametrize("language", ("vi", "en", "zh", "ko"))
def test_streaming_model_catalog_covers_onevoice_languages(
    language: str,
) -> None:
    options = asr_model_options("sherpa_onnx", language)
    assert options == (
        DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE[language],
    )
    assert SHERPA_STREAMING_MODELS[options[0]].language == language


def test_sherpa_registry_and_language_model_validation() -> None:
    assert "sherpa_onnx" in registry.names("asr")
    assert asr_model_options("sherpa_onnx", "auto") == ("auto",)
    en_model = DEFAULT_SHERPA_STREAMING_MODEL_BY_LANGUAGE["en"]
    with pytest.raises(ValueError, match="supports 'en'"):
        validate_asr_selection("sherpa_onnx", en_model, "vi")
    with pytest.raises(ValueError, match="requires vi, en, zh, or ko"):
        validate_asr_selection("sherpa_onnx", "auto", "auto")


def test_sherpa_loads_online_transducer_without_hotwords(
    tmp_path: Path,
) -> None:
    config = _config("vi")
    config.model_dir = str(tmp_path)
    _assets(tmp_path, config)
    backend = SherpaOnnxZipformerAsrBackend(config)
    captured: dict[str, object] = {}
    recognizer = _FakeRecognizer()

    class _Factory:
        @staticmethod
        def from_transducer(**kwargs):
            captured.update(kwargs)
            return recognizer

    backend._sherpa = SimpleNamespace(OnlineRecognizer=_Factory)
    backend.load()

    assert backend._recognizer is recognizer
    assert captured["encoder"] == str(
        (tmp_path / backend.model_spec.encoder).resolve()
    )
    assert captured["enable_endpoint_detection"] is False
    assert captured["decoding_method"] == "greedy_search"
    assert captured["provider"] == "cpu"
    assert "hotwords_file" not in captured


def test_native_hotwords_require_modified_beam_search() -> None:
    backend = SherpaOnnxZipformerAsrBackend(_config("en"))

    with pytest.raises(ValueError, match="modified_beam_search"):
        backend.configure_native_hotwords(
            (
                SimpleNamespace(
                    text="WINDSURFING",
                    score=2.0,
                    token_count=1,
                ),
            ),
            global_score=1.5,
        )


def test_sherpa_creates_stream_with_profile_hotwords(
    tmp_path: Path,
) -> None:
    config = _config("zh")
    config.model_dir = str(tmp_path)
    config.sherpa.decoding_method = "modified_beam_search"
    _assets(tmp_path, config)
    backend = SherpaOnnxZipformerAsrBackend(config)
    (tmp_path / backend.model_spec.tokens).write_text(
        "<blk> 0\n紧 1\n急 2\n停 3\n止 4\n按 5\n钮 6\n",
        encoding="utf-8",
    )
    recognizer = _FakeRecognizer(("你好",))
    captured: dict[str, object] = {}

    class Factory:
        @staticmethod
        def from_transducer(**kwargs):
            captured.update(kwargs)
            return recognizer

    backend._sherpa = SimpleNamespace(OnlineRecognizer=Factory)
    backend.configure_native_hotwords(
        (
            SimpleNamespace(
                text="紧急停止按钮",
                score=1.8,
                token_count=6,
            ),
            SimpleNamespace(
                text="不存在",
                score=1.5,
                token_count=3,
            ),
        ),
        global_score=1.5,
    )
    backend.load()
    backend.transcribe(_segment(160), "zh")

    assert captured["decoding_method"] == "modified_beam_search"
    assert captured["modeling_unit"] == "cjkchar"
    assert captured["bpe_vocab"] == ""
    assert recognizer.hotword_inputs == ["紧急停止按钮 :1.8"]
    metrics = backend.take_metrics()
    assert metrics["asr_hotword_term_count"] == 1
    assert metrics["asr_hotword_rejection_count"] == 1
    assert metrics["asr_hotword_stream_count"] == 1


def test_bpe_hotword_vocab_is_exported_from_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config("en")
    backend = SherpaOnnxZipformerAsrBackend(config)
    (tmp_path / "bpe.model").write_bytes(b"fake")

    class Processor:
        def __init__(self, *, model_file):
            assert model_file.endswith("bpe.model")

        @staticmethod
        def get_piece_size():
            return 2

        @staticmethod
        def id_to_piece(index):
            return ("<unk>", "▁TEST")[index]

        @staticmethod
        def get_score(index):
            return (0.0, -1.5)[index]

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=Processor),
    )

    modeling_unit, vocab = backend._ensure_hotword_assets(
        tmp_path
    )

    assert modeling_unit == "bpe"
    assert Path(vocab).read_text(encoding="utf-8") == (
        "<unk>\t0.0\n▁TEST\t-1.5\n"
    )


def test_streaming_backend_feeds_only_unseen_audio_tail() -> None:
    config = _config("vi")
    config.sherpa.final_padding_ms = 0
    backend = SherpaOnnxZipformerAsrBackend(config)
    recognizer = _FakeRecognizer()
    backend._recognizer = recognizer

    partial = backend.transcribe(_segment(8_000), "vi")
    final = backend.transcribe(_segment(16_000, final=True), "vi")

    stream = recognizer.streams[0]
    assert [len(samples) for _, samples in stream.accepted] == [8_000, 8_000]
    assert partial.text == "xin"
    assert partial.revision == 1
    assert final.text == "xin chào"
    assert final.tokens == ("xin", "chào")
    assert final.revision == 2
    assert final.is_final
    assert stream.finished
    assert backend._stream is None
    assert backend._processed_samples == 0


def test_streaming_backend_redecodes_shrinking_semantic_final() -> None:
    config = _config("en")
    config.sherpa.final_padding_ms = 0
    backend = SherpaOnnxZipformerAsrBackend(config)
    backend._recognizer = _FakeRecognizer(("hello",))
    backend.transcribe(_segment(1_000), "en")

    final = backend.transcribe(_segment(500, final=True), "en")

    assert len(backend._recognizer.streams) == 2
    assert [
        len(samples)
        for _, samples in backend._recognizer.streams[1].accepted
    ] == [500]
    assert backend._recognizer.streams[1].finished
    assert final.is_final


def test_streaming_backend_still_rejects_shrinking_active_snapshot() -> None:
    backend = SherpaOnnxZipformerAsrBackend(_config("en"))
    backend._recognizer = _FakeRecognizer(("hello",))
    backend.transcribe(_segment(1_000), "en")

    with pytest.raises(ValueError, match="snapshot shrank within"):
        backend.transcribe(_segment(500), "en")


def test_streaming_backend_starts_fresh_stream_for_new_segment_identity() -> None:
    config = _config("vi")
    backend = SherpaOnnxZipformerAsrBackend(config)
    backend._recognizer = _FakeRecognizer()
    backend.transcribe(_segment(1_000), "vi")
    next_segment = SpeechSegment(
        samples=np.arange(400, dtype=np.float32),
        sample_rate=16_000,
        started_at=2.0,
        ended_at=2.025,
        is_final=False,
    )

    backend.transcribe(next_segment, "vi")

    assert len(backend._recognizer.streams) == 2
    assert [
        len(samples)
        for _, samples in backend._recognizer.streams[1].accepted
    ] == [400]


def test_sherpa_normalizes_all_caps_and_uses_english_punctuator() -> None:
    vi_backend = SherpaOnnxZipformerAsrBackend(_config("vi"))
    assert (
        vi_backend._normalize_text("XIN CHÀO. TÔI KHỎE!", "vi")
        == "Xin chào. Tôi khỏe!"
    )

    class _FakePunctuator:
        @staticmethod
        def add_punctuation_with_case(text: str) -> str:
            assert text == "how are you i am fine"
            return "How are you? I am fine."

    en_backend = SherpaOnnxZipformerAsrBackend(_config("en"))
    en_backend._punctuator = _FakePunctuator()
    assert (
        en_backend._normalize_text("HOW ARE YOU I AM FINE", "en")
        == "How are you? I am fine."
    )


def test_sherpa_exposes_word_timings_from_subword_tokens() -> None:
    backend = SherpaOnnxZipformerAsrBackend(_config("en"))
    backend._recognizer = _FakeTimedRecognizer(
        ("HOW ARE YOU I AM FINE",)
    )

    update = backend.transcribe(_segment(24_000), "en")

    assert [word.text for word in update.words] == [
        "HOW",
        "ARE",
        "YOU",
        "I",
        "AM",
        "FINE",
    ]
    assert update.words[1].start_seconds == pytest.approx(0.3)
    assert update.words[1].end_seconds == pytest.approx(0.54)
    assert update.words[-1].end_seconds == pytest.approx(1.5)


def test_sherpa_uses_endpoint_overlap_as_context_not_transcript() -> None:
    words = (
        AsrWordTiming("INSTRUCTOR", 0.02, 0.16),
        AsrWordTiming("I'M", 0.24, 0.38),
        AsrWordTiming("HERE", 0.4, 0.6),
    )

    text, visible = SherpaOnnxZipformerAsrBackend._strip_acoustic_context(
        "INSTRUCTOR I'M HERE",
        words,
        language="en",
        context_seconds=0.2,
    )

    assert text == "I'M HERE"
    assert [word.text for word in visible] == ["I'M", "HERE"]
    assert visible[0].start_seconds == pytest.approx(0.04)


def test_offline_mode_fails_before_download_when_model_is_missing(
    tmp_path: Path,
) -> None:
    config = _config("ko")
    config.offline = True
    config.sherpa.cache_dir = str(tmp_path)
    backend = SherpaOnnxZipformerAsrBackend(config)

    with pytest.raises(FileNotFoundError, match="disable offline mode"):
        backend.load()


def test_reset_and_close_clear_streaming_state() -> None:
    backend = SherpaOnnxZipformerAsrBackend(_config("zh"))
    backend._recognizer = _FakeRecognizer(("你好",))
    backend.transcribe(_segment(100), "zh")

    backend.reset()
    assert backend._stream is None
    assert backend._processed_samples == 0
    assert backend._segment_started_at is None
    assert backend._revision == 0
    backend.close()
    assert backend._recognizer is None
    assert backend._sherpa is None
