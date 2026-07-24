from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onevoice.backends.asr import (
    DolphinAsrBackend,
    FasterWhisperBackend,
    MoonshineAsrBackend,
    asr_model_options,
)
from onevoice.config import AsrConfig
from onevoice.models import SpeechSegment


def _segment(sample_count: int, *, final: bool = False) -> SpeechSegment:
    return SpeechSegment(
        samples=np.zeros(sample_count, dtype=np.float32),
        sample_rate=16_000,
        started_at=0.0,
        ended_at=sample_count / 16_000,
        is_final=final,
    )


class _FakeMoonshineTranscriber:
    instances: list["_FakeMoonshineTranscriber"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.added: list[tuple[list[float], int]] = []
        self.starts = 0
        self.stops = 0
        self.updates = 0
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.starts += 1

    def add_audio(self, samples: list[float], sample_rate: int) -> None:
        self.added.append((samples, sample_rate))

    def update_transcription(self):
        self.updates += 1
        if self.updates == 1:
            lines = [
                SimpleNamespace(
                    text="hello",
                    start_time=0.0,
                    words=[
                        SimpleNamespace(
                            word="hello",
                            start=0.1,
                            end=0.4,
                            confidence=0.9,
                        )
                    ],
                )
            ]
        else:
            # Moonshine exposes newest lines first.
            lines = [
                SimpleNamespace(text="draft", start_time=0.5, words=[]),
                SimpleNamespace(text="hello", start_time=0.0, words=[]),
            ]
        return SimpleNamespace(lines=lines)

    def stop(self):
        self.stops += 1
        return SimpleNamespace(
            lines=[
                SimpleNamespace(text="world", start_time=0.5, words=[]),
                SimpleNamespace(text="hello", start_time=0.0, words=[]),
            ]
        )

    def close(self) -> None:
        pass


def test_moonshine_only_pushes_unseen_audio_tail() -> None:
    _FakeMoonshineTranscriber.instances.clear()
    backend = MoonshineAsrBackend(AsrConfig(backend="moonshine", language="vi", model="auto"))
    backend._transcriber_type = _FakeMoonshineTranscriber
    backend._get_model = lambda **kwargs: ("cached-model", 1)
    backend._model_arch_type = SimpleNamespace()

    first = backend.transcribe(_segment(8_000), "vi")
    final = backend.transcribe(_segment(16_000, final=True), "vi")

    native = _FakeMoonshineTranscriber.instances[0]
    assert [len(samples) for samples, _ in native.added] == [8_000, 8_000]
    assert native.starts == 1
    assert native.stops == 1
    assert first.text == "hello"
    assert first.words[0].text == "hello"
    assert first.words[0].end_seconds == 0.4
    assert final.text == "hello world"
    assert final.is_final
    assert native.kwargs["options"]["word_timestamps"] == "true"


class _FakeTensor:
    def unsqueeze(self, dimension: int):
        assert dimension == 0
        return self


class _FakeTorch:
    @staticmethod
    def from_numpy(samples: np.ndarray) -> _FakeTensor:
        assert samples.dtype == np.float32
        return _FakeTensor()


def test_dolphin_maps_result_and_rejects_english() -> None:
    backend = DolphinAsrBackend(AsrConfig(backend="dolphin", model="base", language="vi"))
    backend._model = object()
    backend._torch = _FakeTorch()
    backend._transcribe = lambda *args, **kwargs: SimpleNamespace(
        text_nospecial="xin chào", language="vi"
    )

    update = backend.transcribe(_segment(4_000), "vi")
    assert update.text == "xin chào"
    assert update.language == "vi"
    assert update.tokens == ("xin", "chào")

    with pytest.raises(ValueError, match="does not support English"):
        backend.transcribe(_segment(4_000), "en")


def test_faster_whisper_exposes_word_timestamps() -> None:
    backend = FasterWhisperBackend(
        AsrConfig(backend="faster_whisper", model="tiny", language="en")
    )
    captured: dict[str, object] = {}

    class _FakeWhisper:
        def transcribe(self, _samples, **kwargs):
            captured.update(kwargs)
            words = [
                SimpleNamespace(word="Hello", start=0.1, end=0.4, probability=0.8),
                SimpleNamespace(word="world", start=0.5, end=0.9, probability=0.9),
            ]
            return (
                iter([SimpleNamespace(text="Hello world.", words=words)]),
                SimpleNamespace(language="en", language_probability=0.99),
            )

    backend._model = _FakeWhisper()
    update = backend.transcribe(_segment(16_000), "en")

    assert captured["word_timestamps"] is True
    assert update.text == "Hello world."
    assert [word.text for word in update.words] == ["Hello", "world"]
    assert update.words[-1].end_seconds == 0.9


def test_asr_capabilities_fail_before_model_loading() -> None:
    assert asr_model_options("moonshine", "zh") == ("auto", "base")
    assert asr_model_options("moonshine", "en")[-1] == "medium_streaming"

    with pytest.raises(ValueError, match="not available for language 'zh'"):
        MoonshineAsrBackend(
            AsrConfig(backend="moonshine", language="zh", model="tiny_streaming")
        )
    with pytest.raises(ValueError, match="cannot auto-detect"):
        MoonshineAsrBackend(
            AsrConfig(backend="moonshine", language="auto", model="auto")
        )
    with pytest.raises(ValueError, match="does not support English"):
        DolphinAsrBackend(AsrConfig(backend="dolphin", language="en", model="base"))
