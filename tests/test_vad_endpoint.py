from __future__ import annotations

import numpy as np
import pytest

from onevoice.backends.vad import PassthroughVad, WebRtcVadBackend
from onevoice.config import AudioConfig, VadConfig
from onevoice.models import AudioChunk, CommittedTranscript
from onevoice.pipeline import RealtimePipeline


class _AlwaysSpeech:
    def is_speech(self, _pcm: bytes, _sample_rate: int) -> bool:
        return True


def _chunk(samples: int, sequence: int = 0) -> AudioChunk:
    return AudioChunk(np.zeros(samples, dtype=np.float32), 16_000, sequence)


def test_webrtc_vad_honors_requested_endpoint_on_audio_thread() -> None:
    audio = AudioConfig(frame_ms=20, asr_chunk_ms=500)
    vad = WebRtcVadBackend(VadConfig(min_speech_ms=20), audio)
    vad._vad = _AlwaysSpeech()
    vad.reset()

    assert vad.process(_chunk(320)) == []
    vad.request_endpoint()
    segments = vad.process(_chunk(320, sequence=1))

    assert len(segments) == 1
    assert segments[0].is_final
    assert segments[0].samples.size == 640


def test_passthrough_vad_honors_requested_endpoint() -> None:
    audio = AudioConfig()
    vad = PassthroughVad(VadConfig(), audio)
    assert len(vad.process(_chunk(320))) == 1

    vad.request_endpoint()
    segments = vad.process(_chunk(320, sequence=1))

    assert len(segments) == 1
    assert segments[0].is_final
    assert segments[0].samples.size == 640


def _pipeline_with_semantic_threshold(sentences: int) -> tuple[RealtimePipeline, PassthroughVad]:
    from onevoice.config import PipelineConfig

    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.vad.semantic_endpoint_sentences = sentences
    vad = PassthroughVad(config.vad, config.audio)
    return RealtimePipeline(config, vad=vad), vad


def _committed(text: str, revision: int = 1) -> CommittedTranscript:
    from onevoice.text import tokenize_text

    return CommittedTranscript(
        text,
        "en",
        revision=revision,
        is_final=False,
        tokens=tokenize_text(text, "en"),
    )


def test_pipeline_requests_endpoint_after_configured_sentence_count() -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(2)
    pipeline._maybe_request_semantic_endpoint(_committed("One. Two?"))

    assert vad._endpoint_requested.is_set()


@pytest.mark.parametrize(
    ("threshold", "text"),
    [(1, "One. unfinished"), (2, "One. Two? unfinished")],
)
def test_pipeline_does_not_endpoint_with_trailing_fragment(
    threshold: int, text: str
) -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(threshold)
    pipeline._maybe_request_semantic_endpoint(_committed(text))

    assert not vad._endpoint_requested.is_set()


@pytest.mark.parametrize(
    ("threshold", "text"),
    [(1, "One."), (2, "One. Two? First, I'm ready.")],
)
def test_pipeline_endpoints_when_threshold_met_without_trailing_fragment(
    threshold: int, text: str
) -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(threshold)
    pipeline._maybe_request_semantic_endpoint(_committed(text))

    assert vad._endpoint_requested.is_set()


def test_pipeline_does_not_endpoint_when_latest_asr_hypothesis_has_started_tail() -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(1)

    pipeline._maybe_request_semantic_endpoint(
        _committed("One."), active_hypothesis="One. unfinished"
    )

    assert not vad._endpoint_requested.is_set()
