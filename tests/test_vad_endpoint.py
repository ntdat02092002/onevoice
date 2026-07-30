from __future__ import annotations

import numpy as np
import pytest

from onevoice.backends.vad import PassthroughVad, WebRtcVadBackend
from onevoice.config import AudioConfig, VadConfig
from onevoice.models import (
    AsrWordTiming,
    AudioChunk,
    CommittedTranscript,
    SpeechSegment,
)
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


def test_passthrough_endpoint_carries_configured_acoustic_context() -> None:
    audio = AudioConfig()
    config = VadConfig(semantic_endpoint_context_ms=20)
    vad = PassthroughVad(config, audio)
    first = vad.process(_chunk(640))[0]

    vad.request_endpoint(
        started_at=first.started_at,
        cut_sample=640,
    )
    final = vad.process(_chunk(320, sequence=1))[0]
    carried = vad.process(_chunk(320, sequence=2))[0]

    assert final.samples.size == 640
    assert carried.samples.size == 960
    assert carried.context_samples == 320


def test_webrtc_endpoint_cuts_at_snapshot_and_carries_buffered_suffix() -> None:
    audio = AudioConfig(frame_ms=20, asr_chunk_ms=20)
    vad = WebRtcVadBackend(VadConfig(min_speech_ms=20), audio)
    vad._vad = _AlwaysSpeech()
    vad.reset()

    assert vad.process(_chunk(320)) == []
    snapshot = vad.process(_chunk(320, sequence=1))[0]
    assert not snapshot.is_final
    assert snapshot.samples.size == 640

    # More audio reaches VAD while ASR decides the earlier snapshot is a
    # complete semantic endpoint.
    vad.process(_chunk(320, sequence=2))
    vad.request_endpoint(
        started_at=snapshot.started_at,
        cut_sample=snapshot.samples.size,
    )
    final = vad.process(_chunk(320, sequence=3))[0]

    assert final.is_final
    assert final.is_endpoint_cut
    assert final.started_at == snapshot.started_at
    assert final.samples.size == snapshot.samples.size

    carried = vad.process(_chunk(320, sequence=4))[0]
    assert not carried.is_final
    assert carried.started_at != final.started_at
    assert carried.samples.size == 960
    assert carried.context_samples == 0


def test_webrtc_ignores_endpoint_for_an_old_utterance_identity() -> None:
    audio = AudioConfig(frame_ms=20, asr_chunk_ms=20)
    vad = WebRtcVadBackend(VadConfig(min_speech_ms=20), audio)
    vad._vad = _AlwaysSpeech()
    vad.reset()

    assert vad.process(_chunk(320)) == []
    snapshot = vad.process(_chunk(320, sequence=1))[0]
    vad.request_endpoint(
        started_at=snapshot.started_at - 1.0,
        cut_sample=snapshot.samples.size,
    )
    next_segment = vad.process(_chunk(320, sequence=2))[0]

    assert not next_segment.is_final
    assert next_segment.samples.size == 960


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


def test_pipeline_endpoint_carries_snapshot_sample_cursor() -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(1)
    segment = SpeechSegment(
        np.zeros(1_280, dtype=np.float32),
        16_000,
        started_at=42.0,
        ended_at=42.08,
        is_final=False,
    )

    pipeline._maybe_request_semantic_endpoint(
        _committed("One."),
        active_hypothesis="One.",
        segment=segment,
    )

    assert vad._endpoint_requested.is_set()
    assert vad._endpoint_request is not None
    assert vad._endpoint_request.started_at == 42.0
    assert vad._endpoint_request.cut_sample == 1_280


@pytest.mark.parametrize(
    ("threshold", "text", "words", "expected_seconds"),
    [
        (
            1,
            "One. unfinished tail",
            (
                AsrWordTiming("One", 0.1, 0.4),
                AsrWordTiming("unfinished", 0.5, 0.8),
                AsrWordTiming("tail", 0.9, 1.1),
            ),
            0.4,
        ),
        (
            1,
            "One. Two? unfinished tail",
            (
                AsrWordTiming("One", 0.1, 0.3),
                AsrWordTiming("Two", 0.4, 0.7),
                AsrWordTiming("unfinished", 0.8, 1.0),
                AsrWordTiming("tail", 1.1, 1.3),
            ),
            0.3,
        ),
        (
            2,
            "One. Two? unfinished tail",
            (
                AsrWordTiming("One", 0.1, 0.3),
                AsrWordTiming("Two", 0.4, 0.7),
                AsrWordTiming("unfinished", 0.8, 1.0),
                AsrWordTiming("tail", 1.1, 1.3),
            ),
            0.7,
        ),
    ],
)
def test_timestamped_endpoint_cuts_before_trailing_active_fragment(
    threshold: int,
    text: str,
    words: tuple[AsrWordTiming, ...],
    expected_seconds: float,
) -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(threshold)
    segment = SpeechSegment(
        np.zeros(32_000, dtype=np.float32),
        16_000,
        started_at=50.0,
        ended_at=52.0,
        is_final=False,
    )

    pipeline._maybe_request_semantic_endpoint(
        _committed(text),
        active_hypothesis=text,
        segment=segment,
        word_timings=words,
    )

    assert vad._endpoint_request is not None
    assert vad._endpoint_request.cut_sample == round(expected_seconds * 16_000)


def test_unaligned_word_timestamps_do_not_relax_trailing_fragment_guard() -> None:
    pipeline, vad = _pipeline_with_semantic_threshold(1)
    segment = SpeechSegment(
        np.zeros(16_000, dtype=np.float32),
        16_000,
        started_at=60.0,
        ended_at=61.0,
        is_final=False,
    )

    pipeline._maybe_request_semantic_endpoint(
        _committed("One. unfinished"),
        active_hypothesis="One. unfinished",
        segment=segment,
        word_timings=(AsrWordTiming("Different", 0.1, 0.4),),
    )

    assert not vad._endpoint_requested.is_set()
