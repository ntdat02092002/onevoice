from __future__ import annotations

import numpy as np

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


def test_pipeline_requests_endpoint_after_configured_sentence_count() -> None:
    from onevoice.config import PipelineConfig

    config = PipelineConfig()
    config.vad.backend = "passthrough"
    vad = PassthroughVad(config.vad, config.audio)
    pipeline = RealtimePipeline(config, vad=vad)
    committed = CommittedTranscript(
        "One. Two? unfinished",
        "en",
        revision=3,
        is_final=False,
        tokens=("One", ".", "Two", "?", "unfinished"),
    )

    pipeline._maybe_request_semantic_endpoint(committed)

    assert vad._endpoint_requested.is_set()
