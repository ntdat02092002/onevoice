from __future__ import annotations

import time
import threading

import numpy as np

from onevoice.config import PipelineConfig
from onevoice.models import AudioChunk, EventType, TranslationUpdate
from onevoice.pipeline import RealtimePipeline


def test_fake_pipeline_emits_partial_committed_and_final_translation() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    config.asr.language = "en"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    pipeline = RealtimePipeline(config)
    pipeline.start(load_models=False)
    try:
        for sequence in range(3):
            samples = np.zeros(config.audio.sample_rate // 10, dtype=np.float32)
            assert pipeline.push_audio(AudioChunk(samples, config.audio.sample_rate, sequence))
            time.sleep(0.02)
        pipeline.finish()
        assert pipeline.wait_until_idle(timeout=5)
        events = pipeline.poll_events(1000)
        types = [event.type for event in events]
        assert EventType.ASR_PARTIAL in types
        assert EventType.ASR_COMMITTED in types
        assert EventType.ASR_FINAL in types
        assert EventType.TRANSLATION_FINAL in types
        translation = next(event.payload for event in events if event.type == EventType.TRANSLATION_FINAL)
        assert translation.text.startswith("[vi]")
    finally:
        pipeline.close()


def test_pipeline_rejects_audio_before_start() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    chunk = AudioChunk(np.zeros(320, dtype=np.float32), 16_000, 0)
    try:
        pipeline.push_audio(chunk)
    except RuntimeError as exc:
        assert "started" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


class _BlockingTranslator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self._first = True

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.release.set()

    def translate(self, request):
        started_at = time.monotonic()
        if self._first:
            self._first = False
            self.started.set()
            assert self.release.wait(2)
        return TranslationUpdate(
            text=f"[vi] {request.text}",
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=started_at,
        )


def test_translation_revisions_are_isolated_between_utterances() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.vad.semantic_endpoint_enabled = False
    config.asr.backend = "fake"
    config.asr.model = "fake"
    config.asr.language = "en"
    config.translation.backend = "fake"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    translator = _BlockingTranslator()
    pipeline = RealtimePipeline(config, translator=translator)
    pipeline.start(load_models=False)
    try:
        samples = np.zeros(config.audio.sample_rate // 10, dtype=np.float32)

        # Utterance A only produces a final translation at source revision 1.
        pipeline.push_audio(AudioChunk(samples, config.audio.sample_rate, 1))
        pipeline.finish()
        assert translator.started.wait(2)

        # While A is translating, utterance B advances its own revisions.
        for sequence in range(2, 5):
            pipeline.push_audio(AudioChunk(samples, config.audio.sample_rate, sequence))
            time.sleep(0.02)
        pipeline.finish()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with pipeline._latest_lock:
                has_second_utterance = any(
                    utterance_id == 2
                    for _, utterance_id in pipeline._latest_translation_revisions
                )
            if has_second_utterance:
                break
            time.sleep(0.01)
        assert has_second_utterance
        translator.release.set()

        assert pipeline.wait_until_idle(timeout=5)
        events = pipeline.poll_events(1000)
        committed_finals = [
            event for event in events
            if event.type == EventType.ASR_COMMITTED and event.payload.is_final
        ]
        translation_finals = [
            event for event in events if event.type == EventType.TRANSLATION_FINAL
        ]
        assert len(committed_finals) == 2
        assert len(translation_finals) == 2
    finally:
        translator.release.set()
        pipeline.close()
