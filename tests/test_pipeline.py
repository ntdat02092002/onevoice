from __future__ import annotations

import time
import threading
import queue

import numpy as np
import pytest

from onevoice.config import PipelineConfig
from onevoice.models import (
    AudioChunk,
    AsrUpdate,
    CommittedTranscript,
    EventType,
    PipelineEvent,
    SpeechSegment,
    TranslationRequest,
    TranslationUpdate,
    TtsUpdate,
)
from onevoice.pipeline import (
    EnqueueResult,
    RealtimePipeline,
    _AsrJob,
    _TranslationJob,
    _TtsJob,
)


def test_fake_pipeline_emits_partial_committed_and_final_translation() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    config.asr.language = "en"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    config.tts.enabled = True
    config.tts.backend = "fake"
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
        assert EventType.TTS_FINAL in types
        translation = next(
            event.payload
            for event in events
            if event.type == EventType.TRANSLATION_FINAL
        )
        assert translation.text.startswith("[vi]")
        speech = next(event.payload for event in events if event.type == EventType.TTS_FINAL)
        assert speech.samples.dtype == np.float32
        assert speech.sample_rate == 16_000
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


def test_final_job_evicts_only_partial_and_preserves_older_final() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    work_queue: queue.Queue = queue.Queue(maxsize=2)
    older_final = _TranslationJob(0, 1, TranslationRequest("first", "en", "vi", 1, True))
    disposable_partial = _TranslationJob(
        0, 2, TranslationRequest("second partial", "en", "vi", 1, False)
    )
    newer_final = _TranslationJob(0, 2, TranslationRequest("second", "en", "vi", 2, True))
    work_queue.put(older_final)
    work_queue.put(disposable_partial)

    accepted, evicted = pipeline._put_latest(
        work_queue, newer_final, True, stage="MT"
    )

    assert accepted
    assert evicted is disposable_partial
    assert list(work_queue.queue) == [older_final, newer_final]


def test_final_job_waits_instead_of_dropping_another_final() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    work_queue: queue.Queue = queue.Queue(maxsize=1)
    older_final = _TranslationJob(0, 1, TranslationRequest("first", "en", "vi", 1, True))
    newer_final = _TranslationJob(0, 2, TranslationRequest("second", "en", "vi", 1, True))
    work_queue.put(older_final)
    result: list[tuple[bool, object | None]] = []
    producer = threading.Thread(
        target=lambda: result.append(
            pipeline._put_latest(work_queue, newer_final, True, stage="MT")
        )
    )
    producer.start()
    time.sleep(0.05)
    assert producer.is_alive()
    assert work_queue.get_nowait() is older_final
    work_queue.task_done()
    producer.join(timeout=1)

    assert result == [(True, None)]
    assert work_queue.get_nowait() is newer_final
    work_queue.task_done()


def test_partial_event_cannot_evict_completed_transcript_history() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    pipeline._events = queue.Queue(maxsize=1)
    final = PipelineEvent(
        EventType.ASR_COMMITTED,
        CommittedTranscript("kept final", "en", 1, True, ("kept", "final")),
    )
    pipeline._events.put(final)

    assert not pipeline._emit(EventType.ASR_PARTIAL, message="disposable partial")
    assert pipeline._events.get_nowait() is final
    pipeline._events.task_done()


def test_all_tts_chunks_from_final_translation_are_lossless_events() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    pipeline._events = queue.Queue(maxsize=1)
    speech = TtsUpdate(
        samples=np.zeros(160, dtype=np.float32),
        sample_rate=16_000,
        text="first chunk of final",
        language="vi",
        source_revision=4,
        is_final=False,
        phrase_id=7,
        started_at=time.monotonic(),
        source_is_final=True,
    )
    final_source_chunk = PipelineEvent(EventType.TTS_PARTIAL, speech)
    pipeline._events.put(final_source_chunk)

    assert not pipeline._emit(EventType.ASR_PARTIAL, message="disposable")
    assert pipeline._events.get_nowait() is final_source_chunk
    pipeline._events.task_done()


def test_asr_final_survives_partial_event_pressure() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    pipeline._events = queue.Queue(maxsize=3)
    for revision in range(3):
        assert pipeline._emit(EventType.ASR_PARTIAL, message=f"partial {revision}")
    final = AsrUpdate(
        text="complete final",
        language="en",
        confidence=1.0,
        revision=4,
        is_final=True,
        started_at=time.monotonic(),
    )

    assert pipeline._emit(EventType.ASR_FINAL, payload=final)
    for revision in range(20):
        pipeline._emit(EventType.ASR_PARTIAL, message=f"later partial {revision}")

    events = pipeline.poll_events(100)
    finals = [event for event in events if event.type == EventType.ASR_FINAL]
    assert len(finals) == 1
    assert finals[0].payload is final


def _translation_job(
    revision: int, *, utterance_id: int = 1, final: bool = False
) -> _TranslationJob:
    return _TranslationJob(
        0,
        utterance_id,
        TranslationRequest(
            f"revision {revision}", "en", "vi", revision, final
        ),
    )


def test_mt_pending_partial_is_latest_only_and_final_supersedes_it() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    results = [
        pipeline._enqueue_translation(_translation_job(revision))
        for revision in range(1, 101)
    ]

    assert results[0] == EnqueueResult.ENQUEUED
    assert all(result == EnqueueResult.REPLACED_PARTIAL for result in results[1:])
    assert pipeline._translation_queue.qsize() == 1
    assert pipeline._translation_queue.queue[0].request.source_revision == 100

    final = _translation_job(101, final=True)
    assert pipeline._enqueue_translation(final) == EnqueueResult.ENQUEUED
    assert list(pipeline._translation_queue.queue) == [final]
    with pipeline._latest_lock:
        assert pipeline._latest_translation_revisions[(0, 1)] == 101
    pipeline._drain_queue(pipeline._translation_queue)


def test_dropped_partial_revision_does_not_invalidate_existing_inference() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    assert (
        pipeline._enqueue_translation(_translation_job(1, utterance_id=2))
        == EnqueueResult.ENQUEUED
    )
    assert (
        pipeline._enqueue_translation(_translation_job(1, utterance_id=3))
        == EnqueueResult.ENQUEUED
    )
    with pipeline._latest_lock:
        pipeline._latest_translation_revisions[(0, 1)] = 5

    result = pipeline._enqueue_translation(_translation_job(6, utterance_id=1))

    assert result == EnqueueResult.DROPPED_PARTIAL
    with pipeline._latest_lock:
        assert pipeline._latest_translation_revisions[(0, 1)] == 5
    pipeline._drain_queue(pipeline._translation_queue)


def test_mt_lossless_final_lane_keeps_ten_finals_fifo() -> None:
    pipeline = RealtimePipeline(PipelineConfig())
    finals = [_translation_job(1, utterance_id=index, final=True) for index in range(1, 11)]

    assert all(
        pipeline._enqueue_translation(item) == EnqueueResult.ENQUEUED
        for item in finals
    )

    assert list(pipeline._translation_queue.queue) == finals
    assert pipeline._translation_queue.qsize() == 10
    pipeline._drain_queue(pipeline._translation_queue)


class _TaggedFinalAsr:
    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def transcribe(self, segment, language):
        tag = int(round(float(np.mean(segment.samples)) * 20))
        text = f"utterance {tag}."
        return AsrUpdate(
            text=text,
            language="en",
            confidence=1.0,
            revision=1,
            is_final=segment.is_final,
            started_at=time.monotonic(),
            tokens=("utterance", str(tag), "."),
        )


class _SlowEchoTranslator:
    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def translate(self, request):
        time.sleep(0.01)
        return TranslationUpdate(
            text=request.text,
            source_text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            started_at=time.monotonic(),
        )


def test_ten_backlogged_utterance_finals_survive_end_to_end_in_order() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.vad.semantic_endpoint_enabled = False
    config.asr.backend = "fake"
    config.asr.language = "en"
    config.translation.backend = "fake"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    pipeline = RealtimePipeline(
        config, asr=_TaggedFinalAsr(), translator=_SlowEchoTranslator()
    )
    pipeline.start(load_models=False)
    try:
        for tag in range(1, 11):
            samples = np.full(1600, tag / 20, dtype=np.float32)
            assert pipeline.push_audio(AudioChunk(samples, 16_000, tag))
            pipeline.finish()
        assert pipeline.wait_until_idle(timeout=10)
        events = pipeline.poll_events(5000)
        asr_finals = [
            event.payload.text for event in events if event.type == EventType.ASR_FINAL
        ]
        mt_finals = [
            event.payload.source_text
            for event in events
            if event.type == EventType.TRANSLATION_FINAL
        ]
        expected = [f"utterance {tag}." for tag in range(1, 11)]
        assert asr_finals == expected
        assert mt_finals == expected
    finally:
        pipeline.close()


def test_overload_generation_drops_old_asr_mt_tts_and_processes_new_final() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.asr.language = "en"
    config.translation.backend = "fake"
    config.translation.source_language = "en"
    config.translation.target_language = "vi"
    config.tts.enabled = True
    config.tts.backend = "fake"
    tts = _RecordingTts()
    pipeline = RealtimePipeline(
        config,
        asr=_TaggedFinalAsr(),
        translator=_SlowEchoTranslator(),
        tts=tts,
    )
    pipeline.start(load_models=False)
    try:
        pipeline._handle_overload()
        segment = SpeechSegment(
            np.full(1600, 1 / 20, dtype=np.float32),
            16_000,
            time.monotonic(),
            time.monotonic(),
            True,
        )
        old_tts = pipeline._tts_policy.requests_for(
            _translated("old generation audio", 99), (0, 99)
        )[0]
        pipeline._asr_queue.put(_AsrJob(0, 99, segment))
        pipeline._translation_queue.put(_translation_job(99, utterance_id=99, final=True))
        pipeline._tts_queue.put(_TtsJob(0, 99, old_tts))
        pipeline._asr_queue.put(_AsrJob(1, 1, segment))

        assert pipeline.wait_until_idle(timeout=5)
        events = pipeline.poll_events(5000)
        assert [
            event.payload.text for event in events if event.type == EventType.ASR_FINAL
        ] == ["utterance 1."]
        assert [
            event.payload.source_text
            for event in events
            if event.type == EventType.TRANSLATION_FINAL
        ] == ["utterance 1."]
        assert all(request.text != "old generation audio" for request in tts.requests)
        assert any(request.text == "utterance 1." for request in tts.requests)
    finally:
        pipeline.close()


def test_tts_loads_before_asr_to_avoid_onnxruntime_abi_collision() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    config.tts.enabled = True
    config.tts.backend = "fake"
    order: list[str] = []
    pipeline = RealtimePipeline(config)
    pipeline.tts.load = lambda: order.append("tts")
    pipeline.asr.load = lambda: order.append("asr")
    pipeline.translator.load = lambda: order.append("translation")
    pipeline.start()
    try:
        assert order == ["tts", "asr", "translation"]
    finally:
        pipeline.close()


def test_start_rolls_back_loaded_backends_and_can_retry() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    pipeline = RealtimePipeline(config)
    calls: list[str] = []

    pipeline.preprocessor.load = lambda: calls.append("load_preprocessor")
    pipeline.preprocessor.close = lambda: calls.append("close_preprocessor")
    pipeline.vad.load = lambda: calls.append("load_vad")
    pipeline.vad.close = lambda: calls.append("close_vad")

    def fail_commit_load() -> None:
        calls.append("load_commit")
        raise RuntimeError("commit load failed")

    pipeline.committer.load = fail_commit_load
    pipeline.committer.close = lambda: calls.append("close_commit")

    with pytest.raises(RuntimeError, match="commit load failed"):
        pipeline.start(load_models=False)
    assert calls == [
        "load_preprocessor",
        "load_vad",
        "load_commit",
        "close_commit",
        "close_vad",
        "close_preprocessor",
    ]
    assert not pipeline.is_running

    pipeline.committer.load = lambda: calls.append("load_commit_retry")
    pipeline.start(load_models=False)
    try:
        assert pipeline.is_running
    finally:
        pipeline.close()


class _RecordingTts:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests = []
        self.reset_count = 0

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        pass

    def synthesize(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("model failed")
        return TtsUpdate(
            samples=np.zeros(1600, dtype=np.float32),
            sample_rate=16_000,
            text=request.text,
            language=request.language,
            source_revision=request.source_revision,
            is_final=request.is_final,
            phrase_id=request.phrase_id,
            started_at=time.monotonic(),
            source_is_final=request.source_is_final,
        )


def _tts_test_pipeline(tts) -> RealtimePipeline:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    config.tts.enabled = True
    config.tts.backend = "fake"
    config.tts.agreement_updates = 1
    config.tts.min_chunk_tokens = 1
    return RealtimePipeline(config, tts=tts)


def _translated(text: str, revision: int, *, final: bool = True) -> TranslationUpdate:
    return TranslationUpdate(
        text=text,
        source_text="source",
        source_language="en",
        target_language="vi",
        source_revision=revision,
        is_final=final,
        started_at=time.monotonic(),
    )


def test_tts_worker_keeps_older_revision_when_phrase_content_is_still_valid() -> None:
    tts = _RecordingTts()
    pipeline = _tts_test_pipeline(tts)
    pipeline.start(load_models=False)
    try:
        stream_id = (0, 1)
        request = pipeline._tts_policy.requests_for(
            _translated("old translation", 1), stream_id
        )[0]
        pipeline._tts_queue.put(_TtsJob(0, 1, request))
        assert pipeline.wait_until_idle(timeout=2)
        assert tts.requests == [request]
        assert pipeline._tts_policy.is_reserved(request.phrase_id)
    finally:
        pipeline.close()


def test_tts_model_error_resets_final_utterance_state() -> None:
    tts = _RecordingTts(fail=True)
    pipeline = _tts_test_pipeline(tts)
    pipeline.start(load_models=False)
    try:
        stream_id = (0, 1)
        request = pipeline._tts_policy.requests_for(
            _translated("final translation", 1), stream_id
        )[0]
        pipeline._tts_queue.put(_TtsJob(0, 1, request))
        assert pipeline.wait_until_idle(timeout=2)
        assert not pipeline._tts_policy.is_reserved(request.phrase_id)
        assert any("TTS error: model failed" in event.message for event in pipeline.poll_events())
    finally:
        pipeline.close()


class _FailingFinalTranslator:
    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def translate(self, request):
        raise RuntimeError("final MT failed")


def test_final_translation_error_resets_reserved_tts_state() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    config.tts.enabled = True
    config.tts.backend = "fake"
    config.tts.agreement_updates = 1
    config.tts.min_chunk_tokens = 1
    config.tts.final_only = False
    pipeline = RealtimePipeline(config, translator=_FailingFinalTranslator())
    pipeline.start(load_models=False)
    try:
        stream_id = (0, 1)
        reserved = pipeline._tts_policy.requests_for(
            _translated("reserved partial.", 1, final=False), stream_id
        )[0]
        request = TranslationRequest("source", "en", "vi", 2, True)
        pipeline._translation_queue.put(_TranslationJob(0, 1, request))
        assert pipeline.wait_until_idle(timeout=2)
        assert not pipeline._tts_policy.is_reserved(reserved.phrase_id)
    finally:
        pipeline.close()


class _BlockingTranslator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self._first = True
        self.requests = []

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.release.set()

    def translate(self, request):
        started_at = time.monotonic()
        self.requests.append(request)
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


def test_hundred_fast_partials_coalesce_to_latest_before_final() -> None:
    config = PipelineConfig()
    config.vad.backend = "passthrough"
    config.asr.backend = "fake"
    config.translation.backend = "fake"
    translator = _BlockingTranslator()
    pipeline = RealtimePipeline(config, translator=translator)
    pipeline.start(load_models=False)
    try:
        assert pipeline._enqueue_translation(_translation_job(1)) == EnqueueResult.ENQUEUED
        assert translator.started.wait(2)
        for revision in range(2, 101):
            pipeline._enqueue_translation(_translation_job(revision))
        assert pipeline._enqueue_translation(
            _translation_job(101, final=True)
        ) == EnqueueResult.ENQUEUED
        translator.release.set()

        assert pipeline.wait_until_idle(timeout=5)
        assert [request.source_revision for request in translator.requests] == [1, 101]
        finals = [
            event.payload.source_revision
            for event in pipeline.poll_events(1000)
            if event.type == EventType.TRANSLATION_FINAL
        ]
        assert finals == [101]
        assert pipeline.metrics_snapshot()["mt_stale_results"] == 1
    finally:
        translator.release.set()
        pipeline.close()


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
