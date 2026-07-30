from __future__ import annotations

import queue
import threading
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite
from time import monotonic
from typing import Any

import numpy as np

from . import backends as _builtin_backends  # noqa: F401
from .config import PipelineConfig
from .models import (
    AsrUpdate,
    AsrWordTiming,
    AudioChunk,
    CommittedTranscript,
    EventType,
    PipelineEvent,
    SpeechSegment,
    TranslationRequest,
    TtsRequest,
)
from .policy import PhraseTtsPolicy, WaitKTranslationPolicy
from .protocols import (
    AudioPreprocessor,
    CommitPolicy,
    StreamingAsrBackend,
    TranslationBackend,
    TtsBackend,
    VadBackend,
)
from .registry import registry
from .terminology import TerminologyCoverageError, TerminologyManager
from .text import (
    detokenize,
    ends_phrase,
    sentence_token_boundaries,
    tokenize_text,
)


_STOP = object()


@dataclass(slots=True)
class _AsrJob:
    generation: int
    utterance_id: int
    segment: SpeechSegment


@dataclass(slots=True)
class _TranslationJob:
    generation: int
    utterance_id: int
    request: TranslationRequest


@dataclass(slots=True)
class _TtsJob:
    generation: int
    utterance_id: int
    request: TtsRequest


class EnqueueResult(StrEnum):
    ENQUEUED = "enqueued"
    REPLACED_PARTIAL = "replaced_partial"
    DROPPED_PARTIAL = "dropped_partial"


class RealtimePipeline:
    """Threaded, bounded-queue ASR -> stable-prefix -> MT -> TTS pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        preprocessor: AudioPreprocessor | None = None,
        vad: VadBackend | None = None,
        asr: StreamingAsrBackend | None = None,
        committer: CommitPolicy | None = None,
        translator: TranslationBackend | None = None,
        tts: TtsBackend | None = None,
    ) -> None:
        self.config = config
        config.validate()
        if config.vad.semantic_endpoint_context_ms is None:
            config.vad.semantic_endpoint_context_ms = (
                200 if config.asr.backend == "sherpa_onnx" else 0
            )
        if config.tts.language == "auto":
            config.tts.language = config.translation.target_language
        self.preprocessor = preprocessor or registry.create("preprocessor", "passthrough")
        self.vad = vad or registry.create(
            "vad", config.vad.backend, config=config.vad, audio_config=config.audio
        )
        self.asr = asr or registry.create("asr", config.asr.backend, config=config.asr)
        self.committer = committer or registry.create(
            "commit", config.commit.backend, config=config.commit
        )
        self.translator = translator or registry.create(
            "translation", config.translation.backend, config=config.translation
        )
        self.terminology_manager: TerminologyManager | None = None
        if config.terminology.enabled:
            assert config.terminology.bundle_path is not None
            self.terminology_manager = TerminologyManager.from_path(
                config.terminology.bundle_path,
                case_sensitive_for_codes=(
                    config.terminology.matching.case_sensitive_for_codes
                ),
            )
            configure_terminology = getattr(
                self.translator, "configure_terminology", None
            )
            if not callable(configure_terminology):
                raise ValueError(
                    f"Translation backend {type(self.translator).__name__} "
                    "does not support terminology"
                )
            configure_terminology(
                self.terminology_manager,
                domain=config.terminology.domain,
                config=config.terminology.mt,
            )
        self.tts = tts or registry.create("tts", config.tts.backend, config=config.tts)
        audio_capacity = max(8, config.audio.queue_seconds * 1000 // config.audio.frame_ms)
        self._audio_queue: queue.Queue[Any] = queue.Queue(maxsize=audio_capacity)
        self._asr_queue: queue.Queue[Any] = queue.Queue(maxsize=4)
        self._translation_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        # A final translation can split into several sentence-aware phrases at
        # once. Keep enough room for the whole utterance so UI playback does
        # not lose an early phrase while synthesis catches up.
        self._tts_queue: queue.Queue[Any] = queue.Queue(maxsize=32)
        self._events: queue.Queue[PipelineEvent] = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._reset_audio = threading.Event()
        self._semantic_endpoint_pending = threading.Event()
        self._semantic_endpoint_final: tuple[tuple[str, ...], str] | None = (
            None
        )
        self._generation_lock = threading.Lock()
        self._generation = 0
        self._latest_translation_revisions: dict[tuple[int, int], int] = {}
        self._latest_lock = threading.Lock()
        self._metrics: Counter[str] = Counter()
        self._metrics_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._speech_active = False
        self._utterance_sequence = 0
        self._active_utterance_id = 0
        self._translation_policy = WaitKTranslationPolicy(config.translation)
        self._tts_policy = PhraseTtsPolicy(config.tts)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    def metrics_snapshot(self) -> dict[str, int]:
        with self._metrics_lock:
            return dict(self._metrics)

    def _count_metric(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            self._metrics[name] += amount

    def start(self, *, load_models: bool = True) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._emit(EventType.STATUS, message="loading")
        loaded: list[Any] = []
        started_threads: list[threading.Thread] = []
        try:
            for backend in (self.preprocessor, self.vad, self.committer):
                loaded.append(backend)
                backend.load()
            if load_models:
                # sherpa-onnx and Moonshine both bundle ONNX Runtime on Windows.
                # Load sherpa's newer ORT first so Moonshine can use its
                # backwards-compatible API in the same process.
                model_backends = (
                    [self.tts] if self.config.tts.enabled else []
                ) + [self.asr, self.translator]
                for backend in model_backends:
                    loaded.append(backend)
                    backend.load()
            self._threads = [
                threading.Thread(target=self._audio_worker, name="onevoice-audio", daemon=True),
                threading.Thread(target=self._asr_worker, name="onevoice-asr", daemon=True),
                threading.Thread(target=self._translation_worker, name="onevoice-mt", daemon=True),
            ]
            if self.config.tts.enabled:
                self._threads.append(
                    threading.Thread(target=self._tts_worker, name="onevoice-tts", daemon=True)
                )
            for thread in self._threads:
                thread.start()
                started_threads.append(thread)
        except Exception:
            self._rollback_start(loaded, started_threads)
            raise
        self._started = True
        self._emit(EventType.STATUS, message="listening")

    def _rollback_start(
        self, loaded: list[Any], started_threads: list[threading.Thread]
    ) -> None:
        self._stop_event.set()
        for work_queue in (
            self._audio_queue,
            self._asr_queue,
            self._translation_queue,
            self._tts_queue,
        ):
            try:
                work_queue.put_nowait(_STOP)
            except queue.Full:
                pass
        for thread in started_threads:
            thread.join(timeout=1)
        for backend in reversed(loaded):
            try:
                backend.close()
            except Exception:
                pass
        for work_queue in (
            self._audio_queue,
            self._asr_queue,
            self._translation_queue,
            self._tts_queue,
        ):
            self._drain_queue(work_queue)
        self._threads = []
        self._tts_policy.reset()
        self._translation_policy.reset()
        self._semantic_endpoint_pending.clear()
        self._semantic_endpoint_final = None
        self._stop_event.clear()
        self._started = False

    def push_audio(self, chunk: AudioChunk) -> bool:
        if not self.is_running:
            raise RuntimeError("Pipeline must be started before pushing audio")
        try:
            self._audio_queue.put_nowait(chunk)
            return True
        except queue.Full:
            self._handle_overload()
            return False

    def finish(self) -> None:
        if not self.is_running:
            return
        sequence = int(monotonic() * 1000)
        eos = AudioChunk(
            np.empty(0, dtype=np.float32),
            self.config.audio.sample_rate,
            sequence,
            end_of_stream=True,
        )
        try:
            self._audio_queue.put(eos, timeout=1)
        except queue.Full:
            self._handle_overload()
            self._audio_queue.put(eos, timeout=1)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        done = threading.Event()

        def wait_queues() -> None:
            self._audio_queue.join()
            self._asr_queue.join()
            self._translation_queue.join()
            self._tts_queue.join()
            done.set()

        waiter = threading.Thread(target=wait_queues, daemon=True)
        waiter.start()
        return done.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        if not self._started:
            return
        self.finish()
        self.wait_until_idle(timeout)
        self._stop_event.set()
        work_queues = [self._audio_queue, self._asr_queue, self._translation_queue]
        if self.config.tts.enabled:
            work_queues.append(self._tts_queue)
        for work_queue in work_queues:
            try:
                work_queue.put_nowait(_STOP)
            except queue.Full:
                pass
        for thread in self._threads:
            thread.join(timeout=timeout / max(1, len(self._threads)))
        for backend in (
            self.preprocessor,
            self.vad,
            self.asr,
            self.committer,
            self.translator,
            self.tts,
        ):
            backend.close()
        self._translation_policy.reset()
        self._tts_policy.reset()
        self._semantic_endpoint_pending.clear()
        self._semantic_endpoint_final = None
        with self._latest_lock:
            self._latest_translation_revisions.clear()
        self._started = False
        self._emit(EventType.STATUS, message="stopped")

    def poll_events(self, limit: int = 100) -> list[PipelineEvent]:
        output: list[PipelineEvent] = []
        for _ in range(limit):
            try:
                output.append(self._events.get_nowait())
                self._events.task_done()
            except queue.Empty:
                break
        return output

    def _current_generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def _handle_overload(self) -> None:
        with self._generation_lock:
            self._generation += 1
        self._reset_audio.set()
        self._semantic_endpoint_pending.clear()
        self._semantic_endpoint_final = None
        with self._latest_lock:
            self._latest_translation_revisions.clear()
        self._drain_queue(self._audio_queue)
        self._drain_queue(self._asr_queue)
        self._drain_queue(self._translation_queue)
        self._drain_queue(self._tts_queue)
        self._tts_policy.reset()
        self._emit(EventType.OVERLOAD, message="Audio queue full; current utterance was reset")

    @staticmethod
    def _drain_queue(work_queue: queue.Queue[Any]) -> None:
        while True:
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                return

    def _audio_worker(self) -> None:
        while not self._stop_event.is_set():
            item = self._audio_queue.get()
            try:
                if item is _STOP:
                    return
                if self._reset_audio.is_set():
                    self.vad.reset()
                    self._semantic_endpoint_pending.clear()
                    self._semantic_endpoint_final = None
                    self._speech_active = False
                    self._reset_audio.clear()
                chunk = self.preprocessor.process(item)
                for segment in self.vad.process(chunk):
                    if not self._speech_active:
                        self._speech_active = True
                        self._utterance_sequence += 1
                        self._active_utterance_id = self._utterance_sequence
                        self._emit(EventType.SPEECH_START, message="speech")
                    asr_result = self._enqueue_asr(
                        _AsrJob(
                            self._current_generation(),
                            self._active_utterance_id,
                            segment,
                        )
                    )
                    if asr_result == EnqueueResult.REPLACED_PARTIAL:
                        self._count_metric("asr_partials_coalesced")
                    elif asr_result == EnqueueResult.DROPPED_PARTIAL:
                        self._count_metric("asr_partials_dropped")
                    if segment.is_final:
                        self._speech_active = False
                        self._emit(EventType.SPEECH_END, message="transcribing")
            except Exception as exc:
                self._emit(EventType.ERROR, message=f"Audio/VAD error: {exc}")
            finally:
                self._audio_queue.task_done()

    def _asr_worker(self) -> None:
        seen_generation = -1
        seen_stream_id: tuple[int, int] | None = None
        while not self._stop_event.is_set():
            item = self._asr_queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _AsrJob)
                if item.generation != self._current_generation():
                    continue
                if item.generation != seen_generation:
                    self.asr.reset()
                    self.committer.reset()
                    self._translation_policy.reset()
                    seen_generation = item.generation
                    seen_stream_id = None
                stream_id = (item.generation, item.utterance_id)
                if stream_id != seen_stream_id:
                    if seen_stream_id is not None:
                        self.asr.reset()
                        self.committer.reset()
                        self._translation_policy.reset()
                    seen_stream_id = stream_id
                if (
                    self._semantic_endpoint_pending.is_set()
                    and not item.segment.is_final
                ):
                    self._count_metric("asr_partials_frozen_at_endpoint")
                    continue
                language = self.config.asr.language
                if language == "auto":
                    language = self.config.translation.source_language
                update: AsrUpdate = self.asr.transcribe(item.segment, language)
                if item.generation != self._current_generation():
                    continue
                if update.is_final:
                    endpoint_final = self._semantic_endpoint_final
                    if item.segment.is_endpoint_cut and endpoint_final is not None:
                        endpoint_tokens, endpoint_language = endpoint_final
                        update = replace(
                            update,
                            text=detokenize(
                                endpoint_tokens, endpoint_language
                            ),
                            tokens=endpoint_tokens,
                        )
                    self._semantic_endpoint_pending.clear()
                    self._semantic_endpoint_final = None
                event_type = EventType.ASR_FINAL if update.is_final else EventType.ASR_PARTIAL
                self._emit(
                    event_type,
                    payload=update,
                    metrics={"asr_latency_ms": update.latency_ms},
                )
                committed = self.committer.update(update)
                if committed is None:
                    continue
                endpoint_boundary = self._maybe_request_semantic_endpoint(
                    committed,
                    update.text,
                    item.segment,
                    update.words,
                )
                if endpoint_boundary is not None:
                    endpoint_tokens = committed.tokens[:endpoint_boundary]
                    self._semantic_endpoint_final = (
                        endpoint_tokens,
                        committed.language,
                    )
                    committed = replace(
                        committed,
                        text=detokenize(endpoint_tokens, committed.language),
                        tokens=endpoint_tokens,
                    )
                    self.committer.reset()
                    self._translation_policy.reset()
                self._emit(
                    EventType.ASR_COMMITTED,
                    payload=committed,
                    metrics={
                        "asr_commit_latency_ms": max(
                            0.0,
                            (committed.committed_at - item.segment.started_at) * 1000,
                        )
                    },
                )
                request = self._translation_policy.request_for(
                    committed, self.config.translation.target_language
                )
                if request is not None:
                    revision_key = (item.generation, item.utterance_id)
                    request = replace(request, stream_id=revision_key)
                    enqueue_started = monotonic()
                    enqueue_result = self._enqueue_translation(
                        _TranslationJob(item.generation, item.utterance_id, request)
                    )
                    if enqueue_result in (
                        EnqueueResult.ENQUEUED,
                        EnqueueResult.REPLACED_PARTIAL,
                    ):
                        self._translation_policy.mark_enqueued(request)
                        self._count_metric("mt_requests_enqueued")
                        if enqueue_result == EnqueueResult.REPLACED_PARTIAL:
                            self._count_metric("mt_partials_coalesced")
                        if request.is_final:
                            wait_ms = int((monotonic() - enqueue_started) * 1000)
                            self._count_metric("mt_final_queue_wait_ms", wait_ms)
                    else:
                        self._count_metric("mt_partials_dropped")
                if committed.is_final:
                    self._translation_policy.reset()
                    self.asr.reset()
            except Exception as exc:
                # A native streaming recognizer may have consumed part of the
                # failed snapshot. Drop that state so the next growing
                # snapshot can be decoded from its complete audio instead of
                # repeating the same cursor error forever.
                self.asr.reset()
                self._semantic_endpoint_pending.clear()
                self._semantic_endpoint_final = None
                self._emit(EventType.ERROR, message=f"ASR error: {exc}")
            finally:
                self._asr_queue.task_done()

    def _maybe_request_semantic_endpoint(
        self,
        committed: CommittedTranscript,
        active_hypothesis: str | None = None,
        segment: SpeechSegment | None = None,
        word_timings: tuple[AsrWordTiming, ...] = (),
    ) -> int | None:
        config = self.config.vad
        active_text = active_hypothesis or committed.text
        boundaries = sentence_token_boundaries(
            committed.text, committed.language
        )
        if (
            committed.is_final
            or not config.semantic_endpoint_enabled
            or self._semantic_endpoint_pending.is_set()
            or len(boundaries) < config.semantic_endpoint_sentences
        ):
            return None
        cut_sample = self._aligned_sentence_cut_sample(
            committed,
            segment,
            word_timings,
            config.semantic_endpoint_sentences,
        )
        if cut_sample is None and (
            not ends_phrase(committed.text, committed.language)
            or not ends_phrase(active_text, committed.language)
        ):
            return None
        request_endpoint = getattr(self.vad, "request_endpoint", None)
        if request_endpoint is None:
            self._emit(
                EventType.ERROR,
                message="Semantic endpoint is enabled but the VAD backend does not support it",
            )
            return None
        self._semantic_endpoint_pending.set()
        try:
            if segment is None:
                request_endpoint()
            else:
                request_endpoint(
                    started_at=segment.started_at,
                    cut_sample=cut_sample or len(segment.samples),
                )
        except Exception:
            self._semantic_endpoint_pending.clear()
            raise
        boundary_index = (
            config.semantic_endpoint_sentences - 1
            if cut_sample is not None
            else len(boundaries) - 1
        )
        return boundaries[boundary_index]

    @staticmethod
    def _aligned_sentence_cut_sample(
        committed: CommittedTranscript,
        segment: SpeechSegment | None,
        word_timings: tuple[AsrWordTiming, ...],
        sentence_count: int,
    ) -> int | None:
        """Map the configured stable sentence boundary to its final word end."""
        if segment is None or not word_timings:
            return None
        boundaries = sentence_token_boundaries(committed.text, committed.language)
        if len(boundaries) < sentence_count:
            return None
        target_tokens = committed.tokens[: boundaries[sentence_count - 1]]
        target = tuple(
            token.casefold()
            for token in target_tokens
            if any(character.isalnum() for character in token)
        )
        if not target:
            return None

        timed_tokens: list[tuple[str, AsrWordTiming]] = []
        for timing in word_timings:
            for token in tokenize_text(timing.text, committed.language):
                if any(character.isalnum() for character in token):
                    timed_tokens.append((token.casefold(), timing))
        if len(timed_tokens) < len(target):
            return None
        if tuple(token for token, _ in timed_tokens[: len(target)]) != target:
            return None

        end_seconds = timed_tokens[len(target) - 1][1].end_seconds
        if not isfinite(end_seconds):
            return None
        cut_sample = round(end_seconds * segment.sample_rate)
        if cut_sample <= 0:
            return None
        return min(cut_sample, len(segment.samples))

    def _enqueue_asr(self, item: _AsrJob) -> EnqueueResult:
        """Keep only the newest growing partial per utterance; finals are lossless."""
        result = EnqueueResult.ENQUEUED
        coalesced = 0
        with self._asr_queue.mutex:
            stream_id = (item.generation, item.utterance_id)
            matching_partials = [
                index
                for index, queued in enumerate(self._asr_queue.queue)
                if isinstance(queued, _AsrJob)
                and not queued.segment.is_final
                and (queued.generation, queued.utterance_id) == stream_id
            ]

            if not item.segment.is_final and matching_partials:
                keep_index = matching_partials[-1]
                self._asr_queue.queue[keep_index] = item
                for index in reversed(matching_partials[:-1]):
                    del self._asr_queue.queue[index]
                    self._asr_queue.unfinished_tasks -= 1
                    coalesced += 1
                result = EnqueueResult.REPLACED_PARTIAL
                self._asr_queue.not_empty.notify()
            elif item.segment.is_final:
                for index in reversed(matching_partials):
                    del self._asr_queue.queue[index]
                    self._asr_queue.unfinished_tasks -= 1
                    coalesced += 1
            elif self._asr_queue._qsize() >= self._asr_queue.maxsize:
                replace_index = next(
                    (
                        index
                        for index, queued in enumerate(self._asr_queue.queue)
                        if isinstance(queued, _AsrJob) and not queued.segment.is_final
                    ),
                    None,
                )
                if replace_index is None:
                    return EnqueueResult.DROPPED_PARTIAL
                self._asr_queue.queue[replace_index] = item
                self._asr_queue.not_empty.notify()
                return EnqueueResult.REPLACED_PARTIAL

            if result != EnqueueResult.REPLACED_PARTIAL:
                # Final insertion may exceed maxsize when the queue contains
                # only finals. This is the same logical lossless lane used by
                # MT finals.
                self._asr_queue._put(item)
                self._asr_queue.unfinished_tasks += 1
                self._asr_queue.not_empty.notify()

        if coalesced:
            self._count_metric("asr_partials_coalesced", coalesced)
        return result

    def _translation_worker(self) -> None:
        while not self._stop_event.is_set():
            item = self._translation_queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _TranslationJob)
                if item.generation != self._current_generation():
                    continue
                queue_delay_ms = max(
                    0.0, (monotonic() - item.request.requested_at) * 1000
                )
                self._count_metric("mt_inference_count")
                result = self.translator.translate(item.request)
                self._count_metric(
                    "terminology_matches", result.terminology_matches
                )
                self._count_metric(
                    "terminology_hard_matches",
                    result.terminology_hard_matches,
                )
                self._count_metric(
                    "mt_placeholder_expected",
                    result.terminology_expected_placeholders,
                )
                self._count_metric(
                    "mt_placeholder_retry", result.terminology_retries
                )
                self._count_metric(
                    "mt_terminology_fallback",
                    result.terminology_fallbacks,
                )
                self._count_metric(
                    "mt_sentence_cache_hits", result.sentence_cache_hits
                )
                revision_key = (item.generation, item.utterance_id)
                with self._latest_lock:
                    latest = self._latest_translation_revisions.get(revision_key, -1)
                if item.generation != self._current_generation() or result.source_revision < latest:
                    self._count_metric("mt_stale_results")
                    continue
                event_type = (
                    EventType.TRANSLATION_FINAL if result.is_final else EventType.TRANSLATION_PARTIAL
                )
                self._emit(
                    event_type,
                    payload=result,
                    metrics={
                        "mt_latency_ms": result.latency_ms,
                        "mt_queue_delay_ms": queue_delay_ms,
                        "mt_request_tokens": float(
                            len(
                                tokenize_text(
                                    item.request.text,
                                    item.request.source_language,
                                )
                            )
                        ),
                        "terminology_matches": float(
                            result.terminology_matches
                        ),
                        "mt_placeholder_retry": float(
                            result.terminology_retries
                        ),
                        "mt_terminology_fallback": float(
                            result.terminology_fallbacks
                        ),
                        "mt_sentence_cache_hits": float(
                            result.sentence_cache_hits
                        ),
                    },
                )
                if self.config.tts.enabled:
                    stream_id = (item.generation, item.utterance_id)
                    for request in self._tts_policy.requests_for(result, stream_id):
                        accepted, evicted = self._put_latest(
                            self._tts_queue,
                            _TtsJob(item.generation, item.utterance_id, request),
                            request.source_is_final,
                            stage="TTS",
                        )
                        if isinstance(evicted, _TtsJob):
                            self._tts_policy.cancel(evicted.request.phrase_id)
                        if not accepted:
                            self._tts_policy.cancel(request.phrase_id)
                        else:
                            self._count_metric("tts_requests_enqueued")
                if result.is_final:
                    with self._latest_lock:
                        self._latest_translation_revisions.pop(revision_key, None)
            except Exception as exc:
                if isinstance(exc, TerminologyCoverageError):
                    self._count_metric("mt_terminology_validation_error")
                if isinstance(item, _TranslationJob) and item.request.is_final:
                    stream_id = (item.generation, item.utterance_id)
                    self._tts_policy.reset_stream(stream_id)
                self._emit(EventType.ERROR, message=f"Translation error: {exc}")
            finally:
                self._translation_queue.task_done()

    def _tts_worker(self) -> None:
        seen_generation = -1
        while not self._stop_event.is_set():
            item = self._tts_queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _TtsJob)
                if item.generation != self._current_generation():
                    self._tts_policy.cancel(item.request.phrase_id)
                    continue
                if not self._tts_policy.is_reserved(item.request.phrase_id):
                    continue
                if item.generation != seen_generation:
                    self.tts.reset()
                    seen_generation = item.generation
                self._count_metric("tts_inference_count")
                result = self.tts.synthesize(item.request)
                if item.generation != self._current_generation():
                    self._tts_policy.cancel(item.request.phrase_id)
                    continue
                # Phrase validity is content-based inside PhraseTtsPolicy. A
                # newer utterance revision does not invalidate audio whose
                # exact stable prefix/sentence is still reserved.
                if not self._tts_policy.mark_synthesized(item.request.phrase_id):
                    continue
                event_type = EventType.TTS_FINAL if result.is_final else EventType.TTS_PARTIAL
                delivered = self._emit(
                    event_type,
                    payload=result,
                    metrics={
                        "tts_latency_ms": result.latency_ms,
                        "tts_rtf": result.real_time_factor,
                        "tts_queue_delay_ms": max(
                            0.0,
                            (result.started_at - item.request.requested_at) * 1000,
                        ),
                        "tts_audio_duration_ms": result.duration_seconds * 1000,
                    },
                )
                if not delivered:
                    self._tts_policy.cancel(item.request.phrase_id)
            except Exception as exc:
                if isinstance(item, _TtsJob):
                    stream_id = (item.generation, item.utterance_id)
                    self._tts_policy.reset_stream(stream_id)
                    try:
                        self.tts.reset()
                    except Exception:
                        pass
                self._emit(EventType.ERROR, message=f"TTS error: {exc}")
            finally:
                self._tts_queue.task_done()

    def _enqueue_translation(self, item: _TranslationJob) -> EnqueueResult:
        """Coalesce pending partials per stream while keeping every final lossless."""
        stream_id = (item.generation, item.utterance_id)
        coalesced = False
        result = EnqueueResult.ENQUEUED
        # Lock order is latest -> queue everywhere that touches both. Final jobs
        # use an unbounded logical lane, so this transaction never waits while
        # holding either lock. The worker cannot observe a job before its
        # accepted revision is published.
        with self._latest_lock, self._translation_queue.mutex:
            for index, queued in enumerate(self._translation_queue.queue):
                if (
                    isinstance(queued, _TranslationJob)
                    and not queued.request.is_final
                    and (queued.generation, queued.utterance_id) == stream_id
                ):
                    if item.request.is_final:
                        del self._translation_queue.queue[index]
                        self._translation_queue.unfinished_tasks -= 1
                        coalesced = True
                    else:
                        self._translation_queue.queue[index] = item
                        result = EnqueueResult.REPLACED_PARTIAL
                    break

            if result != EnqueueResult.REPLACED_PARTIAL:
                if not item.request.is_final and (
                    self._translation_queue._qsize() >= self._translation_queue.maxsize
                ):
                    result = EnqueueResult.DROPPED_PARTIAL
                else:
                    # Final insertion may exceed Queue.maxsize. This is the
                    # dedicated lossless final lane; only partials are bounded.
                    self._translation_queue._put(item)
                    self._translation_queue.unfinished_tasks += 1

            if result in (EnqueueResult.ENQUEUED, EnqueueResult.REPLACED_PARTIAL):
                self._latest_translation_revisions[stream_id] = item.request.source_revision
                self._translation_queue.not_empty.notify()

        if result == EnqueueResult.DROPPED_PARTIAL:
            self._emit(
                EventType.OVERLOAD,
                message="MT queue is lagging; partial update coalesced/dropped",
            )
            return result
        if coalesced:
            self._count_metric("mt_partials_coalesced")
        return result

    def _put_latest(
        self,
        work_queue: queue.Queue[Any],
        item: Any,
        must_keep: bool,
        *,
        stage: str,
    ) -> tuple[bool, Any | None]:
        try:
            work_queue.put_nowait(item)
            return True, None
        except queue.Full:
            # Never let a newer partial snapshot evict an already queued final.
            if not must_keep:
                self._emit(
                    EventType.OVERLOAD,
                    message=f"{stage} queue is lagging; partial update skipped, final is preserved",
                )
                return False, None
            # A final is lossless pipeline state: dropping an older final here
            # removes a complete utterance from both transcript and translation
            # history. Reclaim a queued partial if possible; if the queue only
            # contains finals, apply backpressure until the consumer advances.
            evicted = self._evict_oldest_partial(work_queue)
            while not self._stop_event.is_set():
                try:
                    work_queue.put(item, timeout=0.1)
                    return True, evicted
                except queue.Full:
                    continue
            return False, evicted

    @staticmethod
    def _evict_oldest_partial(work_queue: queue.Queue[Any]) -> Any | None:
        """Remove one non-final job without disturbing queued final jobs."""
        with work_queue.mutex:
            for index, queued in enumerate(work_queue.queue):
                if queued is _STOP or RealtimePipeline._is_final_job(queued):
                    continue
                del work_queue.queue[index]
                work_queue.unfinished_tasks -= 1
                if work_queue.unfinished_tasks == 0:
                    work_queue.all_tasks_done.notify_all()
                work_queue.not_full.notify()
                return queued
        return None

    @staticmethod
    def _is_final_job(item: Any) -> bool:
        if isinstance(item, _AsrJob):
            return item.segment.is_final
        if isinstance(item, (_TranslationJob, _TtsJob)):
            return item.request.is_final or bool(
                getattr(item.request, "source_is_final", False)
            )
        # Unknown queue items are treated as non-droppable.
        return True

    def acknowledge_tts(self, phrase_id: int) -> bool:
        """Acknowledge that a synthesized phrase entered the playback queue."""
        return self._tts_policy.acknowledge(phrase_id) is not None

    def _emit(
        self,
        event_type: EventType,
        *,
        payload: Any = None,
        message: str = "",
        metrics: dict[str, float] | None = None,
    ) -> bool:
        event = PipelineEvent(event_type, payload, message, metrics=metrics or {})
        try:
            self._events.put_nowait(event)
            return True
        except queue.Full:
            # UI history is built from terminal events. A burst of partials
            # must never evict an already completed utterance.
            if not self._is_lossless_event(event):
                return False
            discarded = self._evict_oldest_partial_event()
            if discarded is not None:
                self._cancel_discarded_tts_event(discarded)
            while not self._stop_event.is_set():
                try:
                    self._events.put(event, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

    @staticmethod
    def _is_lossless_event(event: PipelineEvent) -> bool:
        if event.type in (
            EventType.ERROR,
            EventType.OVERLOAD,
            EventType.ASR_FINAL,
            EventType.TRANSLATION_FINAL,
        ):
            return True
        if event.type == EventType.ASR_COMMITTED:
            return bool(event.payload and event.payload.is_final)
        if event.type == EventType.TTS_FINAL:
            return True
        if event.type == EventType.TTS_PARTIAL and bool(
            getattr(event.payload, "source_is_final", False)
        ):
            return True
        return False

    def _evict_oldest_partial_event(self) -> PipelineEvent | None:
        with self._events.mutex:
            for index, queued in enumerate(self._events.queue):
                if self._is_lossless_event(queued):
                    continue
                del self._events.queue[index]
                self._events.unfinished_tasks -= 1
                if self._events.unfinished_tasks == 0:
                    self._events.all_tasks_done.notify_all()
                self._events.not_full.notify()
                return queued
        return None

    def _cancel_discarded_tts_event(self, event: PipelineEvent) -> None:
        if event.type not in (EventType.TTS_PARTIAL, EventType.TTS_FINAL):
            return
        self._tts_policy.cancel(event.payload.phrase_id)
