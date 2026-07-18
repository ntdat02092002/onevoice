from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np

from . import backends as _builtin_backends  # noqa: F401
from .config import PipelineConfig
from .models import (
    AsrUpdate,
    AudioChunk,
    CommittedTranscript,
    EventType,
    PipelineEvent,
    SpeechSegment,
    TranslationRequest,
)
from .policy import WaitKTranslationPolicy
from .protocols import AudioPreprocessor, CommitPolicy, StreamingAsrBackend, TranslationBackend, VadBackend
from .registry import registry
from .text import count_complete_sentences


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


class RealtimePipeline:
    """Threaded, bounded-queue ASR -> stable-prefix -> MT pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        preprocessor: AudioPreprocessor | None = None,
        vad: VadBackend | None = None,
        asr: StreamingAsrBackend | None = None,
        committer: CommitPolicy | None = None,
        translator: TranslationBackend | None = None,
    ) -> None:
        self.config = config
        config.validate()
        self.preprocessor = preprocessor or registry.create("preprocessor", "passthrough")
        self.vad = vad or registry.create(
            "vad", config.vad.backend, config=config.vad, audio_config=config.audio
        )
        self.asr = asr or registry.create("asr", config.asr.backend, config=config.asr)
        self.committer = committer or registry.create("commit", config.commit.backend, config=config.commit)
        self.translator = translator or registry.create(
            "translation", config.translation.backend, config=config.translation
        )
        audio_capacity = max(8, config.audio.queue_seconds * 1000 // config.audio.frame_ms)
        self._audio_queue: queue.Queue[Any] = queue.Queue(maxsize=audio_capacity)
        self._asr_queue: queue.Queue[Any] = queue.Queue(maxsize=4)
        self._translation_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        self._events: queue.Queue[PipelineEvent] = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._reset_audio = threading.Event()
        self._semantic_endpoint_pending = threading.Event()
        self._generation_lock = threading.Lock()
        self._generation = 0
        self._latest_translation_revisions: dict[tuple[int, int], int] = {}
        self._latest_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._speech_active = False
        self._utterance_sequence = 0
        self._active_utterance_id = 0
        self._translation_policy = WaitKTranslationPolicy(config.translation)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    def start(self, *, load_models: bool = True) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._emit(EventType.STATUS, message="loading")
        self.preprocessor.load()
        self.vad.load()
        self.committer.load()
        if load_models:
            self.asr.load()
            self.translator.load()
        self._threads = [
            threading.Thread(target=self._audio_worker, name="onevoice-audio", daemon=True),
            threading.Thread(target=self._asr_worker, name="onevoice-asr", daemon=True),
            threading.Thread(target=self._translation_worker, name="onevoice-mt", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._started = True
        self._emit(EventType.STATUS, message="listening")

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
        eos = AudioChunk(np.empty(0, dtype=np.float32), self.config.audio.sample_rate, sequence, end_of_stream=True)
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
        for work_queue in (self._audio_queue, self._asr_queue, self._translation_queue):
            try:
                work_queue.put_nowait(_STOP)
            except queue.Full:
                pass
        for thread in self._threads:
            thread.join(timeout=timeout / max(1, len(self._threads)))
        for backend in (self.preprocessor, self.vad, self.asr, self.committer, self.translator):
            backend.close()
        self._started = False
        self._emit(EventType.STATUS, message="stopped")

    def poll_events(self, limit: int = 100) -> list[PipelineEvent]:
        output: list[PipelineEvent] = []
        for _ in range(limit):
            try:
                output.append(self._events.get_nowait())
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
        with self._latest_lock:
            self._latest_translation_revisions.clear()
        self._drain_queue(self._audio_queue)
        self._drain_queue(self._asr_queue)
        self._drain_queue(self._translation_queue)
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
                    self._speech_active = False
                    self._reset_audio.clear()
                chunk = self.preprocessor.process(item)
                for segment in self.vad.process(chunk):
                    if not self._speech_active:
                        self._speech_active = True
                        self._utterance_sequence += 1
                        self._active_utterance_id = self._utterance_sequence
                        self._emit(EventType.SPEECH_START, message="speech")
                    self._put_latest(
                        self._asr_queue,
                        _AsrJob(
                            self._current_generation(),
                            self._active_utterance_id,
                            segment,
                        ),
                        segment.is_final,
                    )
                    if segment.is_final:
                        self._speech_active = False
                        self._emit(EventType.SPEECH_END, message="transcribing")
            except Exception as exc:
                self._emit(EventType.ERROR, message=f"Audio/VAD error: {exc}")
            finally:
                self._audio_queue.task_done()

    def _asr_worker(self) -> None:
        seen_generation = -1
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
                language = self.config.asr.language
                if language == "auto":
                    language = self.config.translation.source_language
                update: AsrUpdate = self.asr.transcribe(item.segment, language)
                if item.generation != self._current_generation():
                    continue
                event_type = EventType.ASR_FINAL if update.is_final else EventType.ASR_PARTIAL
                self._emit(event_type, payload=update, metrics={"asr_latency_ms": update.latency_ms})
                committed = self.committer.update(update)
                if update.is_final:
                    self._semantic_endpoint_pending.clear()
                if committed is None:
                    continue
                self._emit(EventType.ASR_COMMITTED, payload=committed)
                self._maybe_request_semantic_endpoint(committed)
                request = self._translation_policy.request_for(
                    committed, self.config.translation.target_language
                )
                if request is not None:
                    revision_key = (item.generation, item.utterance_id)
                    with self._latest_lock:
                        self._latest_translation_revisions[revision_key] = request.source_revision
                    self._put_latest(
                        self._translation_queue,
                        _TranslationJob(item.generation, item.utterance_id, request),
                        request.is_final,
                    )
                if committed.is_final:
                    self._translation_policy.reset()
                    self.asr.reset()
            except Exception as exc:
                self._emit(EventType.ERROR, message=f"ASR error: {exc}")
            finally:
                self._asr_queue.task_done()

    def _maybe_request_semantic_endpoint(self, committed: CommittedTranscript) -> None:
        config = self.config.vad
        if (
            committed.is_final
            or not config.semantic_endpoint_enabled
            or self._semantic_endpoint_pending.is_set()
            or count_complete_sentences(committed.tokens) < config.semantic_endpoint_sentences
        ):
            return
        request_endpoint = getattr(self.vad, "request_endpoint", None)
        if request_endpoint is None:
            self._emit(
                EventType.ERROR,
                message="Semantic endpoint is enabled but the VAD backend does not support it",
            )
            return
        self._semantic_endpoint_pending.set()
        try:
            request_endpoint()
        except Exception:
            self._semantic_endpoint_pending.clear()
            raise

    def _translation_worker(self) -> None:
        while not self._stop_event.is_set():
            item = self._translation_queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _TranslationJob)
                if item.generation != self._current_generation():
                    continue
                result = self.translator.translate(item.request)
                revision_key = (item.generation, item.utterance_id)
                with self._latest_lock:
                    latest = self._latest_translation_revisions.get(revision_key, -1)
                if item.generation != self._current_generation() or result.source_revision < latest:
                    continue
                event_type = (
                    EventType.TRANSLATION_FINAL if result.is_final else EventType.TRANSLATION_PARTIAL
                )
                self._emit(event_type, payload=result, metrics={"mt_latency_ms": result.latency_ms})
                if result.is_final:
                    with self._latest_lock:
                        self._latest_translation_revisions.pop(revision_key, None)
            except Exception as exc:
                self._emit(EventType.ERROR, message=f"Translation error: {exc}")
            finally:
                self._translation_queue.task_done()

    def _put_latest(self, work_queue: queue.Queue[Any], item: Any, must_keep: bool) -> None:
        try:
            work_queue.put_nowait(item)
        except queue.Full:
            # Never let a newer partial snapshot evict an already queued final.
            if not must_keep:
                self._emit(EventType.OVERLOAD, message="Inference queue is lagging; partial update skipped")
                return
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                pass
            try:
                work_queue.put(item, timeout=1)
            except queue.Full:
                self._emit(EventType.OVERLOAD, message="Inference queue is lagging")

    def _emit(
        self,
        event_type: EventType,
        *,
        payload: Any = None,
        message: str = "",
        metrics: dict[str, float] | None = None,
    ) -> None:
        event = PipelineEvent(event_type, payload, message, metrics=metrics or {})
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                self._events.get_nowait()
                self._events.put_nowait(event)
            except queue.Empty:
                pass
