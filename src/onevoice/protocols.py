from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    AsrUpdate,
    AudioChunk,
    CommittedTranscript,
    SpeechSegment,
    TranslationRequest,
    TranslationUpdate,
    TtsRequest,
    TtsUpdate,
)


@runtime_checkable
class Lifecycle(Protocol):
    def load(self) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class AudioPreprocessor(Lifecycle, Protocol):
    def process(self, chunk: AudioChunk) -> AudioChunk: ...


@runtime_checkable
class VadBackend(Lifecycle, Protocol):
    def process(self, chunk: AudioChunk) -> list[SpeechSegment]: ...
    def flush(self) -> list[SpeechSegment]: ...
    def request_endpoint(self) -> None: ...


@runtime_checkable
class StreamingAsrBackend(Lifecycle, Protocol):
    def transcribe(self, segment: SpeechSegment, language: str | None) -> AsrUpdate: ...


@runtime_checkable
class CommitPolicy(Lifecycle, Protocol):
    def update(self, update: AsrUpdate) -> CommittedTranscript | None: ...


@runtime_checkable
class TranslationBackend(Lifecycle, Protocol):
    def translate(self, request: TranslationRequest) -> TranslationUpdate: ...


@runtime_checkable
class TtsBackend(Lifecycle, Protocol):
    def synthesize(self, request: TtsRequest) -> TtsUpdate: ...
