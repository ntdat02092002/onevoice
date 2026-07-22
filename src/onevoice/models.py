from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Mapping

import numpy as np


SUPPORTED_LANGUAGES = ("vi", "en", "zh", "ko")


class EventType(StrEnum):
    STATUS = "status"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    ASR_PARTIAL = "asr_partial"
    ASR_COMMITTED = "asr_committed"
    ASR_FINAL = "asr_final"
    TRANSLATION_PARTIAL = "translation_partial"
    TRANSLATION_FINAL = "translation_final"
    TTS_PARTIAL = "tts_partial"
    TTS_FINAL = "tts_final"
    OVERLOAD = "overload"
    ERROR = "error"
    METRIC = "metric"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    sequence: int
    captured_at: float = field(default_factory=monotonic)
    end_of_stream: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.samples.ndim != 1:
            raise ValueError("AudioChunk samples must be mono (one-dimensional)")

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    samples: np.ndarray
    sample_rate: int
    started_at: float
    ended_at: float
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class AsrUpdate:
    text: str
    language: str | None
    confidence: float | None
    revision: int
    is_final: bool
    started_at: float
    completed_at: float = field(default_factory=monotonic)
    tokens: tuple[str, ...] = ()

    @property
    def latency_ms(self) -> float:
        return max(0.0, (self.completed_at - self.started_at) * 1000)


@dataclass(frozen=True, slots=True)
class CommittedTranscript:
    text: str
    language: str
    revision: int
    is_final: bool
    tokens: tuple[str, ...]
    committed_at: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    text: str
    source_language: str
    target_language: str
    source_revision: int
    is_final: bool
    requested_at: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class TranslationUpdate:
    text: str
    source_text: str
    source_language: str
    target_language: str
    source_revision: int
    is_final: bool
    started_at: float
    completed_at: float = field(default_factory=monotonic)

    @property
    def latency_ms(self) -> float:
        return max(0.0, (self.completed_at - self.started_at) * 1000)


@dataclass(frozen=True, slots=True)
class TtsRequest:
    text: str
    language: str
    source_revision: int
    is_final: bool
    phrase_id: int = 0
    source_is_final: bool = False
    requested_at: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class TtsUpdate:
    samples: np.ndarray
    sample_rate: int
    text: str
    language: str
    source_revision: int
    is_final: bool
    phrase_id: int
    started_at: float
    source_is_final: bool = False
    completed_at: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError("TTS samples must be mono (one-dimensional)")
        if self.sample_rate <= 0:
            raise ValueError("TTS sample_rate must be positive")

    @property
    def latency_ms(self) -> float:
        return max(0.0, (self.completed_at - self.started_at) * 1000)

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def real_time_factor(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return (self.latency_ms / 1000) / self.duration_seconds


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    type: EventType
    payload: Any = None
    message: str = ""
    emitted_at: float = field(default_factory=monotonic)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload
        if hasattr(payload, "__dataclass_fields__"):
            payload = asdict(payload)
        if isinstance(payload, dict):
            payload = {
                key: ({"sample_count": len(value)} if isinstance(value, np.ndarray) else value)
                for key, value in payload.items()
            }
        elif isinstance(payload, np.ndarray):
            payload = {"sample_count": len(payload)}
        return {
            "type": self.type.value,
            "message": self.message,
            "emitted_at": self.emitted_at,
            "metrics": dict(self.metrics),
            "payload": payload,
        }

