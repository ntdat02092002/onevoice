from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    asr_chunk_ms: int = 500
    queue_seconds: int = 20


@dataclass(slots=True)
class VadConfig:
    backend: str = "webrtc"
    aggressiveness: int = 2
    min_speech_ms: int = 250
    end_silence_ms: int = 600
    speech_padding_ms: int = 200
    max_utterance_seconds: int = 15
    semantic_endpoint_enabled: bool = True
    semantic_endpoint_sentences: int = 2


@dataclass(slots=True)
class AsrConfig:
    backend: str = "moonshine"
    model: str = "auto"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "vi"
    beam_size: int = 1
    model_dir: str | None = None
    update_interval: float = 0.5
    decoding_method: str = "attention"
    offline: bool = False


@dataclass(slots=True)
class CommitConfig:
    backend: str = "local_agreement"
    agreement_updates: int = 2
    hold_tokens: int = 1


@dataclass(slots=True)
class TranslationConfig:
    backend: str = "opus_ct2"
    model: str = "opus-auto"
    device: str = "cpu"
    compute_type: str = "int8"
    model_dir: str | None = None
    source_language: str = "auto"
    target_language: str = "en"
    wait_tokens: int = 3
    update_tokens: int = 3
    timeout_ms: int = 800
    max_new_tokens: int = 256
    offline: bool = False


@dataclass(slots=True)
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)

    def validate(self) -> None:
        if self.vad.backend == "webrtc":
            if self.audio.sample_rate != 16_000:
                raise ValueError("WebRTC VAD requires 16000 Hz audio")
            if self.audio.frame_ms not in (10, 20, 30):
                raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")
            if not 0 <= self.vad.aggressiveness <= 3:
                raise ValueError("WebRTC VAD aggressiveness must be between 0 and 3")
        if self.vad.min_speech_ms <= 0 or self.vad.end_silence_ms <= 0:
            raise ValueError("VAD min_speech_ms and end_silence_ms must be positive")
        if self.vad.max_utterance_seconds <= 0:
            raise ValueError("VAD max_utterance_seconds must be positive")
        if self.vad.semantic_endpoint_sentences < 1:
            raise ValueError("semantic_endpoint_sentences must be at least 1")
        if self.translation.target_language not in ("vi", "en", "zh", "ko"):
            raise ValueError("target language must be vi, en, zh, or ko")
        if self.translation.source_language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("translation source language must be auto, vi, en, zh, or ko")
        if self.asr.language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("ASR language must be auto, vi, en, zh, or ko")
        if self.asr.update_interval <= 0:
            raise ValueError("ASR update_interval must be positive")


T = TypeVar("T")


def _from_mapping(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    if path is None:
        candidate = Path("config/default.yaml")
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) if candidate.exists() else {}
    else:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = PipelineConfig(
        audio=_from_mapping(AudioConfig, data.get("audio")),
        vad=_from_mapping(VadConfig, data.get("vad")),
        asr=_from_mapping(AsrConfig, data.get("asr")),
        commit=_from_mapping(CommitConfig, data.get("commit")),
        translation=_from_mapping(TranslationConfig, data.get("translation")),
    )
    config.validate()
    return config
