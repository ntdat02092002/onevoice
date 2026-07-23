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
    wait_tokens: int = 6
    update_tokens: int = 4
    timeout_ms: int = 1200
    min_request_interval_ms: int = 500
    zh_wait_tokens: int = 12
    zh_update_tokens: int = 8
    zh_timeout_ms: int = 1000
    max_new_tokens: int = 256
    sentence_boundary_only: bool = False
    offline: bool = False


@dataclass(slots=True)
class TtsConfig:
    enabled: bool = False
    backend: str = "sherpa_onnx"
    model: str = "auto"
    model_dir: str | None = None
    tokens: str | None = None
    lexicon: str | None = None
    data_dir: str | None = None
    rule_fsts: list[str] = field(default_factory=list)
    language: str = "auto"
    cache_dir: str = ".cache/onevoice/tts"
    offline: bool = False
    device: str = "cpu"
    num_threads: int = 2
    speaker_id: int = 0
    speed: float = 0.9
    num_steps: int = 8
    min_chunk_tokens: int = 8
    max_chunk_tokens: int = 24
    agreement_updates: int = 2
    timeout_ms: int = 1200
    sentence_boundary_only: bool = True
    final_only: bool = True
    emission_mode: str | None = None


@dataclass(slots=True)
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)

    def validate(self) -> None:
        if self.audio.asr_chunk_ms <= 0:
            raise ValueError("audio.asr_chunk_ms must be positive")
        if self.audio.queue_seconds <= 0:
            raise ValueError("audio.queue_seconds must be positive")
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
        if self.commit.agreement_updates < 1:
            raise ValueError("commit.agreement_updates must be at least 1")
        if self.commit.hold_tokens < 0:
            raise ValueError("commit.hold_tokens must be non-negative")
        if self.translation.target_language not in ("vi", "en", "zh", "ko"):
            raise ValueError("target language must be vi, en, zh, or ko")
        if self.translation.source_language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("translation source language must be auto, vi, en, zh, or ko")
        if self.asr.language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("ASR language must be auto, vi, en, zh, or ko")
        if self.asr.update_interval <= 0:
            raise ValueError("ASR update_interval must be positive")
        if self.translation.wait_tokens < 1 or self.translation.update_tokens < 1:
            raise ValueError("translation wait/update tokens must be at least 1")
        if self.translation.timeout_ms <= 0 or self.translation.min_request_interval_ms <= 0:
            raise ValueError("translation timing values must be positive")
        if self.translation.max_new_tokens < 1:
            raise ValueError("translation.max_new_tokens must be at least 1")
        if self.translation.zh_wait_tokens < 1 or self.translation.zh_update_tokens < 1:
            raise ValueError("Chinese translation wait/update tokens must be at least 1")
        if self.translation.zh_timeout_ms <= 0:
            raise ValueError("translation.zh_timeout_ms must be positive")
        if self.tts.device not in ("cpu", "cuda", "coreml"):
            raise ValueError("TTS device must be cpu, cuda, or coreml")
        if self.tts.language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("TTS language must be auto, vi, en, zh, or ko")
        if self.tts.num_threads < 1:
            raise ValueError("TTS num_threads must be at least 1")
        if self.tts.speed <= 0:
            raise ValueError("TTS speed must be positive")
        if self.tts.num_steps < 1:
            raise ValueError("TTS num_steps must be at least 1")
        if self.tts.min_chunk_tokens < 1:
            raise ValueError("TTS min_chunk_tokens must be at least 1")
        if self.tts.max_chunk_tokens < self.tts.min_chunk_tokens:
            raise ValueError("TTS max_chunk_tokens must be >= min_chunk_tokens")
        if self.tts.agreement_updates < 1:
            raise ValueError("TTS agreement_updates must be at least 1")
        if self.tts.timeout_ms <= 0:
            raise ValueError("TTS timeout_ms must be positive")
        if self.tts.emission_mode not in (None, "final_utterance", "stable_sentence", "stable_phrase"):
            raise ValueError(
                "TTS emission_mode must be final_utterance, stable_sentence, or stable_phrase"
            )


T = TypeVar("T")


def _from_mapping(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge a profile mapping over the project default mapping."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config(path: str | Path | None = None) -> PipelineConfig:
    default_path = Path("config/default.yaml")
    if path is None:
        data = _read_yaml_mapping(default_path) if default_path.exists() else {}
    else:
        selected_path = Path(path)
        profile = _read_yaml_mapping(selected_path)
        is_default = (
            default_path.exists()
            and selected_path.resolve() == default_path.resolve()
        )
        if default_path.exists() and not is_default:
            data = _deep_merge(_read_yaml_mapping(default_path), profile)
        else:
            data = profile
    config = PipelineConfig(
        audio=_from_mapping(AudioConfig, data.get("audio")),
        vad=_from_mapping(VadConfig, data.get("vad")),
        asr=_from_mapping(AsrConfig, data.get("asr")),
        commit=_from_mapping(CommitConfig, data.get("commit")),
        translation=_from_mapping(TranslationConfig, data.get("translation")),
        tts=_from_mapping(TtsConfig, data.get("tts")),
    )
    config.validate()
    return config
