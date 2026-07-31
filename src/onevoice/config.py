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
    semantic_endpoint_context_ms: int | None = None


@dataclass(slots=True)
class SherpaAsrConfig:
    recognizer_mode: str = "online_transducer"
    provider: str = "cpu"
    num_threads: int = 2
    decoding_method: str = "greedy_search"
    max_active_paths: int = 4
    final_padding_ms: int = 500
    cache_dir: str = ".cache/onevoice/asr"
    punctuation_enabled: bool = True


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
    sherpa: SherpaAsrConfig = field(default_factory=SherpaAsrConfig)


@dataclass(slots=True)
class CommitConfig:
    backend: str = "local_agreement"
    agreement_updates: int = 2
    hold_tokens: int = 1
    term_prefix_timeout_ms: int = 1_500


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
class TerminologyMatchingConfig:
    normalization: str = "unicode_nfc"
    longest_match_first: bool = True
    case_sensitive_for_codes: bool = True


@dataclass(slots=True)
class TerminologyAsrConfig:
    initial_prompt_enabled: bool = True
    post_correction_enabled: bool = True
    native_hotwords_enabled: bool = True
    max_prompt_terms: int = 32
    max_prompt_tokens: int = 128
    max_hotword_terms: int = 64
    max_hotword_tokens: int = 256
    hotword_score: float = 1.5


@dataclass(slots=True)
class TerminologyMtConfig:
    strategy: str = "placeholder_with_validation"
    placeholder_formats: list[str] = field(
        default_factory=lambda: [
            "__TERM_{id:04d}__",
            "OVT{id:04d}OVT",
            "ZXTERM{id:04d}ZX",
        ]
    )
    validate_coverage: bool = True
    pivot_canonicalization: bool = True
    validate_order: bool = False
    on_validation_error: str = "raise"


@dataclass(slots=True)
class TerminologyTtsConfig:
    strategy: str = "spoken_form"


@dataclass(slots=True)
class TerminologyConfig:
    enabled: bool = False
    bundle_path: str | None = None
    domain: str | None = None
    matching: TerminologyMatchingConfig = field(
        default_factory=TerminologyMatchingConfig
    )
    asr: TerminologyAsrConfig = field(
        default_factory=TerminologyAsrConfig
    )
    mt: TerminologyMtConfig = field(default_factory=TerminologyMtConfig)
    tts: TerminologyTtsConfig = field(
        default_factory=TerminologyTtsConfig
    )


@dataclass(slots=True)
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    terminology: TerminologyConfig = field(default_factory=TerminologyConfig)

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
        if (
            self.vad.semantic_endpoint_context_ms is not None
            and self.vad.semantic_endpoint_context_ms < 0
        ):
            raise ValueError(
                "semantic_endpoint_context_ms must be non-negative"
            )
        if self.commit.agreement_updates < 1:
            raise ValueError("commit.agreement_updates must be at least 1")
        if self.commit.hold_tokens < 0:
            raise ValueError("commit.hold_tokens must be non-negative")
        if self.commit.term_prefix_timeout_ms < 0:
            raise ValueError(
                "commit.term_prefix_timeout_ms must be non-negative"
            )
        if self.translation.target_language not in ("vi", "en", "zh", "ko"):
            raise ValueError("target language must be vi, en, zh, or ko")
        if self.translation.source_language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("translation source language must be auto, vi, en, zh, or ko")
        if self.asr.language not in ("auto", "vi", "en", "zh", "ko"):
            raise ValueError("ASR language must be auto, vi, en, zh, or ko")
        if self.asr.update_interval <= 0:
            raise ValueError("ASR update_interval must be positive")
        if self.asr.sherpa.recognizer_mode != "online_transducer":
            raise ValueError(
                "asr.sherpa.recognizer_mode must be online_transducer"
            )
        if self.asr.sherpa.provider not in ("cpu", "cuda", "coreml"):
            raise ValueError("asr.sherpa.provider must be cpu, cuda, or coreml")
        if self.asr.sherpa.num_threads < 1:
            raise ValueError("asr.sherpa.num_threads must be at least 1")
        if self.asr.sherpa.decoding_method not in (
            "greedy_search",
            "modified_beam_search",
        ):
            raise ValueError(
                "asr.sherpa.decoding_method must be greedy_search or "
                "modified_beam_search"
            )
        if self.asr.sherpa.max_active_paths < 1:
            raise ValueError("asr.sherpa.max_active_paths must be at least 1")
        if not 0 <= self.asr.sherpa.final_padding_ms <= 2_000:
            raise ValueError(
                "asr.sherpa.final_padding_ms must be between 0 and 2000"
            )
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
        if self.terminology.enabled and not self.terminology.bundle_path:
            raise ValueError(
                "terminology.bundle_path is required when terminology is enabled"
            )
        if self.terminology.matching.normalization != "unicode_nfc":
            raise ValueError(
                "terminology.matching.normalization must be unicode_nfc"
            )
        if not self.terminology.matching.longest_match_first:
            raise ValueError(
                "terminology.matching.longest_match_first must be true"
            )
        if self.terminology.asr.max_prompt_terms < 1:
            raise ValueError(
                "terminology.asr.max_prompt_terms must be positive"
            )
        if self.terminology.asr.max_prompt_tokens < 1:
            raise ValueError(
                "terminology.asr.max_prompt_tokens must be positive"
            )
        if self.terminology.asr.max_hotword_terms < 1:
            raise ValueError(
                "terminology.asr.max_hotword_terms must be positive"
            )
        if self.terminology.asr.max_hotword_tokens < 1:
            raise ValueError(
                "terminology.asr.max_hotword_tokens must be positive"
            )
        if self.terminology.asr.hotword_score <= 0:
            raise ValueError(
                "terminology.asr.hotword_score must be positive"
            )
        mt = self.terminology.mt
        if mt.strategy != "placeholder_with_validation":
            raise ValueError(
                "terminology.mt.strategy must be placeholder_with_validation"
            )
        if not mt.placeholder_formats:
            raise ValueError(
                "terminology.mt.placeholder_formats must not be empty"
            )
        rendered: set[str] = set()
        for index, template in enumerate(mt.placeholder_formats):
            try:
                first = template.format(id=1)
                second = template.format(id=2)
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid terminology.mt.placeholder_formats[{index}]: {template!r}"
                ) from exc
            if first == second or any(character.isspace() for character in first):
                raise ValueError(
                    "terminology.mt placeholder formats must contain a unique id "
                    "and render without whitespace"
                )
            if first in rendered:
                raise ValueError(
                    "terminology.mt.placeholder_formats must be unique"
                )
            rendered.add(first)
        if mt.on_validation_error not in ("raise", "segment_fallback"):
            raise ValueError(
                "terminology.mt.on_validation_error must be raise or "
                "segment_fallback"
            )
        if self.terminology.enabled and not mt.validate_coverage:
            raise ValueError(
                "terminology.mt.validate_coverage must be true when terminology is enabled"
            )
        if self.terminology.enabled and not mt.pivot_canonicalization:
            raise ValueError(
                "terminology.mt.pivot_canonicalization must be true when terminology is enabled"
            )
        if self.terminology.tts.strategy != "spoken_form":
            raise ValueError(
                "terminology.tts.strategy must be spoken_form"
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


def _terminology_from_mapping(
    values: dict[str, Any] | None,
) -> TerminologyConfig:
    data = dict(values or {})
    matching = _from_mapping(
        TerminologyMatchingConfig,
        data.pop("matching", None),
    )
    asr = _from_mapping(
        TerminologyAsrConfig,
        data.pop("asr", None),
    )
    mt = _from_mapping(
        TerminologyMtConfig,
        data.pop("mt", None),
    )
    tts = _from_mapping(
        TerminologyTtsConfig,
        data.pop("tts", None),
    )
    config = _from_mapping(TerminologyConfig, data)
    config.matching = matching
    config.asr = asr
    config.mt = mt
    config.tts = tts
    return config


def _asr_from_mapping(values: dict[str, Any] | None) -> AsrConfig:
    data = dict(values or {})
    sherpa = _from_mapping(SherpaAsrConfig, data.pop("sherpa", None))
    config = _from_mapping(AsrConfig, data)
    config.sherpa = sherpa
    return config


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
        asr=_asr_from_mapping(data.get("asr")),
        commit=_from_mapping(CommitConfig, data.get("commit")),
        translation=_from_mapping(TranslationConfig, data.get("translation")),
        tts=_from_mapping(TtsConfig, data.get("tts")),
        terminology=_terminology_from_mapping(data.get("terminology")),
    )
    config.validate()
    return config
