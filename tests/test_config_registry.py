from pathlib import Path

import pytest

from onevoice.config import PipelineConfig, load_config
from onevoice.registry import BackendRegistry


def test_default_config() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.audio.sample_rate == 16_000
    assert config.asr.backend == "moonshine"
    assert config.asr.model == "auto"
    assert config.asr.language == "vi"
    assert config.translation.backend == "opus_ct2"
    assert config.translation.model == "opus-auto"
    assert config.translation.compute_type == "int8"
    assert config.commit.agreement_updates == 2
    assert config.commit.hold_tokens == 1
    assert config.commit.term_prefix_timeout_ms == 1500
    assert config.translation.wait_tokens == 6
    assert config.translation.update_tokens == 4
    assert config.translation.timeout_ms == 1200
    assert not config.translation.sentence_boundary_only
    assert config.vad.semantic_endpoint_enabled
    assert config.vad.semantic_endpoint_sentences == 2
    assert not config.tts.enabled
    assert config.tts.backend == "sherpa_onnx"
    assert config.tts.emission_mode == "final_utterance"
    assert config.tts.agreement_updates == 2
    assert not config.terminology.enabled
    assert config.terminology.bundle_path is None
    assert config.terminology.matching.normalization == "unicode_nfc"
    assert config.terminology.matching.longest_match_first
    assert config.terminology.matching.case_sensitive_for_codes
    assert config.terminology.asr.initial_prompt_enabled
    assert config.terminology.asr.post_correction_enabled
    assert config.terminology.asr.native_hotwords_enabled
    assert config.terminology.asr.max_prompt_terms == 32
    assert config.terminology.asr.max_prompt_tokens == 128
    assert config.terminology.asr.max_hotword_terms == 64
    assert config.terminology.asr.max_hotword_tokens == 256
    assert config.terminology.asr.hotword_score == 1.5
    assert config.terminology.mt.strategy == "placeholder_with_validation"
    assert len(config.terminology.mt.placeholder_formats) == 3
    assert config.terminology.mt.validate_coverage
    assert config.terminology.mt.pivot_canonicalization
    assert config.terminology.tts.strategy == "spoken_form"


def test_registry_reports_available_backends() -> None:
    registry = BackendRegistry()
    registry.register("asr", "demo", lambda value=1: value)
    assert registry.create("asr", "demo", value=2) == 2
    with pytest.raises(ValueError, match="available"):
        registry.create("asr", "missing")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: setattr(
                config.asr.sherpa, "recognizer_mode", "offline_transducer"
            ),
            "recognizer_mode",
        ),
        (lambda config: setattr(config.audio, "asr_chunk_ms", 0), "asr_chunk_ms"),
        (lambda config: setattr(config.audio, "queue_seconds", 0), "queue_seconds"),
        (lambda config: setattr(config.commit, "agreement_updates", 0), "agreement_updates"),
        (lambda config: setattr(config.commit, "hold_tokens", -1), "hold_tokens"),
        (
            lambda config: setattr(
                config.commit, "term_prefix_timeout_ms", -1
            ),
            "term_prefix_timeout_ms",
        ),
        (lambda config: setattr(config.translation, "wait_tokens", 0), "wait/update"),
        (lambda config: setattr(config.translation, "update_tokens", 0), "wait/update"),
        (lambda config: setattr(config.translation, "timeout_ms", 0), "timing"),
        (lambda config: setattr(config.translation, "max_new_tokens", 0), "max_new_tokens"),
        (lambda config: setattr(config.tts, "emission_mode", "unknown"), "emission_mode"),
        (
            lambda config: setattr(config.terminology, "enabled", True),
            "terminology.bundle_path",
        ),
        (
            lambda config: setattr(
                config.terminology.matching, "normalization", "unknown"
            ),
            "terminology.matching.normalization",
        ),
        (
            lambda config: setattr(
                config.terminology.matching, "longest_match_first", False
            ),
            "terminology.matching.longest_match_first",
        ),
        (
            lambda config: setattr(
                config.terminology.mt, "placeholder_formats", []
            ),
            "placeholder_formats",
        ),
        (
            lambda config: setattr(
                config.terminology.tts, "strategy", "unknown"
            ),
            "terminology.tts.strategy",
        ),
        (
            lambda config: setattr(
                config.terminology.asr, "max_prompt_terms", 0
            ),
            "max_prompt_terms",
        ),
        (
            lambda config: setattr(
                config.terminology.asr, "max_prompt_tokens", 0
            ),
            "max_prompt_tokens",
        ),
        (
            lambda config: setattr(
                config.terminology.asr, "max_hotword_terms", 0
            ),
            "max_hotword_terms",
        ),
        (
            lambda config: setattr(
                config.terminology.asr, "max_hotword_tokens", 0
            ),
            "max_hotword_tokens",
        ),
        (
            lambda config: setattr(
                config.terminology.asr, "hotword_score", 0
            ),
            "hotword_score",
        ),
    ],
)
def test_pipeline_config_rejects_invalid_runtime_limits(mutate, message: str) -> None:
    config = PipelineConfig()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_profile_deep_merges_over_default_yaml(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "vad:\n  semantic_endpoint_sentences: 1\n"
        "tts:\n  speed: 0.85\n",
        encoding="utf-8",
    )

    config = load_config(profile)

    assert config.vad.semantic_endpoint_sentences == 1
    assert config.tts.speed == 0.85
    assert config.asr.language == "vi"
    assert config.asr.sherpa.recognizer_mode == "online_transducer"
    assert config.asr.sherpa.final_padding_ms == 500
    assert config.asr.sherpa.cache_dir == ".cache/onevoice/asr"
    assert config.asr.model == "auto"
    assert config.asr.device == "cpu"
    assert config.asr.sherpa.num_threads == 2
    assert config.translation.model == "opus-auto"
    assert config.tts.backend == "sherpa_onnx"
    assert config.tts.cache_dir == ".cache/onevoice/tts"
    assert config.tts.max_chunk_tokens == 24
    assert not config.terminology.enabled
    assert config.terminology.matching.case_sensitive_for_codes


def test_profile_deep_merges_nested_terminology_settings(tmp_path: Path) -> None:
    profile = tmp_path / "terminology-profile.yaml"
    profile.write_text(
        "terminology:\n"
        "  enabled: true\n"
        "  bundle_path: assets/terminology/factory-sample-v1/terminology.yaml\n"
        "  domain: factory-safety\n"
        "  matching:\n"
        "    case_sensitive_for_codes: false\n"
        "  mt:\n"
        "    validate_order: true\n",
        encoding="utf-8",
    )

    config = load_config(profile)

    assert config.terminology.enabled
    assert config.terminology.domain == "factory-safety"
    assert not config.terminology.matching.case_sensitive_for_codes
    assert config.terminology.matching.normalization == "unicode_nfc"
    assert config.terminology.matching.longest_match_first
    assert config.terminology.mt.validate_order
    assert config.terminology.mt.validate_coverage


@pytest.mark.parametrize(
    ("profile_name", "sentences"),
    [
        ("realtime_conversation.yaml", 1),
        ("continuous_speech.yaml", 2),
        ("stable_demo.yaml", 2),
    ],
)
def test_shipped_profiles_deep_merge_default_model_fields(
    profile_name: str, sentences: int
) -> None:
    config = load_config(Path("config") / profile_name)
    assert config.vad.semantic_endpoint_sentences == sentences
    assert config.asr.model == "auto"
    assert config.asr.language == "vi"
    assert config.asr.sherpa.punctuation_enabled
    assert config.translation.model == "opus-auto"
    assert config.tts.cache_dir == ".cache/onevoice/tts"


def test_realtime_profile_enables_stable_sentence_tts_without_changing_mt_policy() -> None:
    config = load_config(Path("config/realtime_conversation.yaml"))

    assert config.tts.emission_mode == "stable_sentence"
    assert not config.tts.final_only
    assert config.tts.sentence_boundary_only
    assert config.tts.agreement_updates == 2
    assert config.translation.wait_tokens == 6
    assert config.translation.update_tokens == 4
    assert config.translation.timeout_ms == 1200
    assert config.translation.min_request_interval_ms == 500
    assert not config.translation.sentence_boundary_only
