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


def test_registry_reports_available_backends() -> None:
    registry = BackendRegistry()
    registry.register("asr", "demo", lambda value=1: value)
    assert registry.create("asr", "demo", value=2) == 2
    with pytest.raises(ValueError, match="available"):
        registry.create("asr", "missing")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: setattr(config.audio, "asr_chunk_ms", 0), "asr_chunk_ms"),
        (lambda config: setattr(config.audio, "queue_seconds", 0), "queue_seconds"),
        (lambda config: setattr(config.commit, "agreement_updates", 0), "agreement_updates"),
        (lambda config: setattr(config.commit, "hold_tokens", -1), "hold_tokens"),
        (lambda config: setattr(config.translation, "wait_tokens", 0), "wait/update"),
        (lambda config: setattr(config.translation, "update_tokens", 0), "wait/update"),
        (lambda config: setattr(config.translation, "timeout_ms", 0), "timing"),
        (lambda config: setattr(config.translation, "max_new_tokens", 0), "max_new_tokens"),
        (lambda config: setattr(config.tts, "emission_mode", "unknown"), "emission_mode"),
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
    assert config.asr.model == "auto"
    assert config.asr.device == "cpu"
    assert config.translation.model == "opus-auto"
    assert config.tts.backend == "sherpa_onnx"
    assert config.tts.cache_dir == ".cache/onevoice/tts"
    assert config.tts.max_chunk_tokens == 24


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
