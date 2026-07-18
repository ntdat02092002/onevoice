from pathlib import Path

import pytest

from onevoice.config import load_config
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
    assert config.vad.semantic_endpoint_enabled
    assert config.vad.semantic_endpoint_sentences == 2


def test_registry_reports_available_backends() -> None:
    registry = BackendRegistry()
    registry.register("asr", "demo", lambda value=1: value)
    assert registry.create("asr", "demo", value=2) == 2
    with pytest.raises(ValueError, match="available"):
        registry.create("asr", "missing")
