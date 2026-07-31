from pathlib import Path

from onevoice.backends.asr import MOONSHINE_MODELS_BY_LANGUAGE
from onevoice.backends.translation import OPUS_PAIR_MODELS
from onevoice.registry import registry
from onevoice.sherpa_models import SHERPA_STREAMING_MODELS


def test_model_zoo_covers_registered_backends_and_model_routes() -> None:
    text = Path("docs/MODEL_ZOO.md").read_text(encoding="utf-8")

    for kind in ("asr", "translation", "tts", "vad", "preprocessor", "commit"):
        for backend in registry.names(kind):
            assert f"`{backend}`" in text
    for language, models in MOONSHINE_MODELS_BY_LANGUAGE.items():
        assert f"`{language}`" in text
        for model in models:
            assert f"`{model}`" in text
    for model_id in OPUS_PAIR_MODELS.values():
        assert f"`{model_id}`" in text
    for model_id in SHERPA_STREAMING_MODELS:
        assert f"`{model_id}`" in text
    for terminology_capability in (
        "initial_prompt",
        "native hotwords",
        "Exact-alias correction",
    ):
        assert terminology_capability in text


def test_readme_documents_terminology_start_stop_lifecycle() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "--terminology-bundle" in text
    assert "--terminology-domain" in text
    assert "Terminology preflight" in text
    assert "Stop pipeline" in text
