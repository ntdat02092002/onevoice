from __future__ import annotations

from types import SimpleNamespace

import pytest

from onevoice.backends.translation import OpusMtCTranslate2Backend, _OpusEngine
from onevoice.config import TranslationConfig
from onevoice.models import TranslationRequest


def test_opus_routes_all_twelve_directions() -> None:
    languages = ("vi", "en", "zh", "ko")
    routes = {
        (source, target): OpusMtCTranslate2Backend.route(source, target)
        for source in languages
        for target in languages
        if source != target
    }
    assert len(routes) == 12
    assert routes[("vi", "en")] == (("vi", "en"),)
    assert routes[("vi", "zh")] == (("vi", "en"), ("en", "zh"))
    assert routes[("ko", "vi")] == (("ko", "en"), ("en", "vi"))


def test_opus_pivot_preserves_request_metadata() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig(backend="opus_ct2"))
    calls: list[tuple[str, tuple[str, str]]] = []
    backend._ensure_route = lambda source, target: ((source, "en"), ("en", target))

    def translate_once(text: str, pair: tuple[str, str]) -> str:
        calls.append((text, pair))
        return f"{text}:{pair[1]}"

    backend._translate_once = translate_once
    request = TranslationRequest(
        text="xin chào",
        source_language="vi",
        target_language="ko",
        source_revision=7,
        is_final=True,
    )
    update = backend.translate(request)

    assert calls == [
        ("xin chào", ("vi", "en")),
        ("xin chào:en", ("en", "ko")),
    ]
    assert update.text == "xin chào:en:ko"
    assert update.source_revision == 7
    assert update.is_final


class _Tokenizer:
    supported_language_codes = [">>vie<<"]

    def __init__(self) -> None:
        self.encoded = ""

    def encode(self, text: str):
        self.encoded = text
        return [1, 2]

    def convert_ids_to_tokens(self, values):
        return [f"t{value}" for value in values]

    def convert_tokens_to_ids(self, values):
        return [3]

    def decode(self, values, skip_special_tokens: bool):
        assert skip_special_tokens
        return "xin chào"


class _Translator:
    def __init__(self) -> None:
        self.kwargs = {}

    def translate_batch(self, sources, **kwargs):
        self.kwargs = kwargs
        assert sources == [["t1", "t2"]]
        return [SimpleNamespace(hypotheses=[["result"]])]


def test_opus_uses_language_prefix_and_greedy_decode() -> None:
    backend = OpusMtCTranslate2Backend(TranslationConfig(max_new_tokens=64))
    tokenizer = _Tokenizer()
    translator = _Translator()
    backend._engines[("en", "vi")] = _OpusEngine(tokenizer, translator)

    assert backend._translate_once("hello", ("en", "vi")) == "xin chào"
    assert tokenizer.encoded == ">>vie<< hello"
    assert translator.kwargs["beam_size"] == 1
    assert translator.kwargs["max_decoding_length"] == 64


def test_translation_capabilities_fail_before_model_loading() -> None:
    with pytest.raises(ValueError, match="source and target must differ"):
        OpusMtCTranslate2Backend(
            TranslationConfig(
                backend="opus_ct2",
                model="opus-auto",
                source_language="vi",
                target_language="vi",
            )
        )
    with pytest.raises(ValueError, match="model must be 'opus-auto'"):
        OpusMtCTranslate2Backend(
            TranslationConfig(backend="opus_ct2", model="some-random-model")
        )
